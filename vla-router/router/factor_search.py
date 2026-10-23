"""Optional factor search for verdict row 9 (plan.md 3.2.12).

Row 9 is "perception unresolved": nothing restored, and the implied target is
wrong. The plan lets a vision model look at nominal vs failing frames and
propose visual factors the scene spec does not yet list:

    "Show it nominal vs failing policy-input frames (after preprocessing) and
     ask what visually differs. Each answer becomes a new factor to restore,
     and the replay decides."

This module is the interface and a stub client for that call, nothing more.

    VisionFactorSearch(model).propose(verdict, nominal_frames, failing_frames, tested_factors)
        -> FactorProposals

* Only row 9. Any other verdict raises ``ValueError``.
* A proposal is a hypothesis. It never changes the verdict and never reaches
  ``decide``. The caller adds each candidate to the scene spec and replays it
  through the one-factor restores (C2 to Ck); only a replay can move the
  verdict out of "perception unresolved". Running those replays is not done here.
* A bad reply (not JSON, wrong shape, a factor that is already tested, a
  duplicate) is rejected, retried once, then the result is *empty*. There is no
  fallback list: a made-up factor would cost real replays.
* Frames are references (URL or data URI) to policy-input images. Loading them
  from the rollout logs is the caller's job.

TODO(verify): everything about the endpoint. Whether Token Factory's
Cosmos3-Super-Reasoner accepts video or only images (the model card lists text,
image and video; mp4 at about 4 fps recommended, sampled frames are the
fallback); whether it accepts OpenAI-style ``image_url`` content parts, which is
what this stub sends; the model id and prices; and whether
"Nemotron-Nano-V2-12b" is the vision-language variant, which decides if it can
serve as the cheap second opinion. None of that was checked.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional, Sequence

from hagent.models import Model

from router.llm import LLMFailure, LLMOutputError, query_json
from router.schemas import Verdict

log = logging.getLogger("router.factor_search")

ROW_PERCEPTION_UNRESOLVED = 9
_NAME = re.compile(r"^[a-z][a-z0-9_]*(\([a-z0-9_]+\))?$")     # e.g. "lighting", "material(mug)"


@dataclass(frozen=True)
class FrameRef:
    """One policy-input frame (after preprocessing). ``uri``: an image URL or a
    ``data:image/...;base64,`` URI."""
    label: str
    uri: str

    def __post_init__(self) -> None:
        if not self.uri:
            raise ValueError("a frame needs a uri")


@dataclass(frozen=True)
class CandidateFactor:
    name: str                 # scene-spec factor name, e.g. "lighting" or "material(mug)"
    description: str          # what visibly differs, in the model's words


@dataclass(frozen=True)
class FactorProposals:
    """``source`` is "model" or "none" (no valid proposal; ``reason`` says why).
    Candidates are unconfirmed until a replay restores success."""
    candidates: tuple[CandidateFactor, ...]
    source: str
    attempts: int = 0
    reason: str = ""
    cost: float = 0.0
    tokens: int = 0


SYSTEM_PROMPT = """You compare images from a robot policy's camera. A manipulation policy succeeds in the nominal \
scene and fails in another one. Both sets of images are what the policy actually receives (after preprocessing).

Say what VISUALLY differs between the nominal images and the failing images, as candidate scene factors that \
could be changed back to nominal in a simulator (lighting, an object's material or colour, a texture, a \
background, camera exposure, clutter, and so on).

Reply with ONE JSON object and nothing else:
{"factors": [{"name": "snake_case_name or name(object)", "description": "what differs, one sentence"}]}

Rules:
- At most {max_candidates} factors, most likely first. An empty list is a valid answer.
- Do not repeat a factor from the already-tested list.
- Describe only what you can see in the images."""


class VisionFactorSearch:
    def __init__(self, model: Model, *, max_candidates: int = 3, max_frames_per_side: int = 4) -> None:
        if max_candidates < 1 or max_frames_per_side < 1:
            raise ValueError("max_candidates and max_frames_per_side must be >= 1")
        self.model, self.max_candidates, self.max_frames = model, max_candidates, max_frames_per_side

    # -- request ----------------------------------------------------------- #
    def _messages(self, nominal: Sequence[FrameRef], failing: Sequence[FrameRef], tested: Sequence[str]) -> list[dict]:
        content: list[dict] = [{"type": "text", "text": "NOMINAL frames (the policy succeeds):"}]
        for f in nominal[: self.max_frames]:
            content += [{"type": "text", "text": f"nominal: {f.label}"},
                        {"type": "image_url", "image_url": {"url": f.uri}}]
        content.append({"type": "text", "text": "FAILING frames (the policy fails):"})
        for f in failing[: self.max_frames]:
            content += [{"type": "text", "text": f"failing: {f.label}"},
                        {"type": "image_url", "image_url": {"url": f.uri}}]
        content.append({"type": "text", "text": "Already-tested factors (do not repeat): "
                                                + (", ".join(tested) or "(none)") + "\nReturn the JSON now."})
        return [{"role": "system", "content": SYSTEM_PROMPT.replace("{max_candidates}", str(self.max_candidates))},
                {"role": "user", "content": content}]

    # -- reply ------------------------------------------------------------- #
    def _parse(self, obj: dict, tested: frozenset[str]) -> tuple[CandidateFactor, ...]:
        if set(obj) != {"factors"} or not isinstance(obj["factors"], list):
            raise LLMOutputError("the reply must be exactly {\"factors\": [...]}")
        if len(obj["factors"]) > self.max_candidates:
            raise LLMOutputError(f"at most {self.max_candidates} factors, got {len(obj['factors'])}")
        out: list[CandidateFactor] = []
        for i, f in enumerate(obj["factors"]):
            if not isinstance(f, dict) or set(f) != {"name", "description"}:
                raise LLMOutputError(f"factors[{i}] must be an object with exactly 'name' and 'description'")
            name, desc = f["name"], f["description"]
            if not isinstance(name, str) or not _NAME.match(name):
                raise LLMOutputError(f"factors[{i}].name {name!r} must be snake_case, optionally 'name(object)'")
            if not isinstance(desc, str) or not desc.strip():
                raise LLMOutputError(f"factors[{i}].description must be a non-empty string")
            if name in tested:
                raise LLMOutputError(f"'{name}' was already tested; propose something else")
            if any(c.name == name for c in out):
                raise LLMOutputError(f"'{name}' is listed twice")
            out.append(CandidateFactor(name, desc.strip()))
        return tuple(out)

    # -- public ------------------------------------------------------------ #
    def propose(self, verdict: Verdict, nominal_frames: Sequence[FrameRef], failing_frames: Sequence[FrameRef],
                tested_factors: Sequence[str] = ()) -> FactorProposals:
        if verdict.row != ROW_PERCEPTION_UNRESOLVED:
            raise ValueError(f"factor search is only for verdict row 9, got row {verdict.row}")
        if not nominal_frames or not failing_frames:
            raise ValueError("factor search needs at least one nominal and one failing frame")
        tested = frozenset(tested_factors)
        try:
            q = query_json(self.model, self._messages(nominal_frames, failing_frames, tested_factors),
                           lambda obj: self._parse(obj, tested))
        except LLMFailure as e:
            log.warning("factor search gave no proposal: %s", e)
            return FactorProposals((), "none", len(e.replies), str(e),
                                   sum(r.cost for r in e.replies),
                                   sum(r.input_tokens + r.output_tokens for r in e.replies))
        return FactorProposals(q.value, "model", q.attempts, "", q.cost, q.tokens)
