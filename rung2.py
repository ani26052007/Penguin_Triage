"""Rung 2: triage the ambiguous middle (plan.md Section 4).

One big-model call per incident, and the only place a model judges a cause. It reads the
router's output from the RE-RUN on the patched code (3.2.11), never raw video, and returns

    code      -> back to the repair agent with the new evidence
    data_gap  -> hand to the vision data pipeline, ONLY if the gate below passes
    unclear   -> stop; the incident is reported unresolved

The gate is a pure function of the router verdict (``data_gap_allowed``): a "data_gap" reply
must match a router one-factor or all-factors restore (decision rows 6 and 7), otherwise it
is rejected and the result is ``unclear``. The model cannot open a vision route on its own,
and nothing a reply says can change the verdict it was shown.

Structural rule (see tests/test_ladder_structure.py): this module is on the LLM side. It
does not import the decision code (``router.decide``, ``router.router``, ``router.replay``,
the stages or branches); the loop hands it a plain ``Rung2Input``.

Failure handling: a provider error, two bad replies or no model at all give ``unclear``
with the reason; this module never raises for model trouble and never invents a label.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Optional

from router import vocab as V
from router.llm import LLMFailure, LLMOutputError, Model, query_json
from router.report import verdict_text
from router.schemas import Evidence, ReplayResult, Verdict

log = logging.getLogger("rung2")

DATA_GAP_ROWS = (6, 7)


class Rung2Label(str, Enum):
    CODE = "code"
    DATA_GAP = "data_gap"
    UNCLEAR = "unclear"


def data_gap_allowed(verdict: Verdict) -> tuple[bool, str]:
    """May a "data gap" claim stand? Only when the router re-run found a one-factor or
    all-factors restore (rows 6-7). Pure; no model involved."""
    if verdict.row not in DATA_GAP_ROWS:
        return False, (f"the router re-run verdict is row {verdict.row} ({verdict.label.value}), "
                       f"not a restore verdict (rows 6-7)")
    if not any(r.replay_id == V.R_C_ALL or r.replay_id.startswith("C_restore:") for r in verdict.replays):
        return False, "no one-factor or all-factors restore is attached to the router verdict"
    return True, ""


@dataclass(frozen=True)
class Rung2Input:
    """Structured evidence for the call. ``extra`` carries optional text blocks the real system
    can fill (per-subtask progress, action-trace statistics); key frames are not included.
    TODO(verify): where those come from (behavior analysis / Stage 1 traces) once wired in."""
    incident_id: str
    verdict: Verdict                          # the router re-run verdict on the patched code
    replays: tuple[ReplayResult, ...]         # the re-run's full replay table
    evidence: tuple[Evidence, ...]
    rung1: str                                # Rung1Result.compact()
    extra: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Rung2Config:
    retries: int = 1                          # one retry for a malformed reply only
    max_reason_chars: int = 600


@dataclass(frozen=True)
class Rung2Result:
    label: Rung2Label                         # final, after the gate
    model_label: Optional[Rung2Label]         # what the model said (None: no usable reply)
    accepted: bool                            # False when the gate rejected a data_gap claim, or no reply
    reason: str
    evidence_ids: tuple[str, ...] = ()
    rejected_reason: str = ""
    source: str = "model"                     # model | fallback
    calls: int = 0
    cost: float = 0.0
    tokens: int = 0

    def to_dict(self) -> dict:
        return {"label": self.label.value, "model_label": self.model_label.value if self.model_label else None,
                "accepted": self.accepted, "reason": self.reason, "evidence_ids": list(self.evidence_ids),
                "rejected_reason": self.rejected_reason, "source": self.source, "calls": self.calls,
                "cost": self.cost, "tokens": self.tokens}


SYSTEM_PROMPT = """You triage ONE incident for a robot-policy repair system. A code patch was applied and the
simulator check was ambiguous, so the router was re-run on the patched code. The router verdict and the
replay table are facts computed by code: do not contradict or restate them as your own finding.

Decide what the remaining problem is:
- "code": a simple code or config bug is still present; the repair agent should try again with the evidence.
- "data_gap": the policy lacks training data for a visual condition. This is only valid if the router
  re-run found a one-factor or all-factors restore (decision rows 6-7); otherwise your answer is rejected.
- "unclear": the evidence does not support either; the incident is reported unresolved.

Reply with exactly one JSON object and nothing else:
{"label": "code" | "data_gap" | "unclear", "reason": "<=600 chars, plain text", "evidence_ids": ["<ids from the Evidence list>"]}
Rules: no other keys; cite only evidence ids that appear in the list; prefer "unclear" over guessing."""


def build_messages(inp: Rung2Input) -> list[dict]:
    table = [f"- {r.replay_id}: {r.outcome.value if r.outcome else 'unclassified'} ({r.successes}/{r.n} succeeded)"
             for r in inp.replays] or ["- (none)"]
    ev = [f"- {e.id} ({e.signal}, strength {e.strength:.2f})" + (f" @ {e.location}" if e.location else "")
          + (f": {e.detail}" if e.detail else "")
          for e in sorted(inp.evidence, key=lambda e: (-e.strength, e.id))] or ["- (none)"]
    parts = [f"## Incident\n{inp.incident_id}",
             f"## Router re-run verdict (final)\n{verdict_text(inp.verdict)}\nRouter notes: {inp.verdict.notes or '-'}",
             "## Replay table (re-run)\n" + "\n".join(table),
             "## Evidence (cite by id)\n" + "\n".join(ev),
             f"## Rung 1 result for the patch\n{inp.rung1}"]
    parts += [f"## {k}\n{v}" for k, v in inp.extra.items()]
    parts.append("Reply with the JSON object now.")
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "\n\n".join(parts)}]


@dataclass(frozen=True)
class _Parsed:
    label: Rung2Label
    reason: str
    evidence_ids: tuple[str, ...]


class _Validator:
    KEYS = {"label", "reason", "evidence_ids"}

    def __init__(self, evidence_ids: set[str], cfg: Rung2Config):
        self.ids, self.cfg = evidence_ids, cfg

    def __call__(self, obj: dict) -> _Parsed:
        if set(obj) != self.KEYS:
            raise LLMOutputError(f"reply keys must be exactly {sorted(self.KEYS)}, got {sorted(obj)}")
        try:
            label = Rung2Label(obj["label"])
        except ValueError:
            raise LLMOutputError(f"label must be one of {[l.value for l in Rung2Label]}, got {obj['label']!r}") from None
        reason = obj["reason"]
        if not isinstance(reason, str) or not reason.strip():
            raise LLMOutputError("reason must be a non-empty string")
        if len(reason) > self.cfg.max_reason_chars:
            raise LLMOutputError(f"reason is {len(reason)} chars; the limit is {self.cfg.max_reason_chars}")
        raw = obj["evidence_ids"]
        if not isinstance(raw, list) or not all(isinstance(i, str) for i in raw):
            raise LLMOutputError("evidence_ids must be a list of strings")
        unknown = [i for i in raw if i not in self.ids]
        if unknown:
            raise LLMOutputError(f"evidence ids not in the Evidence list: {unknown}")
        return _Parsed(label, reason.strip(), tuple(dict.fromkeys(raw)))


def triage(inp: Rung2Input, model: Optional[Model], cfg: Rung2Config = Rung2Config()) -> Rung2Result:
    """Run the call and apply the gate. Never raises for a bad reply or a provider error."""
    if model is None:
        return _log(inp, Rung2Result(Rung2Label.UNCLEAR, None, False, "no model configured", source="fallback"))
    validator = _Validator({e.id for e in inp.evidence}, cfg)
    try:
        q = query_json(model, build_messages(inp), validator, retries=cfg.retries)
    except LLMFailure as e:
        return _log(inp, Rung2Result(
            Rung2Label.UNCLEAR, None, False, f"no usable reply: {e}", source="fallback", calls=len(e.replies),
            cost=sum(r.cost for r in e.replies), tokens=sum(r.input_tokens + r.output_tokens for r in e.replies)))
    p: _Parsed = q.value
    label, accepted, rejected = p.label, True, ""
    if p.label is Rung2Label.DATA_GAP:
        ok, why = data_gap_allowed(inp.verdict)
        if not ok:
            label, accepted, rejected = Rung2Label.UNCLEAR, False, f"data_gap rejected: {why}"
    return _log(inp, Rung2Result(label, p.label, accepted, p.reason, p.evidence_ids, rejected, "model",
                                 len(q.replies), q.cost, q.tokens))


def _log(inp: Rung2Input, res: Rung2Result) -> Rung2Result:
    log.info("rung2 incident=%s label=%s model_label=%s accepted=%s calls=%d reason=%s%s", inp.incident_id,
             res.label.value, res.model_label.value if res.model_label else "-", res.accepted, res.calls,
             res.reason, f" [{res.rejected_reason}]" if res.rejected_reason else "")
    return res


__all__ = ["DATA_GAP_ROWS", "Rung2Config", "Rung2Input", "Rung2Label", "Rung2Result", "build_messages",
           "data_gap_allowed", "triage"]
