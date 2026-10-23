"""LLM plumbing for the report layer (plan.md 3.2.12).

Reuses hagent's client: ``Model`` (``query(messages) -> Reply``),
``OpenAICompatModel`` and ``ScriptedModel`` come from ``hagent.models``, and a
model endpoint is described by ``hagent.config.ModelConfig``. Nothing is
re-implemented here except what the router needs on top:

* ``query_json``: ask for one JSON object, validate it with a caller-supplied
  function, and on a bad reply retry **once** with the rejection reason in the
  conversation. Two bad replies raise ``LLMRetriesExhausted``; the caller picks
  its fallback. A provider failure (``ModelError``) raises ``LLMCallFailed`` and
  is not retried here, since ``OpenAICompatModel`` already retries transient
  HTTP errors. Both are ``LLMFailure`` and keep the replies received so far.
* ``load_router_models``: the router's own TOML (``[report]``, ``[mapper]``,
  ``[factor_search]``, each a ``ModelConfig``). hagent's ``settings_from_dict``
  rejects unknown sections, so these tables cannot live in hagent's config file.

No model output in the router ever picks a route; see ``router.report``.
"""

from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

from hagent.config import ModelConfig
from hagent.exceptions import ModelError
from hagent.models import Model, OpenAICompatModel, Reply

T = TypeVar("T")

_FENCE = re.compile(r"^```(?:json)?\s*\n(.*?)\n```$", re.DOTALL)


class LLMOutputError(ValueError):
    """A model reply that cannot be used: not JSON, wrong shape, or it cites
    something that does not exist. The message goes back to the model on retry."""


class LLMFailure(RuntimeError):
    """No usable reply. ``errors``: why earlier replies were rejected;
    ``replies``: every reply received (for token and cost accounting)."""

    def __init__(self, message: str, errors: tuple[str, ...], replies: tuple[Reply, ...]):
        super().__init__(message)
        self.errors = errors
        self.replies = replies


class LLMRetriesExhausted(LLMFailure):
    """Every reply, the retry included, was rejected."""

    def __init__(self, errors: tuple[str, ...], replies: tuple[Reply, ...]):
        super().__init__("model output rejected after retry: " + " | ".join(errors), errors, replies)


class LLMCallFailed(LLMFailure, ModelError):
    """The provider failed (``ModelError``) on the first call or on the retry."""

    def __init__(self, cause: ModelError, errors: tuple[str, ...], replies: tuple[Reply, ...]):
        super().__init__(f"model error: {cause}", errors, replies)
        self.cause = cause


def extract_json_object(text: str) -> dict:
    """The reply must be one JSON object, bare or in a single ```json fence.
    Prose around it is rejected: strict means strict."""
    body = text.strip()
    m = _FENCE.match(body)
    if m:
        body = m.group(1).strip()
    try:
        obj = json.loads(body)
    except json.JSONDecodeError as e:
        raise LLMOutputError(f"reply is not valid JSON ({e.msg} at char {e.pos}); "
                             f"reply with one JSON object and nothing else") from None
    if not isinstance(obj, dict):
        raise LLMOutputError(f"reply must be a JSON object, got {type(obj).__name__}")
    return obj


@dataclass(frozen=True)
class JsonQuery:
    value: Any
    attempts: int
    errors: tuple[str, ...]                 # why earlier attempts were rejected
    replies: tuple[Reply, ...]

    @property
    def cost(self) -> float:
        return sum(r.cost for r in self.replies)

    @property
    def tokens(self) -> int:
        return sum(r.input_tokens + r.output_tokens for r in self.replies)


def query_json(model: Model, messages: list[dict], parse: Callable[[dict], T], *, retries: int = 1) -> JsonQuery:
    """Query ``model`` for a JSON object and validate it with ``parse`` (which
    raises ``LLMOutputError``). At most ``1 + retries`` calls."""
    convo = list(messages)
    errors: list[str] = []
    replies: list[Reply] = []
    for attempt in range(1, retries + 2):
        try:
            reply = model.query(convo)
        except ModelError as e:
            raise LLMCallFailed(e, tuple(errors), tuple(replies)) from e
        replies.append(reply)
        try:
            return JsonQuery(parse(extract_json_object(reply.content)), attempt, tuple(errors), tuple(replies))
        except LLMOutputError as e:
            errors.append(str(e))
            convo = convo + [
                {"role": "assistant", "content": reply.content},
                {"role": "user", "content": f"Your reply was rejected: {e}\nReply again with ONLY the corrected "
                                            f"JSON object, following the same rules."}]
    raise LLMRetriesExhausted(tuple(errors), tuple(replies))


# --------------------------------------------------------------------------- #
# router model config
# --------------------------------------------------------------------------- #

@dataclass
class RouterModels:
    """Endpoints for the three LLM jobs. Each is optional: a job whose table is
    absent is simply not configured (``require`` says which one is missing)."""
    report: Optional[ModelConfig] = None          # Nemotron 3 Super 120B (3.2.12)
    mapper: Optional[ModelConfig] = None          # Nemotron 3 Nano 30B
    factor_search: Optional[ModelConfig] = None   # Cosmos3-Super-Reasoner, row 9 only

    def require(self, job: str) -> ModelConfig:
        cfg = getattr(self, job, None)
        if cfg is None:
            raise ValueError(f"no [{job}] table in the router model config")
        return cfg


def router_models_from_dict(data: dict[str, Any]) -> RouterModels:
    known = {f.name for f in fields(RouterModels)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown top-level section(s): {sorted(unknown)}. Valid: {sorted(known)}")
    mc_known = {f.name for f in fields(ModelConfig)}
    out = {}
    for name, table in data.items():
        bad = set(table) - mc_known
        if bad:
            raise ValueError(f"Unknown key(s) in [{name}]: {sorted(bad)}. Valid keys: {sorted(mc_known)}")
        out[name] = ModelConfig(**table)
    return RouterModels(**out)


def load_router_models(path: str | Path) -> RouterModels:
    return router_models_from_dict(tomllib.loads(Path(path).read_text()))


def build_model(cfg: ModelConfig, name: str) -> Model:
    return OpenAICompatModel(cfg, name=name)


__all__ = ["JsonQuery", "LLMCallFailed", "LLMFailure", "LLMOutputError", "LLMRetriesExhausted", "Model", "ModelConfig", "ModelError", "Reply",
           "RouterModels", "build_model", "extract_json_object", "load_router_models", "query_json",
           "router_models_from_dict"]
