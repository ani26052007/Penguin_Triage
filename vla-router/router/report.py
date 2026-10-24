"""Report writer and hagent prompt builder (plan.md 3.2.11, 3.2.12, 3.2.13).

    write_report(verdict, evidence, model, *, repo, context) -> Report

A model (Nemotron 3 Super in the plan) reads the verdict and the evidence list
and returns strict JSON:

    {"summary":      "...",                          # for the human reader
     "evidence_ids": ["..."],                        # evidence the summary rests on
     "task":         "...",                          # what hagent should do, in plain words
     "localization": [{"kind": "file|symbol|config_key", "name": "...",
                       "file": "...",                # optional, symbol only
                       "evidence_ids": ["..."], "reason": "..."}]}   # ranked, best first

Nothing a model says can change a route. The verdict is an input; the report
only holds it. What is enforced, in code, on every reply:

* **JSON**: one object, exactly these keys. A key such as ``route``, ``label``
  or ``destination`` is an unknown key and the reply is rejected.
* **Evidence ids**: every cited id must be in the evidence list.
* **Names**: a localization ``file`` must be a file of the repo (``RepoIndex``)
  or appear in the evidence; a ``symbol`` must be defined in the repo (in the
  stated file, if one is given) or appear in the evidence; a ``config_key`` must
  appear in the evidence. Path-like tokens in the prose (summary, task, reasons)
  are held to the same rule, and so are ids written in square brackets. Other
  identifiers in prose are not checked (they may be library calls), so the
  structured ``localization`` is the only place symbols are named.
* A reply that breaks any rule is rejected whole. The model gets the reason and
  one more try. A second bad reply, or a provider error, gives the **fallback**:
  the evidence ranked by strength, rendered by code, no model text in it.

The hagent prompt is the Step 1A template (``hagent.task_template``). The
verdict line, repro command, rollout summary and repo path are filled by code
from the verdict and ``ReportContext``; the model supplies only ``task``, and the
location block is built from its validated localization. The router verdict is
framed as a hypothesis by the template itself. No bug ledger, no clean commit
(3.3): the model sees only the verdict, the evidence, the replay table and the
repo index.

Evidence "appears in" means: a path-like token anywhere in ``location`` or
``detail`` (files), or an identifier-like token in ``location``, or a code-like
token (contains ``_``, ``.``, ``/``, a camelCase hump, or ends in ``()``) in
``detail`` (symbols and config keys). Plain words in a detail line do not count.

TODO(verify): the model id, endpoint and prompt wording are untested against the
real model; the prompt is written for a model that can follow "JSON only".
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence

from hagent.models import Model
from hagent.task_template import IncidentContext, TaskTemplate, load_task_template

from router.llm import LLMFailure, LLMOutputError, query_json
from router.schemas import Destination, Evidence, Verdict

log = logging.getLogger("router.report")

KINDS = ("file", "symbol", "config_key")
HAGENT_DESTINATIONS = (Destination.HAGENT, Destination.CONFIG_FIX)
_TOP_KEYS = frozenset({"summary", "evidence_ids", "task", "localization"})
_ITEM_KEYS = frozenset({"kind", "name", "file", "evidence_ids", "reason"})

_EXTS = ("py", "yaml", "yml", "json", "toml", "cfg", "ini", "sh", "cpp", "hpp", "cu", "h", "c", "xml", "urdf",
         "usd", "usda", "usdc", "md", "txt")
_PATH = re.compile(r"(?<![\w/.\-])((?:[\w.\-]+/)*[\w\-]+\.(?:" + "|".join(_EXTS) + r"))(?![\w])")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*(?:\(\))?")
_BRACKET_ID = re.compile(r"\[([A-Za-z][\w:\-]*(?:\.[\w:\-]+)+)\]")
_SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".mypy_cache", ".pytest_cache", "build", "dist"}


def _norm(path: str) -> str:
    p = path.strip().replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return p


# --------------------------------------------------------------------------- #
# the repo index
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class RepoIndex:
    """What exists in the controller/wrapper repo: file paths (relative, posix)
    and, for Python files, the names they define. ``Cls.method`` and ``method``
    are both listed for a method."""
    files: frozenset[str]
    symbols: Mapping[str, frozenset[str]] = field(default_factory=dict)
    root: str = ""

    @classmethod
    def from_mapping(cls, files: Mapping[str, Sequence[str]], root: str = "") -> "RepoIndex":
        return cls(frozenset(_norm(f) for f in files), {_norm(f): frozenset(s) for f, s in files.items()}, root)

    @classmethod
    def from_dir(cls, root: str | Path, extensions: Sequence[str] = _EXTS) -> "RepoIndex":
        base = Path(root)
        if not base.is_dir():
            raise FileNotFoundError(f"repo path {root} is not a directory")
        files: set[str] = set()
        symbols: dict[str, frozenset[str]] = {}
        for p in sorted(base.rglob("*")):
            rel = p.relative_to(base)
            if not p.is_file() or any(part in _SKIP_DIRS for part in rel.parts) or p.suffix[1:] not in extensions:
                continue
            key = rel.as_posix()
            files.add(key)
            if p.suffix == ".py":
                try:
                    symbols[key] = frozenset(_python_symbols(p.read_text(errors="replace")))
                except SyntaxError:
                    log.warning("could not parse %s; its symbols are not indexed", key)
        return cls(frozenset(files), symbols, str(base))

    def has_file(self, path: str) -> bool:
        return _norm(path) in self.files

    def has_symbol(self, name: str, file: Optional[str] = None) -> bool:
        if file is not None:
            return name in self.symbols.get(_norm(file), frozenset())
        return any(name in s for s in self.symbols.values())

    def unique_basename(self, path: str) -> bool:
        """True when ``path`` is a bare file name that matches exactly one repo file."""
        if "/" in path:
            return False
        return sum(f.rsplit("/", 1)[-1] == path for f in self.files) == 1

    def prompt_lines(self, max_chars: int = 12000) -> str:
        out, used = [], 0
        for f in sorted(self.files):
            all_syms = self.symbols.get(f, frozenset())
            bare_methods = {q.split(".", 1)[1] for q in all_syms if "." in q}
            syms = sorted(all_syms - bare_methods)       # show Cls.method, not method as well
            line = f"{f}: {', '.join(syms)}" if syms else f
            if used + len(line) + 1 > max_chars:
                out.append("(index truncated)")
                break
            out.append(line)
            used += len(line) + 1
        return "\n".join(out) or "(empty repo index)"


def _python_symbols(source: str) -> set[str]:
    out: set[str] = set()
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.add(node.name)
        elif isinstance(node, ast.ClassDef):
            out.add(node.name)
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out |= {sub.name, f"{node.name}.{sub.name}"}
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            out |= {t.id for t in targets if isinstance(t, ast.Name)}
    return out


# --------------------------------------------------------------------------- #
# what the evidence names
# --------------------------------------------------------------------------- #

def _code_like(tok: str) -> bool:
    return (tok.endswith("()") or any(c in tok for c in "_./")
            or re.search(r"[a-z][A-Z]", tok) is not None)


@dataclass(frozen=True)
class EvidenceNames:
    files: frozenset[str]
    symbols: frozenset[str]        # identifiers and dotted names, with and without the "()" and any dotted prefix


def evidence_names(evidence: Sequence[Evidence]) -> EvidenceNames:
    files: set[str] = set()
    syms: set[str] = set()
    for e in evidence:
        for text in (e.location or "", e.detail):
            files |= {_norm(m) for m in _PATH.findall(text)}
        for i, text in enumerate((e.location or "", e.detail)):
            for tok in _IDENT.findall(text):
                if i == 0 or _code_like(tok):
                    bare = tok[:-2] if tok.endswith("()") else tok
                    syms |= {bare, *bare.split(".")}
    return EvidenceNames(frozenset(files), frozenset(syms))


# --------------------------------------------------------------------------- #
# records
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class LocalizationItem:
    kind: str                       # file | symbol | config_key | evidence (fallback only)
    name: str
    evidence_ids: tuple[str, ...]
    reason: str = ""
    file: Optional[str] = None

    def to_dict(self) -> dict:
        return {"kind": self.kind, "name": self.name, "file": self.file,
                "evidence_ids": list(self.evidence_ids), "reason": self.reason}


@dataclass(frozen=True)
class ReportContext:
    """What the report needs besides the verdict and the evidence. The text
    fields are filled by the caller from the incident; none may carry the bug
    ledger or the clean commit (3.3)."""
    incident_id: str
    scene: str = ""
    repro_command: str = ""
    rollout_summary: str = ""
    repo_path: str = ""


@dataclass(frozen=True)
class ReportConfig:
    max_localization: int = 8
    max_summary_chars: int = 1500
    max_repo_chars: int = 12000
    retries: int = 1


@dataclass(frozen=True)
class Report:
    """``source`` is "model" or "fallback". ``hagent_prompt`` is set only when
    the verdict's destination is hagent or config_fix. The ``verdict`` is the
    one that was passed in, untouched. ``task`` is the repair task: the model's
    validated ``task``, or ``default_task`` when the model gave none or the
    report fell back. It is the only model text the repair agent's TASK gets;
    the verdict line and the hypothesis framing come from code and the
    template (DECISIONS.md, Fix F2a)."""
    verdict: Verdict
    summary: str
    evidence_ids: tuple[str, ...]
    localization: tuple[LocalizationItem, ...]
    hagent_prompt: Optional[str]
    source: str
    attempts: int = 0
    rejections: tuple[str, ...] = ()
    fallback_reason: str = ""
    cost: float = 0.0
    tokens: int = 0
    task: str = ""

    def to_dict(self) -> dict:
        return {"verdict": self.verdict.to_dict(), "summary": self.summary,
                "evidence_ids": list(self.evidence_ids),
                "localization": [i.to_dict() for i in self.localization], "hagent_prompt": self.hagent_prompt,
                "source": self.source, "attempts": self.attempts, "rejections": list(self.rejections),
                "fallback_reason": self.fallback_reason, "cost": self.cost, "tokens": self.tokens, "task": self.task}


# --------------------------------------------------------------------------- #
# deterministic text (no model)
# --------------------------------------------------------------------------- #

def verdict_text(v: Verdict) -> str:
    """The router verdict as the hagent template's ``ROUTER_VERDICT``. Built by code from the Verdict."""
    parts = [f"{v.label.value} (decision-table row {v.row}), destination {v.destination.value}"]
    if v.location:
        parts.append(f"suspected location: {v.location}")
    if v.factor:
        parts.append(f"factor: {v.factor}")
    if v.by_elimination:
        parts.append("reached by elimination only, no located cause (low confidence)")
    if v.likely_second_cause:
        parts.append("a second cause is likely")
    if v.multi_cause_flag:
        parts.append("Stage 2 was degraded: likely multi-cause")
    return "; ".join(parts)


_TASK_BY_ROW = {
    4: "Find and fix the bug in the controller/wrapper code that makes the policy fail.",
    5: "Align the eval render settings with the training render settings so the policy succeeds again.",
}


def default_task(v: Verdict, ctx: ReportContext) -> str:
    where = f" in scene {ctx.scene}" if ctx.scene else ""
    return (f"The VLA policy fails{where} (incident {ctx.incident_id}). "
            f"{_TASK_BY_ROW.get(v.row, 'Investigate the failure and fix its cause in the controller/wrapper code.')}")


def _ranked(evidence: Sequence[Evidence]) -> list[Evidence]:
    return sorted(evidence, key=lambda e: (-e.strength, e.id))


def _fmt_evidence(e: Evidence) -> str:
    loc = f" @ {e.location}" if e.location else ""
    return f"- {e.id} ({e.signal}, strength {e.strength:.2f}){loc}: {e.detail}".rstrip(": ")


def fallback_localization(evidence: Sequence[Evidence]) -> tuple[LocalizationItem, ...]:
    """Every evidence record, strongest first. No model text."""
    return tuple(LocalizationItem("evidence", e.location or e.id, (e.id,), e.detail) for e in _ranked(evidence))


def fallback_summary(v: Verdict, evidence: Sequence[Evidence]) -> str:
    top = ", ".join(f"{e.id} ({e.strength:.2f})" for e in _ranked(evidence)[:5]) or "none"
    return f"Router verdict: {verdict_text(v)}. Strongest evidence: {top}."


def _location_block(v: Verdict, loc: Sequence[LocalizationItem], by_id: Mapping[str, Evidence],
                    evidence: Sequence[Evidence]) -> str:
    lines: list[str] = []
    if v.location:
        lines.append(f"Router location: {v.location}")
    if loc:
        lines.append("Ranked candidates:")
        for i, it in enumerate(loc, start=1):
            at = f" in {it.file}" if it.file else ""
            why = f" - {it.reason}" if it.reason and it.kind != "evidence" else ""
            lines.append(f"{i}. [{it.kind}] {it.name}{at}{why} (evidence: {', '.join(it.evidence_ids)})")
    cited = list(dict.fromkeys(i for it in loc for i in it.evidence_ids if i in by_id))
    shown = [by_id[i] for i in cited] or _ranked(evidence)[:8]
    if shown:
        lines.append("Evidence:")
        lines += [_fmt_evidence(e) for e in shown]
    return "\n".join(lines)


def build_hagent_prompt(v: Verdict, ctx: ReportContext, task: str, localization: Sequence[LocalizationItem],
                        evidence: Sequence[Evidence], template: Optional[TaskTemplate] = None) -> str:
    """Render the Step 1A template. Everything except ``task`` and the
    localization comes from the verdict and the context."""
    by_id = {e.id: e for e in evidence}
    tpl = template or load_task_template()
    return tpl.render(IncidentContext(
        task=task, router_verdict=verdict_text(v), location_evidence=_location_block(v, localization, by_id, evidence),
        repro_command=ctx.repro_command, rollout_summary=ctx.rollout_summary, repo_path=ctx.repo_path))


# --------------------------------------------------------------------------- #
# validation of a model reply
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class _Parsed:
    summary: str
    evidence_ids: tuple[str, ...]
    task: str
    localization: tuple[LocalizationItem, ...]


class _Validator:
    def __init__(self, v: Verdict, evidence: Sequence[Evidence], repo: RepoIndex, ctx: ReportContext,
                 cfg: ReportConfig) -> None:
        self.v, self.repo, self.cfg = v, repo, cfg
        self.ids = {e.id for e in evidence}
        self.names = evidence_names(evidence)
        self.needs_task = v.destination in HAGENT_DESTINATIONS
        self.ctx_files = {_norm(m) for t in (ctx.repro_command, ctx.rollout_summary) for m in _PATH.findall(t)}
        self.has_evidence = bool(evidence)

    # -- pieces ----------------------------------------------------------- #
    def _text(self, obj: Mapping, key: str, *, required: bool, limit: Optional[int] = None) -> str:
        val = obj.get(key)
        if not isinstance(val, str):
            raise LLMOutputError(f"'{key}' must be a string")
        val = val.strip()
        if required and not val:
            raise LLMOutputError(f"'{key}' must not be empty")
        if limit and len(val) > limit:
            raise LLMOutputError(f"'{key}' is {len(val)} characters, the limit is {limit}")
        return val

    def _ids(self, raw: object, where: str, *, required: bool) -> tuple[str, ...]:
        if not isinstance(raw, list) or not all(isinstance(i, str) for i in raw):
            raise LLMOutputError(f"{where} must be a list of evidence id strings")
        unknown = [i for i in raw if i not in self.ids]
        if unknown:
            raise LLMOutputError(f"{where} cites evidence ids that do not exist: {unknown}. "
                                 f"Valid ids: {sorted(self.ids)}")
        if required and self.has_evidence and not raw:
            raise LLMOutputError(f"{where} must cite at least one evidence id")
        return tuple(dict.fromkeys(raw))

    def _file_ok(self, path: str) -> bool:
        return self.repo.has_file(path) or _norm(path) in self.names.files

    def _check_prose(self, where: str, text: str) -> None:
        for m in _PATH.findall(text):
            p = _norm(m)
            if not (self._file_ok(p) or p in self.ctx_files or self.repo.unique_basename(p)):
                raise LLMOutputError(f"{where} names the file '{p}', which is neither in the repo nor in the "
                                     f"evidence. Name only files from the repo index or the evidence.")
        for m in _BRACKET_ID.findall(text):
            if m not in self.ids:
                raise LLMOutputError(f"{where} cites the evidence id [{m}], which does not exist")

    def _item(self, raw: object, n: int) -> LocalizationItem:
        where = f"localization[{n}]"
        if not isinstance(raw, dict):
            raise LLMOutputError(f"{where} must be an object")
        extra = set(raw) - _ITEM_KEYS
        if extra or not {"kind", "name", "evidence_ids"} <= set(raw):
            raise LLMOutputError(f"{where} must have keys kind, name, evidence_ids, reason (and optional file); "
                                 f"unexpected: {sorted(extra)}")
        kind, name = raw["kind"], raw["name"]
        if kind not in KINDS:
            raise LLMOutputError(f"{where}.kind must be one of {list(KINDS)}, got {kind!r}")
        if not isinstance(name, str) or not name.strip():
            raise LLMOutputError(f"{where}.name must be a non-empty string")
        name = name.strip()
        file = raw.get("file")
        if file is not None and (kind != "symbol" or not isinstance(file, str)):
            raise LLMOutputError(f"{where}.file is only for kind 'symbol' and must be a string")
        if kind == "file":
            if not self._file_ok(name):
                raise LLMOutputError(f"{where} names the file '{name}', which is neither in the repo nor in the "
                                     f"evidence")
            name = _norm(name)
        elif kind == "symbol":
            in_evidence = name in self.names.symbols
            if file is not None:
                if not self._file_ok(file):
                    raise LLMOutputError(f"{where}.file '{file}' is neither in the repo nor in the evidence")
                if not (self.repo.has_symbol(name, file) or in_evidence):
                    raise LLMOutputError(f"{where} names the symbol '{name}', which '{file}' does not define "
                                         f"and the evidence does not mention")
            elif not (self.repo.has_symbol(name) or in_evidence):
                raise LLMOutputError(f"{where} names the symbol '{name}', which is neither defined in the repo "
                                     f"nor mentioned in the evidence")
        else:  # config_key
            if name not in self.names.symbols:
                raise LLMOutputError(f"{where} names the config key '{name}', which the evidence does not mention")
        reason = self._text(raw, "reason", required=False) if "reason" in raw else ""
        self._check_prose(f"{where}.reason", reason)
        return LocalizationItem(kind, name, self._ids(raw["evidence_ids"], f"{where}.evidence_ids", required=True),
                                reason, _norm(file) if file else None)

    # -- the whole reply -------------------------------------------------- #
    def __call__(self, obj: dict) -> _Parsed:
        extra, missing = set(obj) - _TOP_KEYS, _TOP_KEYS - set(obj)
        if extra or missing:
            raise LLMOutputError(f"reply keys must be exactly {sorted(_TOP_KEYS)}; unexpected: {sorted(extra)}, "
                                 f"missing: {sorted(missing)}. The route is decided by the router, not by you.")
        summary = self._text(obj, "summary", required=True, limit=self.cfg.max_summary_chars)
        task = self._text(obj, "task", required=self.needs_task)
        ids = self._ids(obj["evidence_ids"], "evidence_ids", required=True)
        loc_raw = obj["localization"]
        if not isinstance(loc_raw, list):
            raise LLMOutputError("'localization' must be a list")
        if len(loc_raw) > self.cfg.max_localization:
            raise LLMOutputError(f"'localization' has {len(loc_raw)} items, the limit is {self.cfg.max_localization}")
        items = tuple(self._item(r, i) for i, r in enumerate(loc_raw))
        dup = [k for k, n in _count((it.kind, it.name, it.file) for it in items).items() if n > 1]
        if dup:
            raise LLMOutputError(f"'localization' lists the same candidate twice: {dup}")
        self._check_prose("summary", summary)
        self._check_prose("task", task)
        return _Parsed(summary, ids, task, items)


def _count(xs) -> dict:
    out: dict = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return out


# --------------------------------------------------------------------------- #
# the model call
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = """You write the incident report for an automatic failure router. The router has ALREADY decided \
the verdict from replays and checks. You do not decide, change or second-guess the route; you explain it and \
point at where to look.

Reply with ONE JSON object and nothing else (no prose, no markdown), with exactly these keys:
{
  "summary": "2-5 sentences for a human reader: what failed, what the router concluded and why",
  "evidence_ids": ["ids of the evidence the summary rests on"],
  "task": "what the repair agent should do, in plain words, 1-3 sentences (empty string if the verdict is not a code or config fix)",
  "localization": [
    {"kind": "file" | "symbol" | "config_key", "name": "...", "file": "path, optional, symbol only",
     "evidence_ids": ["at least one"], "reason": "one sentence"}
  ]
}

Rules:
- Cite only evidence ids that appear in the evidence list below. Cite them exactly as written.
- Name only files and symbols that appear in the repo index below, or that the evidence text itself mentions. \
Use the full path from the index. Config keys must come from the evidence. Do not invent names.
- "localization" is ranked, most likely first, at most 8 items. Use an empty list if nothing points anywhere.
- Evidence strength is in [0, 1]; 0.5 means just past the firing threshold. Do not overstate weak evidence.
- If the verdict was reached by elimination, say that it is low confidence.
- Do not add any other keys."""


def build_messages(v: Verdict, evidence: Sequence[Evidence], repo: RepoIndex, ctx: ReportContext,
                   cfg: ReportConfig) -> list[dict]:
    replay_lines = [f"- {r.replay_id}: {r.outcome.value if r.outcome else 'unclassified'} "
                    f"({r.successes}/{r.n} succeeded)" for r in v.replays] or ["- (none attached to the verdict)"]
    user = "\n".join([
        f"## Incident\n{ctx.incident_id}" + (f" (scene {ctx.scene})" if ctx.scene else ""),
        f"## Router verdict (final; do not change it)\n{verdict_text(v)}\nRouter notes: {v.notes or '-'}",
        "## Replays that decided it\n" + "\n".join(replay_lines),
        "## Evidence (cite by id)\n" + ("\n".join(_fmt_evidence(e) for e in _ranked(evidence)) or "- (none)"),
        f"## Repo index (file: symbols)\n{repo.prompt_lines(cfg.max_repo_chars)}",
        "Write the JSON report now."])
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def write_report(verdict: Verdict, evidence: Sequence[Evidence], model: Optional[Model], *, repo: RepoIndex,
                 context: ReportContext, template: Optional[TaskTemplate] = None,
                 config: ReportConfig = ReportConfig()) -> Report:
    """Run the report writer. Never raises for a bad model reply or a provider
    error: both give the fallback report. ``model=None`` gives it directly.
    The verdict is not modified and no reply can change it."""
    evidence = tuple(evidence)
    rejections: tuple[str, ...] = ()
    attempts, cost, tokens = 0, 0.0, 0
    reason = "no model configured" if model is None else ""
    parsed: Optional[_Parsed] = None
    if model is not None:
        validator = _Validator(verdict, evidence, repo, context, config)
        try:
            q = query_json(model, build_messages(verdict, evidence, repo, context, config), validator,
                           retries=config.retries)
            parsed, attempts, rejections, cost, tokens = q.value, q.attempts, q.errors, q.cost, q.tokens
        except LLMFailure as e:
            reason = str(e)
            rejections, attempts = e.errors, len(e.replies)
            cost, tokens = sum(r.cost for r in e.replies), sum(r.input_tokens + r.output_tokens for r in e.replies)
        if parsed is None:
            log.warning("report for %s fell back to ranked evidence: %s", context.incident_id, reason)

    if parsed is not None:
        summary, ids, localization = parsed.summary, parsed.evidence_ids, parsed.localization
        task = parsed.task or default_task(verdict, context)
        source = "model"
    else:
        summary, ids = fallback_summary(verdict, evidence), tuple(e.id for e in _ranked(evidence))
        localization, task, source = fallback_localization(evidence), default_task(verdict, context), "fallback"

    prompt = (build_hagent_prompt(verdict, context, task, localization, evidence, template)
              if verdict.destination in HAGENT_DESTINATIONS else None)
    return Report(verdict, summary, ids, localization, prompt, source, attempts, rejections,
                  "" if source == "model" else reason, cost, tokens, task)
