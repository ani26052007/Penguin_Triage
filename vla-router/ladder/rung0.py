"""Rung 0: cheap checks before any simulation (plan.md Section 4).

    run_rung0(env, cfg, task=..., reviewer=...) -> Rung0Result

In order, cheapest first, stopping at the first failure so the simulator and the big
model are never spent on a patch that is obviously broken:

  1. diff sanity   non-empty, not too large, touches no forbidden path, touches at least
                   one expected file when the router located one ("did the patch touch the
                   right file")
  2. lint          ``cfg.lint_cmd`` (skipped if empty)
  3. tests         ``cfg.test_cmd``
  4. smoke run     ``cfg.smoke_cmd``: a few steps that catch hard crashes such as tensor
                   shape mismatches. TODO(verify): the real short smoke command for the
                   controller repo / simulator (a few headless steps).
  5. reviewer      hagent's ``helpers.review`` (one big-model call). Fails open: an
                   unparseable reply counts as APPROVE, so this rung cannot loop. Its
                   ``pass_probability`` is logged in the result and never gates anything.

``env`` is anything with hagent's ``Env`` surface used here: ``execute(cmd, timeout_s)``
and ``git_diff()``. hagent's own ``final_review`` already runs the same reviewer inside
``solve()``; pass ``reviewer=None`` to this rung in that flow, or the big model is called
twice for one patch. A reviewer ``ModelError`` fails open (the check passes, the error is
recorded), as hagent's own review does.

``rung0_hook`` adapts this to the loop's ``Rung0Hook(request, outcome)`` (Fix F3, ISSUES #9).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, Callable, Optional

from hagent.config import Settings
from hagent.exceptions import ModelError
from hagent.helpers import review
from hagent.models import Model
from hagent.runner import make_env
from hagent.textutil import truncate_middle

if TYPE_CHECKING:
    from ladder.repair import RepairOutcome, RepairRequest
    from ladder.workspace import WorkspaceManager

log = logging.getLogger("rung0")


@dataclass(frozen=True)
class Rung0Config:
    lint_cmd: str = ""
    test_cmd: str = ""
    smoke_cmd: str = ""
    expected_files: tuple[str, ...] = ()           # the patch must touch at least one (when non-empty)
    forbidden_prefixes: tuple[str, ...] = (".git/",)
    max_diff_lines: int = 400
    timeout_s: int = 120
    max_output_chars: int = 3000


@dataclass(frozen=True)
class CheckResult:
    name: str                                       # diff | lint | tests | smoke | reviewer
    status: str                                     # pass | fail | skipped | error (reviewer failed; counts as pass)
    detail: str = ""


@dataclass(frozen=True)
class Rung0Result:
    passed: bool
    checks: tuple[CheckResult, ...]
    feedback: str = ""                              # what the repair agent should read when it failed
    pass_probability: Optional[float] = None        # reviewer's estimate; logged only
    reviewer_cost: float = 0.0
    reviewer_calls: int = 0
    reviewer_error: str = ""                        # set when the reviewer raised ModelError (failed open)

    def to_dict(self) -> dict:
        return {"passed": self.passed, "checks": [c.__dict__ for c in self.checks], "feedback": self.feedback,
                "pass_probability": self.pass_probability, "reviewer_cost": self.reviewer_cost,
                "reviewer_calls": self.reviewer_calls, "reviewer_error": self.reviewer_error}


_FILE = re.compile(r"^\+\+\+ (?:b/)?(\S+)", re.MULTILINE)


def files_in_diff(diff: str) -> tuple[str, ...]:
    """Paths changed by a unified diff (``+++`` headers; deletions show ``/dev/null`` and are read from ``---``)."""
    out = [m.group(1) for m in _FILE.finditer(diff) if m.group(1) != "/dev/null"]
    if not out:
        out = [m.group(1) for m in re.finditer(r"^--- (?:a/)?(\S+)", diff, re.MULTILINE) if m.group(1) != "/dev/null"]
    return tuple(dict.fromkeys(out))


def diff_sanity(diff: str, cfg: Rung0Config) -> CheckResult:
    if not diff.strip():
        return CheckResult("diff", "fail", "the patch is empty")
    lines = diff.count("\n")
    if lines > cfg.max_diff_lines:
        return CheckResult("diff", "fail", f"the patch is {lines} lines; the limit is {cfg.max_diff_lines}. "
                                           f"Make a smaller, more targeted change.")
    files = files_in_diff(diff)
    bad = [f for f in files if any(f.startswith(p) for p in cfg.forbidden_prefixes)]
    if bad:
        return CheckResult("diff", "fail", f"the patch touches forbidden paths: {bad}")
    if cfg.expected_files and not any(f == e or f.endswith("/" + e) or e.endswith("/" + f)
                                      for f in files for e in cfg.expected_files):
        return CheckResult("diff", "fail", f"the patch touches {list(files)} but none of the located files "
                                           f"{list(cfg.expected_files)}")
    return CheckResult("diff", "pass", f"{len(files)} file(s), {lines} lines")


def run_rung0(env, cfg: Rung0Config, *, task: str = "", reviewer: Optional[Model] = None) -> Rung0Result:
    checks: list[CheckResult] = []
    outputs: list[str] = []

    def done(passed: bool, feedback: str = "", **kw) -> Rung0Result:
        res = Rung0Result(passed, tuple(checks), feedback, **kw)
        log.info("rung0 passed=%s checks=%s", passed, [(c.name, c.status) for c in checks])
        return res

    diff = env.git_diff()
    d = diff_sanity(diff, cfg)
    checks.append(d)
    if d.status == "fail":
        return done(False, f"Rung 0 (diff sanity) failed: {d.detail}")

    for name, cmd in (("lint", cfg.lint_cmd), ("tests", cfg.test_cmd), ("smoke", cfg.smoke_cmd)):
        if not cmd:
            checks.append(CheckResult(name, "skipped", "no command configured"))
            continue
        obs = env.execute(cmd, cfg.timeout_s)
        tail = truncate_middle(obs.output, cfg.max_output_chars)
        outputs.append(f"$ {cmd}\n{tail}")
        if obs.timed_out or obs.returncode != 0:
            why = "timed out" if obs.timed_out else f"exit code {obs.returncode}"
            checks.append(CheckResult(name, "fail", f"{why}\n{tail}"))
            return done(False, f"Rung 0 ({name}) failed: {why}\n{tail}")
        checks.append(CheckResult(name, "pass"))

    if reviewer is None:
        checks.append(CheckResult("reviewer", "skipped", "no reviewer configured"))
        return done(True)
    try:
        verdict, reply = review(reviewer, task, diff, "\n\n".join(outputs))
    except ModelError as e:
        err = f"{type(e).__name__}: {e}"
        log.warning("rung0 reviewer failed, accepting the patch: %s", err)
        checks.append(CheckResult("reviewer", "error", f"reviewer failed, patch accepted (fail open): {err}"))
        return done(True, reviewer_error=err)
    checks.append(CheckResult("reviewer", "pass" if verdict.ok else "fail", verdict.feedback))
    extra = dict(pass_probability=verdict.pass_probability, reviewer_cost=reply.cost, reviewer_calls=1)
    if not verdict.ok:
        return done(False, f"Rung 0 (reviewer) rejected the patch: {verdict.feedback}", **extra)
    return done(True, **extra)


class _PatchEnv:
    """``env`` whose ``git_diff()`` is the attempt's patch. The rebuilt copy's HEAD already is the
    patched code, so its own diff against HEAD would be empty."""

    def __init__(self, env, diff: str):
        self._env, self._diff = env, diff

    def execute(self, command: str, timeout_s: Optional[int] = None):
        return self._env.execute(command, timeout_s)

    def git_diff(self) -> str:
        return self._diff


def rung0_hook(workspaces: "WorkspaceManager", settings: Settings, cfg: Rung0Config, *,
               reviewer: Optional[Model] = None,
               task: Callable[["RepairRequest"], str] = lambda req: (
                   f"Fix the {req.kind.value} problem the router found: {req.verdict.label.value} "
                   f"(row {req.verdict.row}).")) -> Callable[["RepairRequest", "RepairOutcome"], Rung0Result]:
    """The loop's ``Rung0Hook`` (Fix F3, ISSUES #9). Builds the patched code from its name
    (``WorkspaceManager.materialize(outcome.code_ref)``) in a temp copy, runs ``run_rung0`` there
    (``make_env`` honours ``settings.env``) and deletes the copy: no second copy of the patched files
    is kept. The reviewer runs only if hagent's own final review did not already approve this patch
    (``RepairOutcome.final_review_approved``)."""

    def hook(req: "RepairRequest", out: "RepairOutcome") -> Rung0Result:
        with tempfile.TemporaryDirectory(prefix="rung0-") as td:
            work = workspaces.materialize(out.code_ref, Path(td) / "repo")
            env = make_env(work, settings)
            try:
                return run_rung0(_PatchEnv(env, out.diff), cfg, task=task(req),
                                 reviewer=None if out.final_review_approved else reviewer)
            finally:
                env.close()
    return hook


__all__ = ["CheckResult", "Rung0Config", "Rung0Result", "diff_sanity", "files_in_diff", "run_rung0", "rung0_hook"]
