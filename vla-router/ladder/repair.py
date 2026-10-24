"""The repair step of the incident loop (plan.md 3.2.10, 3.3).

``Repairer.repair(request) -> RepairOutcome`` is the interface the loop calls for both
destinations that edit the system: ``hagent`` (code / config bug) and ``config_fix``
(rendering mismatch: "config edit, hagent task, then Rung 1", 3.2.10). It returns a new
``CodeRef``: the patched system. The simulator backend is what makes that ref runnable.

``HagentRepairer`` is the real implementation. Every attempt (Fix F2a, ISSUES #6-#8):

  1. builds the code under repair, ``request.incident.code_ref``, in a fresh working copy
     (``WorkspaceManager.materialize``). That is the last accepted code, never the bare
     checkout, and a failed attempt's copy is thrown away with it;
  2. renders the Step 6 prompt (router verdict framed as a hypothesis by the template, located
     evidence, repro command, rollout summary; nothing from the bug ledger or a clean commit).
     The repo path in it is the copy the agent works in (the container workdir under docker);
     the TASK is the report model's validated task, or ``default_task`` (DECISIONS.md, F2a);
  3. runs hagent's ``solve()`` there. HEAD of the copy is the code under repair, so hagent's
     patch is this attempt's change only;
  4. names the result with ``WorkspaceManager.extend``: the patch is applied to a second fresh
     copy and the new name comes from diffing those files against the base snapshot. A patch
     that does not rebuild is reported in ``rebuild_error`` with its ``rebuild_cause``
     (``UnrebuildablePatch.CAUSES``). Under the local backend the agent's own working copy is
     also named (``code_ref_of``); if the rebuilt patch gives a different name, the patch lost
     something (non-UTF-8 bytes: ``encoding``; anything else: ``mismatch``). hagent's patch is
     taken against the commit it started from, so changes the agent committed are in it and is refused,
     so a lossy patch never silently becomes different code.

The original checkout is never written: only the manager reads it.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Protocol

from hagent.config import Settings
from hagent.models import Model
from hagent.runner import solve
from hagent.task_template import TaskTemplate

from ladder.workspace import UnrebuildablePatch, WorkspaceError, WorkspaceManager, code_ref_of
from router.report import RepoIndex, ReportContext, build_hagent_prompt, write_report
from router.schemas import CodeRef, Evidence, Incident, Verdict
from router.stage2 import Branch


class RepairKind(str, Enum):
    CODE = "code"        # destination hagent
    CONFIG = "config"    # destination config_fix


@dataclass(frozen=True)
class RepairRequest:
    kind: RepairKind
    incident: Incident                   # its code_ref is the code under repair (the last accepted code)
    verdict: Verdict
    evidence: tuple[Evidence, ...]
    branch: Optional[Branch]
    attempt: int                         # 1-based, over the whole incident
    feedback: tuple[str, ...] = ()       # why earlier attempts failed (Rung 0 / Rung 1 / Rung 2 text)


@dataclass(frozen=True)
class RepairOutcome:
    """``diff``: this attempt's change, against the code under repair. ``code_diff``: the whole
    change of ``code_ref`` against its base (what ``WorkspaceManager.materialize`` applies).
    ``rebuild_error``: set when the attempt's patch could not be rebuilt on a fresh copy;
    ``code_ref`` is then the input code. ``rebuild_cause``: one of ``UnrebuildablePatch.CAUSES``.
    The diffs are the lossless ``str`` view of bytes (``ladder.workspace.as_text``)."""
    code_ref: CodeRef                    # equal to the input code when nothing was changed
    diff: str = ""
    notes: str = ""
    cost: float = 0.0
    big_calls: int = 0
    code_diff: str = ""
    rebuild_error: str = ""
    rebuild_cause: str = ""


class Repairer(Protocol):
    def repair(self, request: RepairRequest) -> RepairOutcome: ...


class HagentRepairer:
    """``solve()`` on a working copy from ``workspaces``. ``report_model=None`` renders the prompt
    from the fallback localization and ``default_task`` (ranked evidence, no model text)."""

    def __init__(self, workspaces: WorkspaceManager, settings: Settings, *, mode: str = "hybrid",
                 small: Optional[Model] = None, big: Optional[Model] = None, report_model: Optional[Model] = None,
                 repro_command: str = "run_rung1 --seeds 50", template: Optional[TaskTemplate] = None):
        self.workspaces, self.settings, self.mode = workspaces, settings, mode
        self.small, self.big, self.report_model = small, big, report_model
        self.repro_command, self.template = repro_command, template
        self.calls = 0

    def prompt_path(self, work: Path) -> str:
        """The repo path the agent's commands see: the copy itself, or the container workdir."""
        return self.settings.env.workdir if self.settings.env.backend == "docker" else str(work)

    def build_prompt(self, req: RepairRequest, work: Path) -> str:
        inc = req.incident
        summary = (f"scene {inc.scene.key()}; failing seeds {list(inc.failing_seeds)}; "
                   f"observed success rate {inc.observed_success_rate}; expected {inc.reference_rate}")
        ctx = ReportContext(inc.incident_id, inc.scene.key(), self.repro_command, summary, self.prompt_path(work))
        report = write_report(req.verdict, req.evidence, self.report_model, repo=RepoIndex.from_dir(work),
                              context=ctx, template=self.template)
        task = report.task
        if req.kind is RepairKind.CONFIG:
            task += " The fix is a configuration edit."
        if req.feedback:
            task += "\n\nEarlier attempts on this incident failed validation:\n" + "\n".join(
                f"- {f}" for f in req.feedback)
        return build_hagent_prompt(req.verdict, ctx, task, report.localization, req.evidence, self.template)

    def _local_name(self, work: Path) -> Optional[str]:
        """The name of the files the agent left in ``work`` (local backend only; under docker the
        host copy is not the agent's). None when the copy no longer names itself."""
        if self.settings.env.backend == "docker":
            return None
        try:
            return code_ref_of(work).code_hash
        except WorkspaceError:
            return None

    def repair(self, req: RepairRequest) -> RepairOutcome:
        self.calls += 1
        base = req.incident.code_ref
        with tempfile.TemporaryDirectory(prefix="repair-") as td:
            work = self.workspaces.materialize(base, Path(td) / "repo")
            prompt = self.build_prompt(req, work)
            result = solve(prompt, work, self.settings, self.mode, small=self.small, big=self.big)
            actual = self._local_name(work) if result.submitted else None
        big = result.usage.get("big")
        calls = big.calls if big else 0
        notes = f"hagent exit status: {result.exit_status}"
        if not result.submitted:
            return RepairOutcome(base, result.patch, f"{notes}; not submitted ({result.detail[:200]})",
                                 result.total_cost, calls, self.workspaces.diff_of(base))
        try:
            new = self.workspaces.extend(base, result.patch)
            if actual is not None and actual != new.code_hash:
                cause = "encoding" if "\ufffd" in result.patch else "mismatch"
                raise UnrebuildablePatch(cause, f"the patch rebuilds as {new.code_hash}, but the files the agent "
                                                f"left are {actual}")
        except UnrebuildablePatch as e:
            return RepairOutcome(base, result.patch, f"{notes}; patch could not be rebuilt ({e.cause})",
                                 result.total_cost, calls, self.workspaces.diff_of(base), rebuild_error=str(e),
                                 rebuild_cause=e.cause)
        return RepairOutcome(new, result.patch, notes, result.total_cost, calls, self.workspaces.diff_of(new))

__all__ = ["HagentRepairer", "RepairKind", "RepairOutcome", "RepairRequest", "Repairer"]
