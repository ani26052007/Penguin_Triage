"""The repair step of the incident loop (plan.md 3.2.10, 3.3).

``Repairer.repair(request) -> RepairOutcome`` is the interface the loop calls for both
destinations that edit the system: ``hagent`` (code / config bug) and ``config_fix``
(rendering mismatch: "config edit, hagent task, then Rung 1", 3.2.10). It returns a new
``CodeRef``: the patched system. The simulator backend is what makes that ref runnable.

``HagentRepairer`` is the real implementation. It renders the Step 6 prompt (router verdict
framed as a hypothesis, located evidence, repro command, rollout summary; nothing from the
bug ledger or a clean commit) and calls hagent's ``solve()`` on a throwaway COPY of the repo,
so the original checkout is never edited and every attempt starts from the code under repair.

TODO(verify): the patched ``CodeRef`` is ``<base code hash>+<sha1 of the patch>``. The real
backend must be able to build the code at that ref (apply ``RepairOutcome.diff`` to the base
checkout). The mock backend only needs the key.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Protocol

from hagent.config import Settings
from hagent.models import Model
from hagent.runner import solve
from hagent.task_template import TaskTemplate

from router.report import RepoIndex, ReportContext, build_hagent_prompt, default_task, write_report
from router.schemas import CodeRef, Evidence, Incident, Verdict
from router.stage2 import Branch


class RepairKind(str, Enum):
    CODE = "code"        # destination hagent
    CONFIG = "config"    # destination config_fix


@dataclass(frozen=True)
class RepairRequest:
    kind: RepairKind
    incident: Incident                   # its code_ref is the code under repair
    verdict: Verdict
    evidence: tuple[Evidence, ...]
    branch: Optional[Branch]
    attempt: int                         # 1-based, over the whole incident
    feedback: tuple[str, ...] = ()       # why earlier attempts failed (Rung 0 / Rung 1 / Rung 2 text)


@dataclass(frozen=True)
class RepairOutcome:
    code_ref: CodeRef                    # equal to the input code when nothing was changed
    diff: str = ""
    notes: str = ""
    cost: float = 0.0
    big_calls: int = 0


class Repairer(Protocol):
    def repair(self, request: RepairRequest) -> RepairOutcome: ...


def patched_code_ref(base: CodeRef, patch: str) -> CodeRef:
    if not patch.strip():
        return base
    return CodeRef(f"{base.code_hash}+{hashlib.sha1(patch.encode('utf-8', 'replace')).hexdigest()[:12]}",
                   base.eval_config)


class HagentRepairer:
    """``solve()`` on a copy of ``repo``. ``report_model=None`` renders the prompt from the
    fallback localization (ranked evidence, no model text)."""

    def __init__(self, repo: str | Path, settings: Settings, *, mode: str = "hybrid", small: Optional[Model] = None,
                 big: Optional[Model] = None, report_model: Optional[Model] = None,
                 repro_command: str = "run_rung1 --seeds 50", template: Optional[TaskTemplate] = None):
        self.repo, self.settings, self.mode = Path(repo), settings, mode
        self.small, self.big, self.report_model = small, big, report_model
        self.repro_command, self.template = repro_command, template
        self.calls = 0

    def build_prompt(self, req: RepairRequest) -> str:
        inc = req.incident
        summary = (f"scene {inc.scene.key()}; failing seeds {list(inc.failing_seeds)}; "
                   f"observed success rate {inc.observed_success_rate}; expected {inc.reference_rate}")
        ctx = ReportContext(inc.incident_id, inc.scene.key(), self.repro_command, summary, str(self.repo))
        report = write_report(req.verdict, req.evidence, self.report_model, repo=RepoIndex.from_dir(self.repo),
                              context=ctx, template=self.template)
        task = default_task(req.verdict, ctx)
        if req.kind is RepairKind.CONFIG:
            task += " The fix is a configuration edit."
        if req.feedback:
            task += "\n\nEarlier attempts on this incident failed validation:\n" + "\n".join(
                f"- {f}" for f in req.feedback)
        return build_hagent_prompt(req.verdict, ctx, task, report.localization, req.evidence, self.template)

    def repair(self, req: RepairRequest) -> RepairOutcome:
        self.calls += 1
        prompt = self.build_prompt(req)
        base = req.incident.code_ref
        with tempfile.TemporaryDirectory(prefix="repair-") as td:
            work: Path | str = self.repo
            if self.settings.env.backend == "local":       # docker copies the repo itself
                work = Path(td) / "repo"
                shutil.copytree(self.repo, work, symlinks=True)
            result = solve(prompt, work, self.settings, self.mode, small=self.small, big=self.big)
        big = result.usage.get("big")
        notes = f"hagent exit status: {result.exit_status}"
        if not result.submitted:
            return RepairOutcome(base, result.patch, f"{notes}; not submitted ({result.detail[:200]})",
                                 result.total_cost, big.calls if big else 0)
        return RepairOutcome(patched_code_ref(base, result.patch), result.patch, notes, result.total_cost,
                             big.calls if big else 0)


__all__ = ["HagentRepairer", "RepairKind", "RepairOutcome", "RepairRequest", "Repairer", "patched_code_ref"]
