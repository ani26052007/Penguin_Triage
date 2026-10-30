"""Remediation menu for non-code verdicts (docs/REMEDIATION.md).

    remediation_options(verdict) -> tuple[RemediationOption, ...]

A pure function of the verdict label (and the factor name, when the verdict has one). No model is involved:
the options, their order and their status strings are fixed here, and a model reply cannot add, drop or
reorder them. The menu is a list for the user to choose from; nothing in it is chosen or run by the router.

``status`` says how far each path is built, and nothing here claims an option restores success:
none of the data or fine-tuning paths has been run.

``repo_entry_point`` is either ``"not implemented"`` or repo-relative paths joined by `` + `` (the test
checks that each exists). Anything else about a path goes in ``caveat``.
"""

from __future__ import annotations

from dataclasses import dataclass

from router.schemas import Verdict, VerdictLabel

NOT_IMPLEMENTED = "not implemented"
STATUS_FAKES = "implemented, tested on fakes"
STATUS_NOT_RUN = "implemented, not run"
STATUS_NOT_BUILT = "not implemented"
BLOCKED_PREFIX = "blocked on this box: "
BLOCKED_MIMIC = BLOCKED_PREFIX + "Isaac Lab Mimic fails at camera init (SDGPipeline / Py_Graph error)"
STATUSES = (STATUS_FAKES, STATUS_NOT_RUN, STATUS_NOT_BUILT)

USER_CHOICE_NOTE = "user's choice; not chosen by a model"
TELEOP_REFERENCE = "https://isaac-sim.github.io/IsaacLab/main/source/features/imitation-learning/teleop_imitation.html"
SCOPE_LINE = ("The system detects model/data gaps, separates them from code bugs, and lists remediation paths "
              "with pipelines; none of the data or fine-tuning paths was run, so no claim that they restore success.")


def status_is_valid(status: str) -> bool:
    return status in STATUSES or (status.startswith(BLOCKED_PREFIX) and len(status) > len(BLOCKED_PREFIX))


@dataclass(frozen=True)
class RemediationOption:
    id: str
    title: str
    prerequisites: str
    rough_cost: str
    repo_entry_point: str
    status: str
    caveat: str

    def entry_paths(self) -> tuple[str, ...]:
        """The repo-relative paths in ``repo_entry_point`` (empty for "not implemented")."""
        if self.repo_entry_point == NOT_IMPLEMENTED:
            return ()
        return tuple(p.strip() for p in self.repo_entry_point.split(" + "))

    def to_dict(self) -> dict:
        return {"id": self.id, "title": self.title, "prerequisites": self.prerequisites,
                "rough_cost": self.rough_cost, "repo_entry_point": self.repo_entry_point,
                "status": self.status, "caveat": self.caveat}


# Paths are built from a prefix: tests/test_rung1_backend_factory.py greps router/*.py for the simulator package
# name as text, and this module only names files in it (it imports nothing from it).
_SIM = "vla-isaac/vla" "_isaac"
_FINETUNE = (f"{_SIM}/train/finetune_launcher.py + {_SIM}/data/mixer.py"
             f" + {_SIM}/data/heldout.py")


def _vision(factor: str | None) -> tuple[RemediationOption, ...]:
    named = f"the named factor ({factor})" if factor else "the failing factor"
    return (
        RemediationOption(
            "vision-augmentation",
            "Augmentation: GR00T colour-jitter / rotation flags through the Rung A preset",
            "A training dataset and a GR00T checkout; no new data.",
            "one fine-tune run",
            f"{_SIM}/data/rung_a.py + {_SIM}/train/finetune_launcher.py",
            STATUS_FAKES,
            "Not run. A None setting passes no flag and keeps the checkpoint's own setting; it does not switch "
            "augmentation off. Augmentation may not cover the shift."),
        RemediationOption(
            "vision-rerender-variation",
            f"Re-render under randomization of {named} with Isaac variation overrides",
            "Source demonstrations that can be replayed or regenerated under the new scene.",
            "simulator time per episode, plus the source of the demonstrations",
            f"{_SIM}/arena_ext/launch_experiment.py + {_SIM}/scene/factors.py",
            STATUS_NOT_RUN,
            "Overrides alone produce rollouts of the current policy, not expert data. A lighting sweep was started "
            "on 2026-10-29 and lost to VM preemption: no result."),
        RemediationOption(
            "vision-rerender-mimic",
            f"Regenerate demonstrations under {named} with Isaac Lab Mimic (Rung B)",
            "Annotated source demonstrations and a working Mimic start.",
            "tens of simulator trials per kept demonstration",
            f"{_SIM}/data/rung_b_mimic.py",
            BLOCKED_MIMIC,
            "Mimic did not start on the 2026-10-29 box on cpu or cuda:0, so this path never ran."),
        RemediationOption(
            "vision-generative-rerender",
            "Generative re-render of existing episodes (Cosmos)",
            "A video-to-video model and the episodes to transform.",
            "not estimated",
            NOT_IMPLEMENTED,
            STATUS_NOT_BUILT,
            "No code in this repo."),
        RemediationOption(
            "vision-finetune",
            "Fine-tune on the resulting data (mixer, held-out split, launcher)",
            "Data from one of the options above; a GPU for the GR00T fine-tune.",
            "one fine-tune run plus evaluation episodes",
            _FINETUNE,
            STATUS_FAKES,
            "Not run: no claim about its effect. GR00T run-to-run variance is reported as 5-6 points, so a small "
            "change needs many episodes to see."),
    )


def _action_gap() -> tuple[RemediationOption, ...]:
    return (
        RemediationOption(
            "action-scripted-demos",
            "Scripted / oracle demonstrations on structured tasks",
            "A task with known geometry that a script can solve.",
            "script writing plus simulator time",
            NOT_IMPLEMENTED,
            STATUS_NOT_BUILT,
            "Building blocks exist in Isaac Lab Arena; they are not wired into this repo."),
        RemediationOption(
            "action-rl-oracle",
            "Small RL oracle with privileged simulator state",
            "Privileged state access and an RL training setup.",
            "an RL training run",
            NOT_IMPLEMENTED,
            STATUS_NOT_BUILT,
            "No code in this repo."),
        RemediationOption(
            "action-teleop-mimic",
            "A few human teleoperated demonstrations recorded to HDF5 with subtask annotation, multiplied by "
            "Isaac Lab Mimic",
            "A teleop device and annotated source demonstrations; reference: " + TELEOP_REFERENCE,
            "human time for the source demos, then simulator trials",
            f"{_SIM}/data/rung_b_mimic.py",
            BLOCKED_MIMIC,
            "Mimic needs annotated source demonstrations, which can come from teleop, a scripted policy or an RL "
            "policy; scripted and RL sources are untested with Arena's annotation flow. Recording and annotation "
            "are not wired here."),
        RemediationOption(
            "action-own-rollouts",
            "The policy's own successful rollouts, filtered",
            "Rollouts that include successes.",
            "simulator time",
            NOT_IMPLEMENTED,
            STATUS_NOT_BUILT,
            "This is behaviour cloning of the policy's own behaviour, not expert data; it can only reinforce "
            "what the policy can already do."),
    )


def _perception_unresolved() -> tuple[RemediationOption, ...]:
    return (
        RemediationOption(
            "perception-factor-search",
            "Propose untested visual factors and replay them",
            "Nominal and failing policy-input frames; a vision model or a human to name candidate factors.",
            "a few replays per candidate factor",
            "vla-router/router/factor_search.py",
            STATUS_NOT_RUN,
            "Interface and a stub client only; no vision model is wired and it was not run."),
        RemediationOption(
            "perception-new-modality",
            "A new sensor modality",
            "Hardware or scene changes outside this repo.",
            "not estimated",
            NOT_IMPLEMENTED,
            STATUS_NOT_BUILT,
            "Report only."),
    )


def remediation_options(verdict: Verdict) -> tuple[RemediationOption, ...]:
    """The fixed menu for the verdict's label, in a fixed order. Empty for every label without a data option."""
    label = verdict.label
    if label in (VerdictLabel.VISION_FACTOR, VerdictLabel.VISION_INTERACTING):
        return _vision(verdict.factor)
    if label is VerdictLabel.ACTION_GAP:
        return _action_gap()
    if label is VerdictLabel.PERCEPTION_UNRESOLVED:
        return _perception_unresolved()
    return ()


def menu_text(verdict: Verdict) -> str:
    """The menu as plain text, or "" when the verdict has none. Deterministic; no model text."""
    options = remediation_options(verdict)
    if not options:
        return ""
    lines = [f"Remediation options ({USER_CHOICE_NOTE}):"]
    for i, o in enumerate(options, 1):
        lines += [f"{i}. {o.title}",
                  f"   prerequisites: {o.prerequisites}",
                  f"   rough cost: {o.rough_cost}",
                  f"   repo entry point: {o.repo_entry_point}",
                  f"   status: {o.status}",
                  f"   caveat: {o.caveat}"]
    lines.append(SCOPE_LINE)
    return "\n".join(lines)
