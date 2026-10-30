# Remediation advisor

For verdicts that are not code bugs, the router can list what a user could do next. The list comes from
`vla-router/router/remediation.py`: `remediation_options(verdict)` is a pure function of the verdict label (and
the factor name when the verdict has one). No model is involved and no model reply can add, drop or reorder an
option (`vla-router/tests/test_remediation.py`). The report writer (`router/report.py`) appends the menu to its
report as plain text, marked "user's choice; not chosen by a model", for verdicts that have one. The router
does not run any option, and `decide.py`, the Stage 1 checks and the replays are unchanged.

**Scope.** The system detects model/data gaps, separates them from code bugs, and lists remediation paths with
pipelines; none of the data or fine-tuning paths was run, so no claim that they restore success.

## Table

| Verdict (row) | Options, in the order listed | Data options |
|---|---|---|
| `vision_factor` (6), `vision_interacting` (7) | augmentation; re-render under randomization of the named factor; Mimic regeneration; generative re-render; fine-tune | yes |
| `action_gap` (8) | scripted / oracle demos; small RL oracle; teleop demos multiplied by Mimic; the policy's own filtered rollouts | yes |
| `perception_unresolved` (9) | propose untested factors and replay them; new sensor modality | replay only; modality is report only |
| `sim_asset` (2), `infra` (3) | none: environment fixes, not model fixes | no |
| `code` (4) | none: routed to hagent | no |
| `rendering_mismatch` (5) | none: config edit | no |
| `no_significant_failure` (1), `undetermined` (10) | none | no |

## Options

Status strings come from a fixed set: "implemented, tested on fakes", "implemented, not run", "blocked on this
box: <reason>", "not implemented". "Tested on fakes" means the GPU-free tests in `vla-isaac/tests`; the code was
not run on real data. The "blocked" reason is the Mimic failure of 2026-10-29: Isaac Lab Mimic fails at camera init
(SDGPipeline / Py_Graph error) on `--device cpu` and `--device cuda:0`.

### Vision rows (6, 7)

| Id | Option | Prerequisites | Repo entry point | Status | Caveat |
|---|---|---|---|---|---|
| vision-augmentation | GR00T colour-jitter / rotation flags via the Rung A preset, no new data | A dataset and a GR00T checkout | `vla-isaac/vla_isaac/data/rung_a.py`, `vla-isaac/vla_isaac/train/finetune_launcher.py` | implemented, tested on fakes | A `None` setting passes no flag and keeps the checkpoint's own setting. Augmentation may not cover the shift. |
| vision-rerender-variation | Re-render under randomization of the named factor (Isaac variation overrides) | Source demonstrations that can be replayed or regenerated | `vla-isaac/vla_isaac/arena_ext/launch_experiment.py`, `vla-isaac/vla_isaac/scene/factors.py` | implemented, not run | Overrides alone give rollouts of the current policy, not expert data. The lighting sweep itself ran (see the worked example below); this option's re-render path did not. |
| vision-rerender-mimic | Mimic regeneration under the named factor (Rung B) | Annotated source demonstrations; a working Mimic | `vla-isaac/vla_isaac/data/rung_b_mimic.py` | blocked on this box: Mimic camera-init failure | Never ran. |
| vision-generative-rerender | Generative re-render (Cosmos) | A video-to-video model | not implemented | not implemented | No code. |
| vision-finetune | Fine-tune on the resulting data | Data from an option above; a GPU | `vla-isaac/vla_isaac/train/finetune_launcher.py`, `.../data/mixer.py`, `.../data/heldout.py` | implemented, tested on fakes | Not run: no claim about its effect. GR00T run-to-run variance is reported as 5-6 points. |

### Action gap (row 8)

| Id | Option | Prerequisites | Repo entry point | Status | Caveat |
|---|---|---|---|---|---|
| action-scripted-demos | Scripted / oracle demos on structured tasks | A task a script can solve | not implemented | not implemented | Building blocks exist in Isaac Lab Arena; not wired here. |
| action-rl-oracle | Small RL oracle with privileged simulator state | Privileged state; an RL setup | not implemented | not implemented | No code. |
| action-teleop-mimic | A few human teleop demos recorded to HDF5 with subtask annotation, multiplied by Isaac Lab Mimic | A teleop device; annotated source demos. Reference: [Isaac Lab teleop and imitation learning](https://isaac-sim.github.io/IsaacLab/main/source/features/imitation-learning/teleop_imitation.html) | `vla-isaac/vla_isaac/data/rung_b_mimic.py` (the Mimic stage only) | blocked on this box: Mimic camera-init failure | Mimic needs annotated source demos, which can come from teleop, a scripted policy or an RL policy; scripted and RL sources are untested with Arena's annotation flow. Recording and annotation are not wired here. |
| action-own-rollouts | The policy's own successful rollouts, filtered | Rollouts that include successes | not implemented | not implemented | Behaviour cloning of the policy's own behaviour, not expert data. |

### Perception unresolved (row 9)

| Id | Option | Prerequisites | Repo entry point | Status | Caveat |
|---|---|---|---|---|---|
| perception-factor-search | Propose untested visual factors and replay them | Nominal and failing frames; a vision model or a human to name candidates | `vla-router/router/factor_search.py` | implemented, not run | Interface and stub client only. |
| perception-new-modality | A new sensor modality | Outside this repo | not implemented | not implemented | Report only. |

## Tests

`vla-router/tests/test_remediation.py`: one test per row group; every path in `repo_entry_point` exists; status
strings come from the fixed set; the module imports no model or decision code; a scripted model reply (including
one that asks for a `remediation` key) cannot change the menu in the report; verdicts without a menu get none.
The existing router tests pass unchanged.

## Worked example: the dark cell

The lighting sweep (policy-wrapper, dome-light intensity 10 and 30 against the default 100, seeds 1-2) gave 0/20 at each of 10 and 30 and 17/20 at the default on the same seeds, a real vision-failure cell. If the router had reached a `vision_factor` verdict for it (it did not; `decide()` on real evidence stopped at `undetermined` because the reference replay was missing, and the observation check fires value-range and colour-balance evidence on this cell), the advisor would list the vision-row options in this order:

1. Augmentation via the Rung A preset (implemented, tested on fakes). A lighting range wide enough to cover intensity 10 is not guaranteed to be covered by the colour-jitter flags.
2. Re-render under randomization of the named factor, i.e. dome-light intensity (implemented, not run). Overrides alone give rollouts of the current policy, not expert data.
3. Mimic regeneration under the factor (blocked on this box: the Mimic camera-init failure).
4. Generative re-render (Cosmos): not implemented.
5. Fine-tune on the resulting data (implemented, tested on fakes; not run, no claim about its effect).

The advisor lists these; none was run. A short Rung A fine-tune on this cell is the natural next experiment.
