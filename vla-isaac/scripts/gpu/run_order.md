# GPU runbook: run order on the rented box

Written in Phase 7 of `docs/PROMPT_sim_side.md`. **The scripts have not been run as a sequence.** Parts of the stack were run by hand in one session on 2026-10-29 (see "Box prerequisites found in the 2026-10-29 session" and the root `README.md`, "Status"). They are checked
only by GPU-free tests (`vla-isaac/tests/test_gpu_runbook.py`: strict headers, environment variables,
no secrets, no downloads outside setup/download scripts, the GPU check against a fake `nvidia-smi`, the
helpers and the collector on fixture data). Expect the first real run to find things to fix; every
unread API is a row in `docs/VERIFY_ON_GPU.md`, and `collect_measurements.py` says which rows the runs settled.

Target box: one RTX 6000-class GPU (RTX 6000 Ada / RTX PRO 6000, 48 GB or more, with RT cores), Linux x86_64, NVIDIA
driver with `nvidia-smi`, `git`, and `uv` ≥ 0.8.4. Paths below are relative to `vla-isaac/scripts/gpu/`.

## Rules every script follows

- `set -euo pipefail`; refuses to start, naming the variables, when a required environment variable is missing.
- Writes only into `measurements/<RUN_TS>/<script>/` (default `measurements/` at the repo root) and prints that
  directory first and last. Each finished directory gets a `result.json` and a `DONE` marker.
- Idempotent: a finished script is skipped (`FORCE=1` redoes it, moving the old directory aside, never deleting
  it); an unfinished one resumes, skipping each Arena cell or Mimic run that already has its summary.
- A failed script still leaves `result.json` with `"status": "failed"` and its exit code, plus the logs.
- Downloads only in `R0_setup.sh` (packages, submodules), `R1_download_checkpoint.sh` and
  `R1b_download_datasets.sh` (Hugging Face). Nothing prints or writes a token: export `HF_TOKEN` in your own
  shell if a repo is gated; the scripts only test whether it is set.
- Task-specific values (checkpoint, dataset, policy, episode counts, probe levels, gate thresholds) are in
  `vla-isaac/config/runbook_gr1_fridge.yaml`, not in the scripts. A DROID pilot is a new runbook YAML
  passed with `VLA_RUNBOOK_CONFIG`.

## Environment

Copy `env.example` outside the repo, edit it, `source` it in every shell.

| Variable | Needed by | Meaning |
|---|---|---|
| `ARENA_ROOT` | all but 00 | the IsaacLab-Arena checkout (`external/IsaacLab-Arena`) |
| `VLA_VENV` | all but 00 | the vla-isaac venv that `R0_setup.sh` creates (hybrid-agent, vla-router, vla-isaac) |
| `VLA_BACKEND_CONFIG` | all but 00 | backend YAML; `R0_setup.sh` writes it once from `config/backend.example.yaml` |
| `MODELS_DIR` | R0, R1, R2 | the checkpoint goes to `$MODELS_DIR/ckpt` |
| `DATA_DIR` | R0, R1b | datasets, `IsaacBackend`'s run outputs and its patch store |
| `OMNI_KIT_ACCEPT_EULA` | scripts that start Isaac Sim | `YES`: you accept the Omniverse EULA |
| `ACCEPT_EULA` | scripts that start Isaac Sim | `Y` |
| `VK_ICD_FILENAMES` | `00_check_gpu.sh`, `R0_setup.sh` (and Isaac Sim itself) | the NVIDIA Vulkan ICD file(s), `:`-separated; default `/usr/share/vulkan/icd.d/nvidia_icd.json`. Both scripts refuse to continue when a listed file does not exist |
| `RUN_TS` | optional, all | the measurement timestamp; set once per session so scripts share `measurements/<RUN_TS>/` (default: a new UTC timestamp per script) |
| `MEASUREMENTS_DIR` | optional, all | default `<repo>/measurements` |
| `FORCE` | optional, all | `1` redoes a finished script (old directory moved aside) |
| `GR00T_HOST` | optional | the stock R2 server's host (default `127.0.0.1`) |
| `GR00T_PORT` | optional | the stock R2 server's port (default `5555`); the backend's own `serve.py` port must differ (R0 writes `GR00T_PORT + 1`) |
| `HF_TOKEN` | optional, R1, R1b | read by the `hf` CLI if the repos are gated; the scripts only check whether it is set |
| `LEROBOT_PREFIX` | optional, R1b | the LeRobot directory inside the dataset repo, when the runbook hint does not pick exactly one |
| `MIMIC_SEED_FLAG` | optional, mimic check | the datagen seed flag of `generate_dataset.py` (default: the task config's `reference.mimic.seed.flag`, a placeholder) |
| `SHIFTED_LEROBOT_DIR` | optional, fine-tune dry run | a LeRobot directory to use as the shifted half instead of converting the Mimic check's HDF5 |
| `VLA_RUNBOOK_CONFIG` | optional, all | runbook YAML (default `vla-isaac/config/runbook_gr1_fridge.yaml`) |

## Box prerequisites found in the 2026-10-29 session

Measured once, on an RTX PRO 6000 Blackwell Server Edition (compute capability 12.0, 97 GB), driver 580.178.04, Ubuntu 22.04
on a cloud VM. Not re-checked on any other box.

- **Vulkan ICD.** Isaac Sim crashed at camera init until the NVIDIA Vulkan driver package was installed. Install
  `libnvidia-gl-<driver major>` matching the driver (`libnvidia-gl-580-server` on that box) and export
  `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json` (already in `env.example`). `00_check_gpu.sh` and `R0_setup.sh`
  now stop with this advice when the file is missing.
- **Blackwell (compute capability 12.0) torch.** The GR00T venv's locked torch 2.7.1+cu126 has no sm_120 kernels
  (`no kernel image is available`). The fix found in that session: `torch==2.7.1+cu128` and `torchvision==0.22.1+cu128` from
  `https://download.pytorch.org/whl/cu128`. The explicit `+cu128` matters: a plain `==2.7.1` is a no-op in uv. flash_attn 2.7.4
  worked afterwards. `R0_setup.sh` now runs that install in the GR00T venv when the first GPU reports 12.0; the scripted version
  is covered by GPU-free tests only and has **not been run**.
- **Pinned `RUN_TS`.** Scripts share `measurements/<RUN_TS>/` only if every shell uses the same value. The session kept it in
  `~/.vla_run_ts` (written once with `date -u +%Y%m%dT%H%M%SZ > ~/.vla_run_ts`) and set `export RUN_TS="$(cat ~/.vla_run_ts)"`
  in each shell instead of `env.example`'s per-shell `date`. Use one or the other, not both.
- **Mimic camera init.** The Isaac Lab Mimic override check failed at camera init with both `--device cpu` and
  `--device cuda:0` (`ValueError: Invalid object in Py_Graph in getWrappedGraphFromNode`, "Unable to create prim for graph at
  /Render/PostProcess/SDGPipeline"); `policy_runner.py` created cameras fine. Mimic uses `gr1_pink` and `policy_runner` uses
  `gr1_joint`: a hypothesis, not tested. Mimic is therefore unverified: V-mimic-override and the Mimic decision remain open.
- **Git over HTTPS.** Arena's nested submodules use SSH URLs. Without GitHub SSH keys run
  `git config --global url."https://github.com/".insteadOf git@github.com:` before `R0_setup.sh`.
- **`R2_server.sh` and pipes.** The server keeps the pipe open, so `R2_server.sh | tail` hangs. Redirect to a file:
  `R2_server.sh > "$MEASUREMENTS_DIR/r2.log" 2>&1`. The server answered ping after 8-14 s.
- **Parallel runs.** Start parallel cells (separate processes, own ports) 60 s apart: three simulators starting at once
  collided on an HDF5 lock ("unable to lock file"). Three parallel runs peaked at 52 GB VRAM and 98% GPU utilisation.
- **Spot instances.** The session's VM was a Spot instance and was preempted once; results were saved before it went.
  Copy `measurements/` off the box as you go.
- **`check_client_tests.sh`.** In that session it reported ok although the vla venv had a collection error (exit 2) and the
  GR00T venv had no pytest. It now writes `"status": "not_settled"` in that case (GPU-free tests only; not re-run on the box).

## The first three commands

```bash
source ~/vla-env.sh                                  # your edited copy of env.example
vla-isaac/scripts/gpu/00_check_gpu.sh                # GPU usable? (Ampere+, RT cores, VRAM)
vla-isaac/scripts/gpu/R0_setup.sh                    # Arena, GR00T, vla-isaac venvs; backend config; versions
```

## Order

Run from the repo root, in this order, in one shell with `RUN_TS` fixed. Steps 1-6 need no GR00T server.

| # | Script | Needs | Settles (VERIFY) | Feeds decision |
|---|---|---|---|---|
| 0 | `00_check_gpu.sh` | `nvidia-smi` | (box check) | |
| 1 | `R0_setup.sh` | uv, git, network | (versions recorded) | |
| 2 | `R1_download_checkpoint.sh` | R0 | V-hf-rev, V-ckpt-cfg, V-stats-format | |
| 3 | `R1b_download_datasets.sh` | R0 | V-train-instr, V-mimic-source | instruction |
| 4 | `check_client_tests.sh` | R0 | V-client-tests | |
| 5 | `R3_list_variations.sh` | R0 | V-list-var | |
| 6 | `mimic_override_check.sh` | R1b | V-mimic-override, V-mimic-seed, V-mimic-out, V-mimic-guarantee | Mimic usable |
| 7 | `R2_server.sh` | R1 | | |
| 8 | `R8_vram.sh` (runs `R4_nominal_clean.sh`) | R2 | V-viz, V-timestamped-dir, V-jsonl-success, V-progress | pilot (gate) |
| 9 | `R7_dark_cell.sh` | R2 | V-var-record | pilot |
| 10 | `R7b_hdr_and_colour_temp.sh` | R2 | | pilot |
| 11 | `instruction_check.sh` | R1b, R2 | V-instruction | instruction |
| 12 | `runner_parity.sh` | R2 (reuses R4) | V-runner-parity | |
| 12b | `wrapper_parity.sh` | R1, R2 (own `serve.py` on the backend port; reuses R4 only with the same instruction) | V-wrapper-parity | policy-wrapper reproduces the stock client |
| 13 | `R5_nominal_incident.sh` | R2 | | mode gap |
| 14 | `R6_replay_agreement.sh` | R1 (own server) | V-exp-yaml, V-exp-remote-policy | (agreement) |
| 15 | `R2_server.sh stop`, then `finetune_dryrun.sh` | R1, R1b, step 6 | V-converter-noisaac, V-finetune-log | merged dataset trains |
| 16 | `$VLA_VENV/bin/python vla-isaac/scripts/gpu/collect_measurements.py` | anything above | prints settled / open ids | all five |
| (route check) | `router_obs_check.py` on instrument directories (see below) | traced runs of clean and bugged code | (none; not a VERIFY row) | router evidence on real frames |

After step 10, look at the pilot decision before spending more (`collect_measurements.py` can run at any time).
If it says "switch to DROID", stop here: steps 11-15 measure the GR1 pilot.

## Route check on real frames

`router_obs_check.py` and `record_cell.sh` come from the 2026-10-29 session and ran only there. `record_cell.sh <checkout>
<label> <episodes> <seed> <port> [video 0|1]` starts policy-wrapper's `serve.py` and one policy-wrapper cell (its paths are
that box's; edit them). With `VLA_INSTRUMENT_DIR` set (frame every 5 steps) the policy-wrapper instrument leaves
`env*_ep*.json` files per episode. Build the nominal from a clean run (the session used 3 episodes, seed 11) and check other
directories against it, including a held-out clean run (seed 12):

```bash
$VLA_VENV/bin/python vla-isaac/scripts/gpu/router_obs_check.py <clean_seed11_dir> <clean_seed12_dir> <bugged_dir>
```

On the session's data this gave 0 evidence items for the held-out clean run and one each for BUG-02 (`s1.obs.orientation.ego_view`)
and BUG-01 (`s1.obs.channel_order.ego_view`), 3 episodes per condition. Only the observation-contract check runs; this is not the
router loop.

## What each script should print and write

Each script prints `output dir: measurements/<RUN_TS>/<script>` first and last; the files below are in that
directory, and every finished one has `result.json` and `DONE`.

- **`00_check_gpu.sh`**. One line per GPU: name, compute capability, VRAM, driver; `GPU OK`. On an A100 /
  H100 / H200-class card: `REJECTED: ... has no RT cores ...` and exit 1; before Ampere (compute capability
  < 8.0): `REJECTED: ... pre-Ampere; flash-attn ... needs 8.0 or newer` and exit 1. Below 46000 MiB a note that
  the server and Isaac Sim may not share the card. Writes `nvidia_smi.csv`, `result.json` (`gpus[]`, `ok`, `reason`).
- **`R0_setup.sh`**. The uv logs, then one line `isaacsim <v> | isaaclab <v> | arena <git describe>`. Writes
  `versions.json` (isaacsim, isaaclab, isaaclab_mimic, isaaclab_arena, torch, git describes), `versions_gr00t.json`
  (gr00t, torch, flash_attn, transformers), and `VLA_BACKEND_CONFIG` if it did not exist ("exists, not overwritten" otherwise).
  The versions are recorded as found; no Isaac Sim version is assumed.
- **`R1_download_checkpoint.sh`**. `<repo> @ <sha>`, the `grep` of `tune_top_llm_layers` / `num_inference_timesteps`
  / `action_horizon`, and a reminder to pin the sha in `eval.yaml` and `finetune.base_revision`. Writes
  `revision.txt`, `checkpoint.json` (the four files present? the three fields, the `statistics.json` shape),
  `training_spec.json` (read its `extra.missing` and `extra.disagreements`).
- **`R1b_download_datasets.sh`**. The number of repo files, `LeRobot directory: <prefix>` (or an error listing
  the candidates: then set `LEROBOT_PREFIX`), and the `tasks.jsonl` lines. Links the two paths of the backend
  config's `datasets` map to the downloads.
- **`check_client_tests.sh`** (V-client-tests). Per interpreter (vla-isaac venv, GR00T venv): exit code and
  pytest's last line. `client_tests_<venv>.json` has `imports_without_isaac`, the missing modules and which of
  them are Isaac modules. **Expect** either a pass (the copied client tests in policy-wrapper mirror Arena's) or a
  `No module named ...` naming what blocks them.
- **`R3_list_variations.sh`** (V-list-var; in the session it printed `false` because `ranch_dressing_hope_robolab.disappear` is new; the five existing factors were identical and the hand-written table was kept). `factor table unchanged: true|false` and the zero-action exit code
  (0 expected). Writes `list_variations.txt`, the regenerated `factors.yaml` and `factors.diff`. If `false`,
  review the diff and copy `factors.yaml` over `config/factors_gr1_fridge.yaml` by hand.
- **`mimic_override_check.sh`** (V-mimic-override, -seed, -out, -guarantee). First `datagen seed flag --seed
  in generate_dataset.py --help: true|false` (and `seed_flags_in_help.txt`); then the runs `override` (2 trials,
  dark light), `seed_a`, `seed_a_repeat`, and `seed_b` only when the flag exists (5 trials each); then one line
  `override check: pass|fail|inconclusive | seed mode: fixed|per_run|stop: ... | stdout parse ok: true|false`.
  `check.json` holds the brightness ratio, any variation keys found in the HDF5, the HDF5 digests, the parsed
  per-attempt outcomes, the Mimic progress lines (the stdout format, V-mimic-out) and attempts vs trials.
  **Expect** a ratio well below 0.8 for the dark override; a repeat with the same seed identical to the first.
- **`R2_server.sh`**. `GR00T server answers ping after <n>s` (or that one already answers). Writes `server.log`,
  `vram_server_only.csv`. `R2_server.sh stop` stops it.
- **`R8_vram.sh` → `R4_nominal_clean.sh`**. `cell nominal: 50 episodes, 1 env(s)`, then the R4 output, then the
  VRAM summary: per GPU peak and mean MiB. R4's `nominal/summary.json` has `success` (k, n, rate, Wilson
  lower/upper), `timestamped_subdir`, `success_types` (expect `["bool"]`), a `progress` sample of a success and a
  failure, and `unrecognized_arguments` (non-empty means a flag such as `--viz none` was refused: V-viz).
- **`R7_dark_cell.sh`**. The same for the `dark` cell (light intensity 100). `variations_sample` is the recorded
  format (V-var-record). **Expect** a clear drop from R4.
- **`R7b_hdr_and_colour_temp.sh`**. One cell per probe (`hdr_home_office`, `colour_temp_warm`, `colour_temp_cool`).
- **`instruction_check.sh`** (V-instruction). First one line: `eval_yaml='...' | converter_yaml='...' |
  tasks_jsonl='...' | differ: <pairs>`; then one 20-episode cell per distinct string (`instruction_<i>`, in that
  order). `strings.json` maps each source to its string and file.
- **`runner_parity.sh`** (V-runner-parity). The two cells (`policy_runner` reused from R4 when possible,
  `experiment_runner` with its `experiment.yaml`), then `comparison.json`: shared episodes, agreement, both /
  only-first / only-second counts, both rates with intervals, Fisher p, `intervals_overlap`. **Expect** overlapping
  intervals; different per-episode outcomes are possible if the two runners seed differently.
- **`wrapper_parity.sh`** (V-wrapper-parity; pilot step 2 "done when"). `instruction (policy-wrapper eval config): ...`,
  the `stock` cell (Arena's client against R2, with that instruction; R4 is reused only when it logged the same
  string), then `policy-wrapper serve.py on <host>:<port>`, the `wrapper` cell (`policy_runner.py` with
  `--policy_type policy_wrapper.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy`, the checkout on
  `PYTHONPATH`), the server stopped, and one line `stock k/n = r [lo, hi] | wrapper ... | per-seed agreement ... |
  intervals overlap: ... | parity: ...`. `parity.json` has both rates with Wilson intervals, the agreement with its
  interval, both / only-stock / only-wrapper counts, `per_seed` (`env_id:episode_in_env` → both outcomes), Fisher p,
  the logged instructions and `parity` (both complete and the intervals overlap). `wrapper/serve.log` is the server's
  log. **Expect** overlapping intervals and high agreement; GR00T's server seeds nothing per request while
  `serve.py` does (D16), so some per-seed discordance is possible without a client difference.
- **`R5_nominal_incident.sh`**. One cell, 10 envs, 50 episodes.
- **`R6_replay_agreement.sh`**. The agreement tool's output; `agreement/replay_agreement.{json,csv}` with the
  agreement k/n and interval, both-success / both-failure / discordant counts, `informative` and its note, and
  both runs' directories.
- **`finetune_dryrun.sh`** (V-converter-noisaac, V-finetune-log). First `tune_top_llm_layers of the base
  checkpoint: {...}` (a null value stops the launcher: V-ckpt-cfg). Then `convert.json` (`ok`, and
  `blocked_isaac_import: true` would mean the converter needs Isaac), `mixed: {...}` (the selection), the launcher
  log, and `dryrun.json`: `stats_json_after` (expect true: GR00T computed `meta/stats.json`), `stats_generated_line`,
  `parameters` (trainable / total), `final_model_config`, `trained`.
- **`collect_measurements.py`**. `settled <n>: ...`, `open <n>: ...`, one line per decision, and the path of
  `measurements/summary.json`.

## Decisions and the results that settle them

| Decision | Settled by | Rule |
|---|---|---|
| **Pilot stays or switches to DROID** | R4 (via R8), R7, R7b | Switch to DROID (`docs/PILOT_AND_ORDER.md`, Candidate C) if R4's nominal rate < 0.8, or if no dark / HDR / colour-temperature cell drops to about 40 % or below (the lighting family would not exist). Otherwise the pilot stays, and the failing cells become the grid's levels (`config/grid_gr1_fridge.yaml`, `rung_b.conditions`). Report R4's Wilson interval with the decision; a lower bound under 0.8 is noted. Thresholds: runbook `gate`. |
| **Mimic usable or not** | `mimic_override_check.sh` | Usable when the override took effect, the same seed repeats identically, and the stdout parses (V-mimic-out). Then set `reference.mimic.seed.mode` to the check's mode (`fixed` when the flag is missing or different seeds give the same runs; `per_run` with the found flag otherwise). Not usable when the override did not reach Mimic (G2/C1 and Rung B would run on the wrong scene) or the same seed differs. **Inconclusive** (not pass, not fail) when either run has no camera frames or the nominal frames are all black: the brightness rule cannot decide, so check the override by hand (V-mimic-override) before relying on Mimic. A stdout mismatch is fixable: correct `reference.mimic.output.progress`. |
| **policy-wrapper reproduces the stock client** (pilot step 2) | `wrapper_parity.sh` | Reproduces when both 50-episode runs complete and their Wilson intervals overlap (`parity: true`). If not, the copy differs from Arena's client or server in a way that matters (read the discordant seeds in `per_seed` and `wrapper/serve.log`); fix that before any policy-wrapper number (R6, the grid, the router) is trusted. |
| **Mode gap real or not** | R4, R5 | Real when the two-sided Fisher exact test on clean vs incident counts gives p < 0.05 (runbook `gate.alpha`); always unpaired (IMPL_PLAN D10). If real, incident-mode results are not compared with clean-mode ones. |
| **Instruction string** | `instruction_check.sh` | Switch `policy-wrapper/config/eval.yaml` `client.language_instruction` to the training string (`meta/tasks.jsonl`) if it scores at least as well as the eval string (IMPL_PLAN Q2); otherwise keep it. Rung B's converter must use the nominal set's string either way (the mixer refuses different `tasks.jsonl`; the dry run already does this). |
| **The 50/50 merged dataset trains** | `finetune_dryrun.sh` | Trains when the launcher exits 0, the manifest is complete (parameter counts and `final_model_config.json`), and `meta/stats.json` was computed for the merged directory. If not, the mixer's single-directory approach (Phase 6) needs revisiting before Rung B. |

## Cost: which box

Nothing here has been measured, so this is a plan, not a measurement. R8 is the first VRAM number.

- **Safe on an L4 (24 GB, Ada, has RT cores) or any small box**: `00_check_gpu.sh`, `R0_setup.sh`,
  `R1_download_checkpoint.sh`, `R1b_download_datasets.sh`, `check_client_tests.sh` (no GPU used),
  `R3_list_variations.sh` (Isaac Sim alone), `mimic_override_check.sh` (Isaac Sim with cameras, Mimic on CPU,
  no GR00T), `collect_measurements.py`.
- **Needs the big card first**: `R8_vram.sh` / `R4` (GR00T server and Isaac Sim with cameras on one GPU, VRAM
  unknown until R8 runs), `R5_nominal_incident.sh` (10 envs with cameras), `R6_replay_agreement.sh` (its own
  server next to the stock one unless R2 is stopped), `finetune_dryrun.sh` (Arena's fine-tune minimum is 24 GB,
  so an L4 has no headroom), and the policy cells of `R7`, `R7b`, `instruction_check.sh`, `runner_parity.sh`, `wrapper_parity.sh` (its own server next to the stock one).
- **After R8**: if its peak is well under 24 GB, the policy cells (R7, R7b, the grid) can move to L4s with
  `vla_isaac.experiments.grid --host`; fine-tuning stays on the big card.
- Episode counts per run (wall time unmeasured; R4's sets the scale): R4 50, R7 50, R7b 3 × 50, instruction
  check 20 per distinct string (2 or 3), runner parity 50 (+50 if R4 is not reusable), wrapper parity 50 + 50 (R4 is reusable only with the same instruction), R5 50, R6 2 × 50, Mimic
  check 2 + 3 × 5 trials.
