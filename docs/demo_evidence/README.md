# Demo evidence

Files from the 2026-10-29/30 GPU session. Nothing here was edited, except that `demo_bundle7/rescore07_output.txt` was written from the session's output and `rescore07.py` is a merged version of two steps that ran separately on the VM (the merged form was not run). A scan for `NEBIUS`, `api_key`, `Bearer`, `hf_` and `sk-` tokens
found none in the text files of bundles 1-9 (binary files and videos were not scanned; bundles 8 and 9 and `rescore07.py` were scanned again on 2026-10-30).

## Present

| Path | What |
|---|---|
| `demo_bundle1/evidence/` | `bug02_fix.patch`, `task_bug02.md`, `task_bug02b.md`, `hagent_bug02_traj.json`, `hagent_bug02_attempt1_traj.json` |
| `demo_bundle1/{clean,bugged,fixed}/arena/`, `nominal_stock/` | Arena result files, reports and simulator-POV videos |
| `demo_bundle2/clips/`, `demo_bundle2/logs/par_*` | video clips; logs of the clean, bugged and fixed runs |
| `demo_bundle3/`, `demo_bundle5/` | `router_check.py`, `router_check2.py`, `router_check3.py`, `router_check_output.txt`, `router_check3_output.txt`, `record_cell2.sh`, `router_ev/` (instrument directories), `meta/` (runbook results), `mimic_gpu/mimic_tail.txt` |
| `demo_bundle6/` | `auto_repair.py`, `b_suite.sh`, `b07.sh`, `auto5.log`, `auto_repair/` (attempt patches, prompts, `results.json`), `logs/par_bug*` |
| `demo_bundle7/` | `auto_repair.py`, `auto_bug03.py`, `auto_bug07.py`, `cfg_check.py`, `rescore07.py`, `rescore07_output.txt`, `light_sweep.sh`, `light_sweep.log`, `auto_repair/`, `auto_bug03/`, `auto_bug07/` (patches, prompts where saved, `results.json`), `logs/par_*` |
| `demo_bundle8/` | `router_more_checks.py` and `router_more_checks_output.txt`, `decide_real.py` and `decide_real_output.txt`, `dark_check.py` and `dark_check_output.txt`, `chain2.sh`, `chain2.log`, `light_sweep.sh`, `light_sweep.log`, `logs/par_light{10,30,1000}_s{1,2}/arena.log`, `logs/ev_*/arena.log`, `router_ev/` (instrument files: clean, clean2, clean3, clean4, bugged, bug01, bug09, dark10) |
| `demo_bundle9/` | A later, partial copy of `demo_bundle8` (no `.sh` scripts, no `logs/ev_*`); `light_sweep.log` and the `par_light*_s2` logs differ from bundle 8's. Treat bundle 8 as the reference; bundle 9 can be dropped if the difference is not wanted. |

## Missing (not found locally)

- Separate `auto_bug03` / `auto_bug07` prompt files (those scripts did not save prompts).
- The hagent benchmark artifacts (`rebench/` on the author's laptop: score tables per task and mode, trajectories, `analysis/`). Expected under `docs/hagent_benchmark/`, not yet copied.

## Notes

- `*.log` is in `.gitignore`: every `par_*/arena.log`, `auto*.log` and `light_sweep.log` is ignored and has to be force-added to be published.
- `demo_bundle6/auto_repair/` and `demo_bundle7/auto_repair/` hold the same BUG-02 attempts (bundle 7 is a later copy).
- The committed, generalised versions of the session scripts are under `vla-isaac/scripts/gpu/` (`router_obs_check.py`, `router_cfg_check.py`, `auto_repair.py`, `passk.py`).
- The lighting-sweep results now exist: `demo_bundle8/logs/par_light*/arena.log` hold the final `Metrics:` lines (the earlier bundle-7 logs from the preempted attempt hold none).
- Absolute home paths with the box's username (`/home/aneesh262007/...`, 97 lines) occur in `demo_bundle5/meta/**/*.json` and
  `demo_bundle5/mimic_gpu/mimic_tail.txt`. They were left as recorded; replace them before publishing if the username should not appear.
- The username `aneesh262007` also occurs in absolute paths in 20 files of `demo_bundle8` and `demo_bundle9` (all of them `*.log` files). To redact them before publishing (review the diff first; run from the repo root):

  ```bash
  grep -rIlF /home/aneesh262007 docs/demo_evidence/demo_bundle8 docs/demo_evidence/demo_bundle9 | xargs sed -i 's#/home/aneesh262007#/home/USER#g'
  ```
