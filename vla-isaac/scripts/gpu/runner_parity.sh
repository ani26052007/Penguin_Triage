#!/usr/bin/env bash
# Day one, V-runner-parity (IMPL_PLAN Q1). The same nominal episodes (R4's build seed, clean mode, runbook
# parity.episodes) through policy_runner.py and through the experiment runner, both with Arena's stock GR00T
# client against the R2 server; then the per-episode comparison on (env_id, episode_in_env). The
# policy_runner half reuses R4's cell when R4 finished in this RUN_TS with the same episode count.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

require_isaac
require_vla
require_server
begin_measurement

episodes="$(rb_get parity.episodes)"
seed="$(rb_get nominal.seed)"
r4_cell="$meas_root/$run_ts/R4_nominal_clean/nominal"
if [[ ! -f "$out/policy_runner/summary.json" && -f "$r4_cell/summary.json" && "$episodes" == "$(rb_get nominal.episodes)" ]]; then
  mkdir -p "$out/policy_runner"
  cp "$r4_cell/summary.json" "$out/policy_runner/summary.json"
  echo "$r4_cell" > "$out/policy_runner/reused_from.txt"
  echo "policy_runner half: reusing R4 ($r4_cell)"
fi
run_cell policy_runner "$episodes" 1

dir="$out/experiment_runner"
if [[ ! -f "$dir/summary.json" ]]; then
  [[ -d "$dir" ]] && mv "$dir" "$dir.failed-$(date -u +%s)"
  mkdir -p "$dir"
  tool experiment-argv --runbook "$runbook_cfg" --backend "$VLA_BACKEND_CONFIG" --python "$(arena_py)" --out "$dir" \
    --episodes "$episodes" --seed "$seed" --host "$gr00t_host" --port "$gr00t_port" > "$dir/argv.txt"
  mapfile -t argv < "$dir/argv.txt"
  rc=0
  (cd "$ARENA_ROOT" && timeout "$(rb_get cell_timeout_s)" "${argv[@]}") > "$dir/arena.log" 2>&1 || rc=$?
  tool summarize --runbook "$runbook_cfg" --run-dir "$dir/results" --episodes "$episodes" --num-envs 1 \
    --exit-code "$rc" --log "$dir/arena.log" --json-out "$dir/summary.tmp.json"
  ((rc == 0)) || { mv "$dir/summary.tmp.json" "$dir/summary.failed.json"; die "experiment runner exited $rc; see $dir/arena.log"; }
  mv "$dir/summary.tmp.json" "$dir/summary.json"
fi

tool compare --runbook "$runbook_cfg" --a "$out/policy_runner/summary.json" --b "$dir/summary.json" \
  --json-out "$out/comparison.json"
cat "$out/comparison.json"
finish_measurement --json "comparison=$out/comparison.json"
