#!/usr/bin/env bash
# R8. VRAM while R4 runs: samples nvidia-smi (whole GPU and per process) during R4_nominal_clean.sh, which it
# runs itself into the same RUN_TS. Run R8 instead of R4 to get both; if R4 already finished in this RUN_TS,
# start a new RUN_TS (or FORCE=1), because VRAM must be sampled while R4 runs.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

require_isaac
require_vla
require_server
begin_measurement
r4="$meas_root/$run_ts/R4_nominal_clean"
if [[ -f "$r4/DONE" && "${FORCE:-0}" != 1 ]]; then
  die "R4 already finished in RUN_TS=$run_ts ($r4); VRAM is sampled while R4 runs: use a new RUN_TS or FORCE=1"
fi
interval="$(rb_get vram.interval_s)"

nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total --format=csv,noheader,nounits -l "$interval" \
  > "$out/vram.csv" 2> "$out/vram.err" &
gpu_sampler=$!
(while true; do
   nvidia-smi --query-compute-apps=timestamp,pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true
   sleep "$interval"
 done) > "$out/vram_apps.csv" &
app_sampler=$!
stop_samplers() { kill "$gpu_sampler" "$app_sampler" 2>/dev/null || true; }
trap 'rc=$?; stop_samplers; on_exit $rc' EXIT

r4_rc=0
"$gpu_dir/R4_nominal_clean.sh" || r4_rc=$?
stop_samplers
((r4_rc == 0)) || die "R4 failed (exit $r4_rc) while sampling; samples kept in $out"
tool vram-summary --csv "$out/vram.csv" --json-out "$out/vram.json"
cat "$out/vram.json"
finish_measurement --json "vram=$out/vram.json" --set "r4_out=$r4"
