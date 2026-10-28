#!/usr/bin/env bash
# Day one: does training accept the mixer's merged dataset? About 50 steps (runbook finetune_dryrun) of the
# default recipe on one GPU.
#   1. reads tune_top_llm_layers from the base checkpoint and prints it (the launcher refuses without it);
#   2. the shifted half: SHIFTED_LEROBOT_DIR, or else mimic_override_check.sh's override HDF5 converted with
#      Arena's converter, Isaac imports blocked (V-converter-noisaac), with the nominal set's instruction;
#   3. the mixer writes one merged LeRobot directory, without meta/stats.json;
#   4. the launcher runs on it; afterwards stats.json must exist (GR00T computed it) and the manifest holds the
#      parameter counts and final config (V-finetune-log).
# Needs R1, R1b and (without SHIFTED_LEROBOT_DIR) mimic_override_check.sh in this RUN_TS. No server, no Isaac Sim.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

require_env ARENA_ROOT
require_vla
require_file "$(gr00t_py)" "run R0_setup.sh"
begin_measurement

pipeline="$(rb_get pipeline)"
top="$(tool tune-top --backend "$VLA_BACKEND_CONFIG" --pipeline "$pipeline")"
echo "$top" > "$out/tune_top_llm_layers.json"
echo "tune_top_llm_layers of the base checkpoint: $top"

max_steps="$(rb_get finetune_dryrun.max_steps)"
tool pipeline-variant --pipeline "$pipeline" --out "$out/pipeline_dryrun.yaml" --max-steps "$max_steps" \
  --timeout-s "$(rb_get finetune_dryrun.timeout_s)" >/dev/null

shifted="${SHIFTED_LEROBOT_DIR:-}"
convert_args=()
if [[ -z "$shifted" ]]; then
  hdf5="$meas_root/$run_ts/mimic_override_check/override/generated.hdf5"
  require_file "$hdf5" "run mimic_override_check.sh in this RUN_TS first, or set SHIFTED_LEROBOT_DIR"
  mkdir -p "$out/shifted"
  [[ -f "$out/shifted/generated.hdf5" ]] || cp "$hdf5" "$out/shifted/generated.hdf5"
  shifted="$out/shifted/generated_lerobot"
  if [[ ! -f "$out/convert.json" ]]; then
    # the mixer refuses different tasks.jsonl, so the shifted set takes the nominal set's instruction string
    convert_args=(--instruction "$(tool nominal-task --backend "$VLA_BACKEND_CONFIG" --pipeline "$pipeline")")
    rm -rf "$shifted"
    tool convert --runbook "$runbook_cfg" --backend "$VLA_BACKEND_CONFIG" --pipeline "$out/pipeline_dryrun.yaml" \
      --hdf5 "$out/shifted/generated.hdf5" --work "$out/convert" "${convert_args[@]}" --json-out "$out/convert.json"
  fi
  cat "$out/convert.json"
  "$(vla_py)" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["ok"] else 1)' "$out/convert.json" \
    || die "the converter failed; see $out/convert.json"
fi
require_file "$shifted/meta/info.json" "not a LeRobot directory: $shifted"

if [[ ! -f "$out/mix.json" ]]; then
  rm -rf "$out/mixed"
  tool mix --runbook "$runbook_cfg" --backend "$VLA_BACKEND_CONFIG" --pipeline "$out/pipeline_dryrun.yaml" \
    --shifted "$shifted" --out "$out/mixed" --json-out "$out/mix.json"
fi
"$(vla_py)" -c 'import json,sys; m=json.load(open(sys.argv[1])); print("mixed:", m["selection"]); sys.exit(1 if m["stats_json_present"] else 0)' \
  "$out/mix.json" || die "the merged dataset already has meta/stats.json; the dry run must start without it"

rc=0
if [[ ! -f "$out/finetune/vla_finetune_manifest.json" ]] || ! grep -q '"result"' "$out/finetune/vla_finetune_manifest.json"; then
  rm -rf "$out/finetune"
  "$(vla_py)" -m vla_isaac.train.finetune_launcher run --backend "$VLA_BACKEND_CONFIG" \
    --pipeline "$out/pipeline_dryrun.yaml" --dataset "$out/mixed" --out "$out/finetune" 2>&1 | tee "$out/launcher.log" \
    || rc=$?
fi
tool dryrun-report --dataset "$out/mixed" --finetune-out "$out/finetune" --exit-code "$rc" --max-steps "$max_steps" \
  --json-out "$out/dryrun.json"
cat "$out/dryrun.json"
((rc == 0)) || die "the fine-tune dry run failed (exit $rc); see $out/finetune/finetune.log"
args=(--json "tune_top_llm_layers=$out/tune_top_llm_layers.json" --json "mix=$out/mix.json" --json "dryrun=$out/dryrun.json")
if [[ -f "$out/convert.json" ]]; then args+=(--json "convert=$out/convert.json"); fi
finish_measurement "${args[@]}"
