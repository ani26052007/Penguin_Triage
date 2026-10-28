#!/usr/bin/env bash
# Day one: decides whether the Mimic reference is usable (decisions 3 and 4).
#   V-mimic-override  2 Mimic trials with a lighting override (runbook mimic_check.override). Mimic writes no
#                     Arena episode JSONL, so "recorded variations" are looked for in the generated HDF5
#                     (any key or attribute naming a variation) and, as the decisive check, the override run's
#                     camera frames must be darker than the nominal run's (mean pixel ratio < darker_ratio).
#   V-mimic-seed      two nominal runs with different datagen seeds and one repeat of the first seed, compared
#                     by per-attempt outcomes and a content digest of the generated HDF5. The seed flag is
#                     MIMIC_SEED_FLAG, else the task config's reference.mimic.seed.flag (a PLACEHOLDER); it is
#                     looked up in generate_dataset.py --help first, and without it only the repeat runs.
#   V-mimic-out       the stdout format: the progress lines, kept in the result, parsed by the reference's parser.
#   V-mimic-guarantee attempts vs trials asked, per run.
# Needs no GR00T server. Its override HDF5 feeds finetune_dryrun.sh.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

require_isaac
require_vla
begin_measurement

task_cfg="$(rb_get task_config)"
script_rel="$("$(vla_py)" -c 'import sys; from vla_isaac.config import load_task_config; print(load_task_config(sys.argv[1]).reference.mimic.script)' "$task_cfg")"
flag="${MIMIC_SEED_FLAG:-$("$(vla_py)" -c 'import sys; from vla_isaac.config import load_task_config; print(load_task_config(sys.argv[1]).reference.mimic.seed.flag)' "$task_cfg")}"
timeout_s="$(rb_get cell_timeout_s)"

if [[ ! -f "$out/generate_dataset_help.txt" ]]; then
  (cd "$ARENA_ROOT" && timeout 900 "$(arena_py)" "$script_rel" --help) > "$out/generate_dataset_help.txt" 2>&1 || true
fi
flag_in_help=false
if grep -q -- "$flag\b" "$out/generate_dataset_help.txt"; then flag_in_help=true; fi
echo "datagen seed flag $flag in generate_dataset.py --help: $flag_in_help"
grep -n -i "seed" "$out/generate_dataset_help.txt" > "$out/seed_flags_in_help.txt" || true

# mimic_run NAME TRIALS [mimic-argv args...]
mimic_run() {
  local name="$1" trials="$2" rc=0
  shift 2
  local dir="$out/$name"
  if [[ -f "$dir/probe.json" ]]; then
    echo "run $name: already complete"
    return 0
  fi
  if [[ -d "$dir" ]]; then mv "$dir" "$dir.failed-$(date -u +%s)"; fi
  mkdir -p "$dir"
  echo "$trials" > "$dir/trials"
  tool mimic-argv --runbook "$runbook_cfg" --backend "$VLA_BACKEND_CONFIG" --trials "$trials" \
    --output-file "$dir/generated.hdf5" "$@" > "$dir/argv.txt"
  local -a argv
  mapfile -t argv < "$dir/argv.txt"
  echo "run $name: $trials trial(s); log $dir/mimic.log"
  (cd "$ARENA_ROOT" && timeout "$timeout_s" "${argv[@]}") > "$dir/mimic.log" 2>&1 || rc=$?
  echo "$rc" > "$dir/exit_code"
  if ((rc != 0)); then die "run $name: Mimic exited $rc; see $dir/mimic.log"; fi
  require_file "$dir/generated.hdf5" "Mimic wrote no HDF5; see $dir/mimic.log"
  tool h5-probe --hdf5 "$dir/generated.hdf5" --camera "$(rb_get mimic_check.camera)" --json-out "$dir/probe.json"
}

override=()
while IFS= read -r kv; do override+=(--factor "$kv"); done < <(
  "$(vla_py)" -c 'import json,sys; [print(f"{k}={v}") for k, v in json.loads(sys.argv[1]).items()]' "$(rb_get mimic_check.override)")
mapfile -t seeds < <("$(vla_py)" -c 'import json,sys; [print(s) for s in json.loads(sys.argv[1])]' "$(rb_get mimic_check.seeds)")
((${#seeds[@]} == 2)) || die "runbook mimic_check.seeds must list two different seeds"
seed_trials="$(rb_get mimic_check.seed_trials)"
seed_a=()
seed_b=()
if [[ "$flag_in_help" == true ]]; then
  seed_a=(--seed-flag "$flag" --seed "${seeds[0]}")
  seed_b=(--seed-flag "$flag" --seed "${seeds[1]}")
fi

mimic_run override "$(rb_get mimic_check.override_trials)" "${override[@]}"
mimic_run seed_a "$seed_trials" "${seed_a[@]}"
mimic_run seed_a_repeat "$seed_trials" "${seed_a[@]}"
if [[ "$flag_in_help" == true ]]; then
  mimic_run seed_b "$seed_trials" "${seed_b[@]}"
else
  echo "seed flag not found: the different-seed run is skipped; V-mimic-seed rests on the repeat (mode fixed if identical)"
fi

tool mimic-check-result --runbook "$runbook_cfg" --out "$out" --flag "$flag" --flag-in-help "$flag_in_help" \
  --json-out "$out/check.json"
"$(vla_py)" -c 'import json,sys; c=json.load(open(sys.argv[1]))
print("override took effect:", c["override"]["took_effect"], "| seed mode:", c["seed"]["mode"],
      "| stdout parse ok:", c["stdout"]["parse_ok"])' "$out/check.json"
finish_measurement --json "check=$out/check.json" --set "override_hdf5=$out/override/generated.hdf5"
