#!/usr/bin/env bash
# Day one, V-instruction / V-train-instr (IMPL_PLAN Q2). Prints in one line the instruction string of
# policy-wrapper's eval.yaml, of the original dataset's meta/tasks.jsonl and of the converter YAML, with the
# pairs that differ; then runs nominal episodes (runbook instruction_check.episodes, clean mode, R4's seed)
# with each distinct string, passed explicitly with --language_instruction. Needs R1b and the R2 server.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

require_isaac
require_vla
require_server
begin_measurement

tool instruction-strings --runbook "$runbook_cfg" --backend "$VLA_BACKEND_CONFIG" --json-out "$out/strings.json"
mapfile -t strings < <("$(vla_py)" -c 'import json,sys; [print(s) for s in json.load(open(sys.argv[1]))["distinct"]]' "$out/strings.json")
((${#strings[@]} > 0)) || die "no instruction strings found"
episodes="$(rb_get instruction_check.episodes)"
i=0
for s in "${strings[@]}"; do
  echo "string $i: $s"
  run_cell "instruction_$i" "$episodes" 1 --instruction "$s"
  i=$((i + 1))
done
finish_measurement --json "strings=$out/strings.json"
