#!/usr/bin/env bash
# R3. The pilot env's variation catalogue (V-list-var) and a headless zero-action smoke run (step 0's
# "done when"). No GR00T server needed. The factor table is regenerated from the catalogue and compared
# with config/factors_gr1_fridge.yaml; it is written into the measurement, never over the config.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

require_isaac
require_vla
begin_measurement

mapfile -t env_cmd < <(tool env-args --runbook "$runbook_cfg")
((${#env_cmd[@]} > 0)) || die "could not read the env sub-command from the task config"
timeout_s="$(rb_get cell_timeout_s)"
factor_table="$("$(vla_py)" -c 'import sys; from vla_isaac.config import load_task_config; print(load_task_config(sys.argv[1]).factor_table)' "$(rb_get task_config)")"
names="$(rb_get factor_names)"

if [[ ! -f "$out/list_variations.txt" ]]; then
  (cd "$ARENA_ROOT" && timeout "$timeout_s" "$(arena_py)" isaaclab_arena/evaluation/policy_runner.py \
    --policy_type zero_action --list_variations "${env_cmd[@]}") > "$out/list_variations.tmp" 2>&1 \
    || die "--list_variations failed; see $out/list_variations.tmp"
  mv "$out/list_variations.tmp" "$out/list_variations.txt"
fi
"$(vla_py)" -m vla_isaac.scene.list_variations --input "$out/list_variations.txt" --output "$out/factors.yaml" \
  --names "$names" 2>&1 | tee "$out/list_variations_parse.log"
equal="$("$(vla_py)" -c 'import sys, yaml
a, b = (yaml.safe_load(open(p)) for p in sys.argv[1:3])
print("true" if a == b else "false")' "$out/factors.yaml" "$factor_table")"
diff <(sed '/^ *#/d' "$factor_table") <(sed '/^ *#/d' "$out/factors.yaml") > "$out/factors.diff" || true
echo "factor table unchanged: $equal (diff: $out/factors.diff)"

zero_rc=0
if [[ ! -f "$out/zero_action.done" ]]; then
  read -r -a app_flags <<< "$("$(vla_py)" -c 'import sys; from vla_isaac.config import load_backend_config; print(" ".join(load_backend_config(sys.argv[1]).app_flags))' "$VLA_BACKEND_CONFIG")"
  (cd "$ARENA_ROOT" && timeout "$timeout_s" "$(arena_py)" isaaclab_arena/evaluation/policy_runner.py "${app_flags[@]}" \
    --policy_type zero_action --num_steps 200 --enable_cameras --output_base_dir "$out/zero_action" "${env_cmd[@]}") \
    > "$out/zero_action.log" 2>&1 || zero_rc=$?
  ((zero_rc == 0)) && touch "$out/zero_action.done"
fi
echo "zero-action smoke run exit code: $zero_rc"
((zero_rc == 0)) || die "zero-action run failed (exit $zero_rc); see $out/zero_action.log"

finish_measurement --set "factor_table_equal=$equal" --set "factor_table=$factor_table" \
  --set "regenerated=$out/factors.yaml" --set "zero_action_exit=$zero_rc"
