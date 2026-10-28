#!/usr/bin/env bash
# R7. One shifted cell: the dark dome light (docs/PILOT_AND_ORDER.md R7), same episodes and seed as R4.
# The override goes through the factor table (config), not a hand-written Hydra string. Also settles
# V-var-record: the value format Arena records under `variations` for a build-time draw.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

require_isaac
require_vla
require_server
begin_measurement

factors=()
while IFS= read -r kv; do factors+=(--factor "$kv"); done < <(
  "$(vla_py)" -c 'import json,sys; [print(f"{k}={v}") for k, v in json.loads(sys.argv[1]).items()]' "$(rb_get dark_cell)")
((${#factors[@]} > 0)) || die "runbook dark_cell is empty"
run_cell dark "$(rb_get nominal.episodes)" 1 "${factors[@]}"
finish_measurement --set "condition=$(rb_get dark_cell)"
