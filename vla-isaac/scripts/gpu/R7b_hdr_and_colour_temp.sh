#!/usr/bin/env bash
# R7b. The other shifted-lighting probes: an HDR environment map and two colour temperatures (runbook
# `probes`), one cell each, same episodes and seed as R4. With R7 they decide whether the pilot's
# failure family exists: some cell must drop to or below the gate's shifted_max.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

require_isaac
require_vla
require_server
begin_measurement

episodes="$(rb_get probe_episodes)"
mapfile -t probes < <("$(vla_py)" -c 'import json,sys; [print(k) for k in json.loads(sys.argv[1])]' "$(rb_get probes)")
((${#probes[@]} > 0)) || die "runbook probes is empty"
for probe in "${probes[@]}"; do
  run_cell "$probe" "$episodes" 1 --probe "$probe"
done
finish_measurement --set "probes=$(rb_get probes)"
