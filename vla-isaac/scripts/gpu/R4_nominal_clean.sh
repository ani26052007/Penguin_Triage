#!/usr/bin/env bash
# R4. Nominal success in CLEAN mode: 50 episodes (not 20: 19/20 straddles 0.8), one env, build seed from the
# runbook, Arena's stock GR00T client against the R2 server (docs/PILOT_AND_ORDER.md R4). The pilot gate.
# Also settles V-viz (the run accepts the app flags), V-timestamped-dir, V-jsonl-success and V-progress.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

require_isaac
require_vla
require_server
begin_measurement

run_cell nominal "$(rb_get nominal.episodes)" 1
finish_measurement
