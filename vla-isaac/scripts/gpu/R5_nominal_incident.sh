#!/usr/bin/env bash
# R5. Nominal success in INCIDENT mode: the R4 command with N envs (docs/PILOT_AND_ORDER.md R5). With R4 it
# gives the mode gap: is the incident-mode (many envs) rate different from the clean one?
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

require_isaac
require_vla
require_server
begin_measurement

run_cell nominal "$(rb_get incident.episodes)" "$(rb_get incident.num_envs)"
finish_measurement
