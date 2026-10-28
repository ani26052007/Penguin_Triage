#!/usr/bin/env bash
# R6. Replay agreement (ISSUES #15) with the Phase 5 tool: the nominal cell twice, same seeds, clean mode,
# through IsaacBackend. That path is policy-wrapper's client and serve.py on the experiment runner, so a
# finished run also settles V-exp-yaml and V-exp-remote-policy. IsaacBackend starts its own server on the
# backend config's port, which must differ from the stock R2 server's (checked).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

require_isaac
require_vla
tool check-ports --backend "$VLA_BACKEND_CONFIG" --port "$gr00t_port" >/dev/null
begin_measurement

tool grid-variant --runbook "$runbook_cfg" --episodes "$(rb_get nominal.episodes)" --out "$out/grid.yaml" >/dev/null
report="$out/agreement/replay_agreement.json"         # vla_isaac.experiments.replay_agreement.AGREEMENT_JSON
if [[ ! -f "$report" ]]; then
  "$(vla_py)" -m vla_isaac.experiments.replay_agreement --grid "$out/grid.yaml" --backend "$VLA_BACKEND_CONFIG" \
    --out "$out/agreement" --nominal-only 2>&1 | tee "$out/agreement.log"
fi
require_file "$report" "the agreement tool wrote no report; see $out/agreement.log"
finish_measurement --json "agreement=$report"
