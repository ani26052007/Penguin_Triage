#!/usr/bin/env bash
# Pilot step 2 "done when" (docs/PILOT_AND_ORDER.md, V-wrapper-parity): the nominal pilot, clean mode, the
# same build seed and episodes (runbook wrapper_parity.episodes), once with Arena's stock GR00T client against
# the stock R2 server and once with policy-wrapper (its serve.py on the backend config's port + the
# policy_wrapper client from the checkout, through policy_runner.py). Both runs get the instruction of
# policy-wrapper's eval config, which is what IsaacBackend passes, so only the client and server differ.
# Reports both rates with Wilson intervals, the per-seed agreement on (env_id, episode_in_env) with its
# interval, and parity = both complete and the two rate intervals overlap. The stock half reuses R4's cell
# when R4 finished in this RUN_TS with the same episode count and logged the same instruction.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

require_isaac
require_vla
require_server
tool check-ports --backend "$VLA_BACKEND_CONFIG" --port "$gr00t_port" >/dev/null
begin_measurement

episodes="$(rb_get wrapper_parity.episodes)"
seed="$(rb_get nominal.seed)"
timeout_s="$(rb_get cell_timeout_s)"
instruction="$(tool wrapper-instruction --runbook "$runbook_cfg" --backend "$VLA_BACKEND_CONFIG")"
pw_base="$(tool wrapper-base --runbook "$runbook_cfg" --backend "$VLA_BACKEND_CONFIG")"
require_file "$pw_base/.git" "the backend config's policy-wrapper checkout"
echo "instruction (policy-wrapper eval config): $instruction"

# Stock half.
r4_cell="$meas_root/$run_ts/R4_nominal_clean/nominal"
if [[ ! -f "$out/stock/summary.json" && -f "$r4_cell/summary.json" && "$episodes" == "$(rb_get nominal.episodes)" ]]; then
  if "$(vla_py)" -c 'import json,sys; s=json.load(open(sys.argv[1])); sys.exit(0 if s.get("language_instructions") == [sys.argv[2]] else 1)' \
      "$r4_cell/summary.json" "$instruction"; then
    mkdir -p "$out/stock"
    cp "$r4_cell/summary.json" "$out/stock/summary.json"
    echo "$r4_cell" > "$out/stock/reused_from.txt"
    echo "stock half: reusing R4 ($r4_cell)"
  else
    echo "stock half: R4 logged a different instruction; running the stock client with the eval config's"
  fi
fi
run_cell stock "$episodes" 1 --instruction "$instruction"

# policy-wrapper half: serve.py from the checkout, then policy_runner.py with the checkout on PYTHONPATH.
dir="$out/wrapper"
server_pid=""
stop_wrapper_server() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  server_pid=""
}
trap 'rc=$?; stop_wrapper_server; on_exit "$rc"' EXIT

if [[ -f "$dir/summary.json" ]]; then
  echo "cell wrapper: already complete"
else
  [[ -d "$dir" ]] && mv "$dir" "$dir.failed-$(date -u +%s)"
  mkdir -p "$dir"
  tool wrapper-serve-argv --runbook "$runbook_cfg" --backend "$VLA_BACKEND_CONFIG" > "$dir/serve_argv.txt"
  mapfile -t serve_argv < "$dir/serve_argv.txt"
  pw_host="" pw_port=""
  for ((i = 0; i < ${#serve_argv[@]} - 1; i++)); do
    [[ "${serve_argv[i]}" == --host ]] && pw_host="${serve_argv[i + 1]}"
    [[ "${serve_argv[i]}" == --port ]] && pw_port="${serve_argv[i + 1]}"
  done
  [[ -n "$pw_host" && -n "$pw_port" ]] || die "serve.py argv has no --host/--port: $dir/serve_argv.txt"
  if tool ping --host "$pw_host" --port "$pw_port"; then
    die "something already answers on $pw_host:$pw_port; stop it (R6 or IsaacBackend may still run a server)"
  fi
  echo "policy-wrapper serve.py on $pw_host:$pw_port; log $dir/serve.log"
  (cd "$pw_base" && PYTHONPATH="$pw_base" exec "${serve_argv[@]}") > "$dir/serve.log" 2>&1 &
  server_pid=$!
  deadline=$((SECONDS + 900))
  until tool ping --host "$pw_host" --port "$pw_port"; do
    kill -0 "$server_pid" 2>/dev/null || die "serve.py exited before answering ping; see $dir/serve.log"
    ((SECONDS < deadline)) || die "serve.py did not answer ping within 900 s; see $dir/serve.log"
    sleep 5
  done
  tool wrapper-runner-argv --runbook "$runbook_cfg" --backend "$VLA_BACKEND_CONFIG" --python "$(arena_py)" \
    --out "$dir/arena" --episodes "$episodes" --seed "$seed" --instruction "$instruction" > "$dir/argv.txt"
  mapfile -t argv < "$dir/argv.txt"
  echo "cell wrapper: $episodes episodes, 1 env; log $dir/arena.log"
  rc=0
  (cd "$ARENA_ROOT" && PYTHONPATH="$pw_base" timeout "$timeout_s" "${argv[@]}") \
    > "$dir/arena.log" 2>&1 || rc=$?
  stop_wrapper_server
  tool summarize --runbook "$runbook_cfg" --run-dir "$dir/arena" --episodes "$episodes" --num-envs 1 \
    --exit-code "$rc" --log "$dir/arena.log" --json-out "$dir/summary.tmp.json"
  if ((rc != 0)); then
    mv "$dir/summary.tmp.json" "$dir/summary.failed.json"
    die "cell wrapper: Arena exited $rc; see $dir/arena.log, $dir/serve.log and $dir/summary.failed.json"
  fi
  mv "$dir/summary.tmp.json" "$dir/summary.json"
fi

tool wrapper-parity --runbook "$runbook_cfg" --stock "$out/stock/summary.json" --wrapper "$dir/summary.json" \
  --json-out "$out/parity.json"
finish_measurement --json "parity=$out/parity.json"
