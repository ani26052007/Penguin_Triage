#!/usr/bin/env bash
# R2. The stock GR00T server for the stock-client runs (R3-R5, R7, R7b, instruction_check, runner_parity),
# in the background on GR00T_HOST:GR00T_PORT (docs/PILOT_AND_ORDER.md R2).
#   R2_server.sh          start it (or keep the one already answering ping)
#   R2_server.sh stop     stop the one this script started
# R6 does not use it: IsaacBackend starts policy-wrapper's serve.py on the backend config's own port.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

require_env ARENA_ROOT MODELS_DIR
require_vla
action="${1:-start}"
pid_file="${MEASUREMENTS_DIR:-$repo_root/measurements}/gr00t_server.pid"

if [[ "$action" == stop ]]; then
  if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    kill "$(cat "$pid_file")"
    echo "stopped GR00T server pid $(cat "$pid_file")"
  else
    echo "no GR00T server started by this script is running"
  fi
  rm -f "$pid_file"
  exit 0
fi
[[ "$action" == start ]] || die "usage: R2_server.sh [start|stop]"

begin_measurement --always
if tool ping --host "$gr00t_host" --port "$gr00t_port"; then
  echo "a GR00T server already answers on $gr00t_host:$gr00t_port"
  finish_measurement --set "host=$gr00t_host" --set "port=$gr00t_port" --set "reused=true"
  exit 0
fi
ckpt="$MODELS_DIR/ckpt"
require_file "$ckpt/config.json" "run R1_download_checkpoint.sh"
embodiment="$(rb_get checkpoint.embodiment_tag)"
start=$SECONDS
(cd "$ARENA_ROOT/submodules/Isaac-GR00T" && nohup "$(gr00t_py)" gr00t/eval/run_gr00t_server.py \
  --model-path "$ckpt" --embodiment-tag "$embodiment" --device cuda --host "$gr00t_host" --port "$gr00t_port" \
  > "$out/server.log" 2>&1 & echo $! > "$pid_file")
echo "server pid $(cat "$pid_file"), log $out/server.log"
for _ in $(seq 1 120); do
  if tool ping --host "$gr00t_host" --port "$gr00t_port"; then
    up=$((SECONDS - start))
    echo "GR00T server answers ping after ${up}s"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv > "$out/vram_server_only.csv" 2>&1 || true
    finish_measurement --set "host=$gr00t_host" --set "port=$gr00t_port" --set "startup_s=$up" --set "reused=false"
    exit 0
  fi
  kill -0 "$(cat "$pid_file")" 2>/dev/null || die "the server exited; see $out/server.log"
  sleep 5
done
die "no ping after 600 s; see $out/server.log"
