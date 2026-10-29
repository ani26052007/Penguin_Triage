#!/usr/bin/env bash
# From the 2026-10-29 GPU session; it ran only there (paths below are that box's). Starts policy-wrapper's serve.py,
# runs Arena's policy_runner against it for one cell, and prints the metrics line. Not covered by a GPU-free test.
# usage: record_cell.sh <checkout> <label> <episodes> <seed> <port> [video 0|1]
set -uo pipefail
PW="$1"; LABEL="$2"; EP="$3"; SEED="$4"; PORT="$5"; VID="${6:-0}"
source ~/vla-env.sh
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
R=~/Penguin_Triage_demo/$LABEL; mkdir -p "$R"
GP=$HOME/comp_and_research/Penguin_Triage/external/IsaacLab-Arena/submodules/Isaac-GR00T/.venv/bin/python
AR=$HOME/comp_and_research/Penguin_Triage/external/IsaacLab-Arena
(cd "$PW" && PYTHONPATH="$PW" exec "$GP" -m policy_wrapper.serve --config config/eval.yaml \
  --host 127.0.0.1 --port "$PORT" --device cuda:0 --checkpoint-dir "$HOME/data/models/pilot/ckpt") > "$R/serve.log" 2>&1 &
SP=$!
trap 'kill $SP 2>/dev/null' EXIT
for i in $(seq 1 120); do
  (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null && break
  kill -0 $SP 2>/dev/null || { echo "$LABEL: server died"; tail -5 "$R/serve.log"; exit 1; }
  sleep 5
done
sleep 20
VIDFLAG=""; [[ "$VID" == 1 ]] && VIDFLAG="--record_camera_video"
cd "$AR" && PYTHONPATH="$PW" timeout 3600 .venv/bin/python isaaclab_arena/evaluation/policy_runner.py \
  --viz none --seed "$SEED" \
  --policy_type policy_wrapper.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy \
  --policy_config_yaml_path "$PW/config/eval.yaml" --policy_device cuda:0 \
  --remote_host 127.0.0.1 --remote_port "$PORT" --num_episodes "$EP" --num_envs 1 \
  --output_base_dir "$R/arena" --enable_cameras $VIDFLAG \
  --language_instruction "Place the sauce bottle on the top shelf of the fridge, and close the fridge door." \
  put_item_in_fridge_and_close_door --embodiment gr1_joint --object ranch_dressing_hope_robolab \
  > "$R/arena.log" 2>&1
echo "$LABEL seed=$SEED exit=$? $(grep -E 'Metrics:' "$R/arena.log" | tail -1 | cut -c1-200)"
