#!/usr/bin/env bash
# 00. Is this GPU usable? Ampere or newer (flash-attn, GR00T) and with RT cores (Isaac Sim camera rendering).
# A100 / H100-class compute cards have no RT cores and are rejected; so is anything before Ampere.
# Needs only nvidia-smi: it runs before R0_setup.sh.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# Datacenter compute GPUs without RT cores (NVIDIA product specs). Matched as whole words in the name.
no_rt_cores="A100 A800 A30 H100 H800 H200 H20 B100 B200 GB200 GH200 V100 P100"
min_major=8                      # compute capability 8.x = Ampere

command -v nvidia-smi >/dev/null || die "nvidia-smi not found: no NVIDIA driver on this machine"
begin_measurement

query="$(nvidia-smi --query-gpu=index,name,compute_cap,memory.total,driver_version --format=csv,noheader,nounits 2>&1)" \
  || die "nvidia-smi failed: $query"
printf '%s\n' "$query" > "$out/nvidia_smi.csv"
nvidia-smi > "$out/nvidia_smi.txt" 2>&1 || true

fail=0
json_gpus=""
while IFS=',' read -r idx name cap mem driver; do
  idx="$(echo "$idx" | xargs)"; name="$(echo "$name" | xargs)"; cap="$(echo "$cap" | xargs)"
  mem="$(echo "$mem" | xargs)"; driver="$(echo "$driver" | xargs)"
  [[ -n "$idx" ]] || continue
  reason=""
  major="${cap%%.*}"
  if [[ ! "$major" =~ ^[0-9]+$ ]]; then
    reason="compute capability unknown ('$cap'); update the driver (nvidia-smi --query-gpu=compute_cap needs >= 510)"
  elif ((major < min_major)); then
    reason="compute capability $cap is pre-Ampere; flash-attn (GR00T) needs 8.0 or newer"
  fi
  for model in $no_rt_cores; do
    pattern="[^A-Za-z0-9]${model}[^0-9A-Za-z]"
    if [[ " $name " =~ $pattern ]]; then
      reason="$name has no RT cores; Isaac Sim's RTX camera rendering needs them (use an RTX 6000 Ada / L40S / L4-class card)"
    fi
  done
  echo "GPU $idx: $name, compute capability $cap, VRAM ${mem} MiB, driver $driver"
  if [[ -n "$reason" ]]; then
    echo "  REJECTED: $reason" >&2
    fail=1
    ok=false
  else
    ok=true
    if [[ "$mem" =~ ^[0-9]+$ ]] && ((mem < 46000)); then
      echo "  note: ${mem} MiB < 48 GB: the GR00T server and Isaac Sim may not fit together; see run_order.md 'Cost'"
    fi
  fi
  json_gpus+="${json_gpus:+, }{\"index\": $idx, \"name\": \"$name\", \"compute_cap\": \"$cap\", \"memory_total_mib\": \"$mem\", \"driver\": \"$driver\", \"ok\": $ok, \"reason\": \"$reason\"}"
done <<< "$query"

[[ -n "$json_gpus" ]] || die "nvidia-smi listed no GPU"
status=ok
((fail == 0)) || status=failed
printf '{"script": "%s", "run_ts": "%s", "status": "%s", "gpus": [%s]}\n' \
  "$script_name" "$run_ts" "$status" "$json_gpus" > "$out/result.json"
((fail == 0)) || die "this GPU cannot run the runbook (reasons above)"
touch "$out/DONE"
echo "GPU OK"
