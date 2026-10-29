#!/usr/bin/env bash
# Shared helpers for the GPU runbook (run_order.md). Every script sources this file; it does nothing on its own.
# Conventions: UPPER_CASE names are environment variables the user exports (run_order.md, "Environment");
# lower_case names are this library's own. Nothing here downloads, installs, or prints a secret.
set -euo pipefail

gpu_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
vla_isaac_dir="$(cd "$gpu_dir/../.." && pwd)"
repo_root="$(cd "$vla_isaac_dir/.." && pwd)"
tools="$gpu_dir/runbook_tools.py"
runbook_cfg="${VLA_RUNBOOK_CONFIG:-$vla_isaac_dir/config/runbook_gr1_fridge.yaml}"
script_name="$(basename "$0" .sh)"
gr00t_host="${GR00T_HOST:-127.0.0.1}"
gr00t_port="${GR00T_PORT:-5555}"

die() {
  echo "ERROR [$script_name]: $*" >&2
  exit 1
}

# require_env VAR...: refuse to run, naming every missing variable.
require_env() {
  local missing=() v
  for v in "$@"; do
    [[ -n "${!v:-}" ]] || missing+=("$v")
  done
  ((${#missing[@]} == 0)) || die "required environment variable(s) not set: ${missing[*]} (see vla-isaac/scripts/gpu/run_order.md, 'Environment'; source your env file)"
}

require_file() {
  [[ -e "$1" ]] || die "$1 not found${2:+: $2}"
}

# Interpreters. Arena's uv venv runs Isaac Sim; GR00T's runs the server and fine-tuning; VLA_VENV runs vla-isaac.
arena_py() { echo "$ARENA_ROOT/.venv/bin/python"; }
gr00t_py() { echo "$ARENA_ROOT/submodules/Isaac-GR00T/.venv/bin/python"; }
vla_py() { echo "$VLA_VENV/bin/python"; }

# Isaac Sim's renderer needs the NVIDIA Vulkan ICD (package libnvidia-gl-<driver major>[-server]); without it the
# process crashed at camera init on the 2026-10-29 session. VK_ICD_FILENAMES may list several files with ':'.
vulkan_icd_files() { echo "${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}"; }

# vulkan_icd_missing: print each listed ICD file that does not exist, one per line; nothing when all exist.
vulkan_icd_missing() {
  local f
  local -a files
  IFS=: read -r -a files <<< "$(vulkan_icd_files)"
  for f in "${files[@]}"; do
    [[ -e "$f" ]] || echo "$f"
  done
}

# vulkan_icd_advice DRIVER_VERSION: what to do when the ICD is missing.
vulkan_icd_advice() {
  echo "Isaac Sim needs the NVIDIA Vulkan ICD: install libnvidia-gl-${1%%.*} (libnvidia-gl-${1%%.*}-server on server driver builds) and" \
    "export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json"
}

# gpu_compute_cap: compute capability of the first GPU, e.g. 12.0 (empty when nvidia-smi cannot say).
gpu_compute_cap() { nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | xargs || true; }

# needs_cu128_torch CAP: GR00T's locked torch (2.7.1+cu126) has no sm_120 kernels ("no kernel image is available");
# measured on an RTX PRO 6000 Blackwell (compute capability 12.0). Only that capability is treated as affected.
needs_cu128_torch() { [[ "$1" == 12.0 ]]; }

# tool ARGS...: runbook_tools.py in the vla-isaac venv.
tool() { "$(vla_py)" "$tools" "$@"; }
rb_get() { tool get --runbook "$runbook_cfg" "$1"; }

# begin_measurement [--always]: create measurements/<RUN_TS>/<script>/ and print it.
# A finished directory (DONE) is skipped unless FORCE=1; an unfinished one is resumed (each step skips
# what it already finished). FORCE=1 or --always moves the old directory aside, never deletes it.
# shellcheck disable=SC2120  # optional --always
begin_measurement() {
  local always="${1:-}"
  run_ts="${RUN_TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
  export RUN_TS="$run_ts"                       # scripts called from this one write into the same timestamp
  meas_root="${MEASUREMENTS_DIR:-$repo_root/measurements}"
  out="$meas_root/$run_ts/$script_name"
  if [[ -f "$out/DONE" && "${FORCE:-0}" != 1 && "$always" != --always ]]; then
    echo "already complete (FORCE=1 to redo): $out"
    echo "output dir: $out"
    exit 0
  fi
  if [[ -d "$out" && ( -f "$out/DONE" || "${FORCE:-0}" == 1 || "$always" == --always ) ]]; then
    mv "$out" "$out.old-$(date -u +%s)"
  fi
  mkdir -p "$out"
  rm -f "$out/result.json"
  echo "output dir: $out"
  trap 'on_exit $?' EXIT
}

on_exit() {
  local rc="$1"
  if ((rc != 0)) && [[ -n "${out:-}" && -d "$out" && ! -f "$out/result.json" ]]; then
    printf '{"script": "%s", "run_ts": "%s", "status": "failed", "exit_code": %d, "out": "%s"}\n' \
      "$script_name" "$run_ts" "$rc" "$out" > "$out/result.json"
    echo "FAILED (exit $rc); partial output kept in $out (rerun to resume)" >&2
  fi
  [[ -n "${out:-}" ]] && echo "output dir: $out"
}

# finish_measurement [result args...]: result.json (cells + extras) and DONE.
# shellcheck disable=SC2120  # optional result arguments
finish_measurement() {
  tool result --out "$out" --script "$script_name" --run-ts "$run_ts" "$@" >/dev/null
  touch "$out/DONE"
  echo "done: $out/result.json"
}

require_vla() {
  require_env VLA_VENV VLA_BACKEND_CONFIG
  require_file "$(vla_py)" "run R0_setup.sh"
  require_file "$VLA_BACKEND_CONFIG" "run R0_setup.sh"
}

require_isaac() {
  require_env ARENA_ROOT OMNI_KIT_ACCEPT_EULA ACCEPT_EULA
  require_file "$(arena_py)" "run R0_setup.sh"
}

# require_server: the stock GR00T server (R2_server.sh) answers ping on GR00T_HOST:GR00T_PORT.
require_server() {
  tool ping --host "$gr00t_host" --port "$gr00t_port" \
    || die "no GR00T server answers on $gr00t_host:$gr00t_port; start it with R2_server.sh"
}

# run_cell NAME EPISODES NUM_ENVS [policy-runner-argv args...]: one policy_runner.py run with Arena's stock
# GR00T client, summarised into NAME/summary.json. A finished cell is skipped; a failed one is retried on rerun.
run_cell() {
  local cell="$1" episodes="$2" envs="$3" rc=0 seed timeout_s
  shift 3
  local dir="$out/$cell"
  if [[ -f "$dir/summary.json" ]]; then
    echo "cell $cell: already complete"
    return 0
  fi
  [[ -d "$dir" ]] && mv "$dir" "$dir.failed-$(date -u +%s)"
  mkdir -p "$dir"
  seed="$(rb_get nominal.seed)"
  timeout_s="$(rb_get cell_timeout_s)"
  tool policy-runner-argv --runbook "$runbook_cfg" --backend "$VLA_BACKEND_CONFIG" --python "$(arena_py)" \
    --out "$dir/arena" --episodes "$episodes" --num-envs "$envs" --seed "$seed" \
    --host "$gr00t_host" --port "$gr00t_port" "$@" > "$dir/argv.txt"
  local -a argv
  mapfile -t argv < "$dir/argv.txt"
  echo "cell $cell: $episodes episodes, $envs env(s); log $dir/arena.log"
  (cd "$ARENA_ROOT" && timeout "$timeout_s" "${argv[@]}") > "$dir/arena.log" 2>&1 || rc=$?
  tool summarize --runbook "$runbook_cfg" --run-dir "$dir/arena" --episodes "$episodes" --num-envs "$envs" \
    --exit-code "$rc" --log "$dir/arena.log" --json-out "$dir/summary.tmp.json"
  if ((rc != 0)); then
    mv "$dir/summary.tmp.json" "$dir/summary.failed.json"
    die "cell $cell: Arena exited $rc; see $dir/arena.log and $dir/summary.failed.json"
  fi
  mv "$dir/summary.tmp.json" "$dir/summary.json"
}
