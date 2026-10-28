#!/usr/bin/env bash
# R0. Setup (once; idempotent). docs/PILOT_AND_ORDER.md R0, plus:
#   - the converter's third-party packages in Arena's venv (V-converter-noisaac),
#   - a vla-isaac venv (VLA_VENV) holding hybrid-agent, vla-router and vla-isaac,
#   - the backend config (VLA_BACKEND_CONFIG) for this box, written once from backend.example.yaml,
#   - the installed isaacsim / isaaclab / Arena / GR00T versions, recorded into measurements (not assumed).
# This is the only script that installs; it downloads packages and submodules, never models or datasets.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

require_env ARENA_ROOT VLA_VENV VLA_BACKEND_CONFIG MODELS_DIR DATA_DIR OMNI_KIT_ACCEPT_EULA ACCEPT_EULA
command -v uv >/dev/null || die "uv not found; install uv >= 0.8.4 first (GR00T's README: needed to build flash-attn)"
command -v git >/dev/null || die "git not found"
require_file "$ARENA_ROOT/pyproject.toml" "ARENA_ROOT must be the IsaacLab-Arena checkout (external/IsaacLab-Arena)"
require_file "$repo_root/policy-wrapper/.git" "policy-wrapper is its own git repo; clone or copy it next to vla-isaac"
require_file "$repo_root/hybrid-agent/pyproject.toml" "hybrid-agent (vla-router's dependency) is missing: git submodule update --init hybrid-agent"
begin_measurement

step() { echo "== $*"; }

step "Arena submodules (Isaac Lab, GR00T)"
git -C "$ARENA_ROOT" submodule update --init submodules/IsaacLab submodules/Isaac-GR00T 2>&1 | tee "$out/submodules.log"

step "Arena venv with the GR00T client group (gr00t.rst:72-74)"
(cd "$ARENA_ROOT" && uv sync --extra dev --group gr00t-client) 2>&1 | tee "$out/arena_sync.log"

step "GR00T venv (G:README.md:107-108)"
(cd "$ARENA_ROOT/submodules/Isaac-GR00T" && uv sync --python 3.10 && uv pip install -e .) 2>&1 | tee "$out/gr00t_sync.log"

step "converter packages in Arena's venv"
uv pip install --python "$(arena_py)" tqdm imageio imageio-ffmpeg pandas pyarrow 2>&1 | tee "$out/converter_deps.log"

step "vla-isaac venv at $VLA_VENV"
[[ -x "$(vla_py)" ]] || uv venv --python 3.12 "$VLA_VENV" 2>&1 | tee "$out/vla_venv.log"
uv pip install --python "$(vla_py)" -e "$repo_root/hybrid-agent" -e "$repo_root/vla-router" -e "$vla_isaac_dir" \
  pytest numpy h5py pyzmq msgpack pyyaml 2>&1 | tee -a "$out/vla_venv.log"

step "backend config"
tool write-backend-config --runbook "$runbook_cfg" --example "$vla_isaac_dir/config/backend.example.yaml" \
  --out "$VLA_BACKEND_CONFIG" --arena-root "$ARENA_ROOT" --arena-python "$(arena_py)" --server-python "$(gr00t_py)" \
  --models-dir "$MODELS_DIR" --data-dir "$DATA_DIR" --repo-root "$repo_root" --server-port "$((gr00t_port + 1))"
tool check-ports --backend "$VLA_BACKEND_CONFIG" --port "$gr00t_port" >/dev/null

step "versions"
"$(arena_py)" - "$ARENA_ROOT" > "$out/versions.json" <<'PY'
import importlib.metadata as md, json, subprocess, sys
arena = sys.argv[1]
def dist(*names):
    for n in names:
        try:
            return md.version(n)
        except md.PackageNotFoundError:
            pass
    return None
def git(path):
    r = subprocess.run(["git", "-C", path, "describe", "--tags", "--always", "--dirty"], capture_output=True, text=True)
    return r.stdout.strip() or None
v = {"isaacsim": dist("isaacsim", "isaacsim-kernel"), "isaaclab": dist("isaaclab"),
     "isaaclab_mimic": dist("isaaclab-mimic", "isaaclab_mimic"), "isaaclab_arena": dist("isaaclab_arena", "isaaclab-arena"),
     "torch_arena": dist("torch"), "python_arena": sys.version.split()[0],
     "git": {"arena": git(arena), "isaaclab": git(arena + "/submodules/IsaacLab"),
             "gr00t": git(arena + "/submodules/Isaac-GR00T")}}
print(json.dumps(v, indent=2, sort_keys=True))
PY
"$(gr00t_py)" -c 'import importlib.metadata as m, json
def d(n):
    try: return m.version(n)
    except m.PackageNotFoundError: return None
print(json.dumps({"gr00t": d("gr00t"), "torch": d("torch"), "flash_attn": d("flash-attn") or d("flash_attn"), "transformers": d("transformers")}))' \
  > "$out/versions_gr00t.json"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader > "$out/driver.txt" 2>&1 || true
"$(vla_py)" -c 'import json,sys; v=json.load(open(sys.argv[1])); print("isaacsim", v["isaacsim"], "| isaaclab", v["isaaclab"], "| arena", v["git"]["arena"])' "$out/versions.json"

finish_measurement --json "versions=$out/versions.json" --json "versions_gr00t=$out/versions_gr00t.json" \
  --set "backend_config=$VLA_BACKEND_CONFIG"
