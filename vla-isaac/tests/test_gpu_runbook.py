"""Phase 7: the GPU runbook (``scripts/gpu/``) checked without a GPU.

Static checks on every script (executable, strict header, documented environment variables only, no
secrets, no downloads outside setup/download scripts, shellcheck), the scripts' own refusals and the GPU
check against a fake ``nvidia-smi``, R4 / R7 end to end against a fake Arena interpreter and the fake
policy server, the helpers in ``runbook_tools.py``, and ``collect_measurements.py`` on fixture
measurement directories. Everything here is a FIXTURE; nothing touches Isaac, GR00T or a GPU.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from ladder.stats import fisher_exact, wilson_interval

from tests.test_data_pipeline import write_lerobot

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
GPU = REPO / "scripts" / "gpu"
VERIFY_MD = ROOT / "docs" / "VERIFY_ON_GPU.md"
RUN_ORDER = GPU / "run_order.md"
RUNBOOK = REPO / "config" / "runbook_gr1_fridge.yaml"
FAKE_SERVER = REPO / "tests" / "fixtures" / "fake_rep_server.py"
# Helpers from the 2026-10-29 session: they ran only there and are not part of the runbook sequence
# (own checks in tests/test_router_obs_check.py).
SESSION_SCRIPTS = {"record_cell.sh", "router_obs_check.py"}
SH = sorted(p for p in GPU.glob("*.sh") if p.name not in SESSION_SCRIPTS)
PY = sorted(p for p in GPU.glob("*.py") if p.name not in SESSION_SCRIPTS)
SCRIPTS = SH + PY
BASH_BUILTINS = {"BASH_SOURCE", "SECONDS", "HOME", "PATH", "PWD", "PIPESTATUS", "RANDOM", "LINENO", "FUNCNAME"}
SECRET_PATTERNS = [r"hf_[A-Za-z0-9]{20,}", r"sk-[A-Za-z0-9_-]{20,}", r"AKIA[0-9A-Z]{16}", r"gh[pousr]_[A-Za-z0-9]{20,}",
                   r"-----BEGIN [A-Z ]*PRIVATE KEY-----", r"^\s*set -[a-z]*x", r"--token\b", r"(?i)(api[_-]?key|password)\s*="]
DOWNLOADS = [r"\bhf\"? download\b", r"snapshot_download", r"hf_hub_download", r"list_repo_files", r"model_info\(",
             r"\bwget\b", r"\bcurl\b", r"\bpip install\b", r"\buv (sync|pip install|venv)\b", r"\bgit clone\b",
             r"\bgit (-C \S+ )?submodule update\b"]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, GPU / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


tools = _load("runbook_tools")
collect = _load("collect_measurements")


def documented_env() -> set[str]:
    text = RUN_ORDER.read_text(encoding="utf-8")
    section = text.split("## Environment", 1)[1].split("\n## ", 1)[0]
    return set(re.findall(r"^\| `([A-Z_][A-Z0-9_]*)` \|", section, re.M))


def env_refs(text: str) -> set[str]:
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    refs = set(re.findall(r"\$\{?!?([A-Z_][A-Z0-9_]*)", code))
    for args in re.findall(r"require_env ([A-Z_0-9 ]+)", code):
        refs |= set(args.split())
    refs |= set(re.findall(r"\bexport ([A-Z_][A-Z0-9_]*)=", code))
    return refs


# --------------------------------------------------------------------------- #
# static checks
# --------------------------------------------------------------------------- #

def test_the_runbook_has_every_script_the_prompt_names():
    names = {p.name for p in SCRIPTS}
    want = {"00_check_gpu.sh", "R0_setup.sh", "R1_download_checkpoint.sh", "R1b_download_datasets.sh", "R2_server.sh",
            "R3_list_variations.sh", "R4_nominal_clean.sh", "R5_nominal_incident.sh", "R6_replay_agreement.sh",
            "R7_dark_cell.sh", "R7b_hdr_and_colour_temp.sh", "R8_vram.sh", "instruction_check.sh", "runner_parity.sh",
            "mimic_override_check.sh", "check_client_tests.sh", "finetune_dryrun.sh", "collect_measurements.py",
            "wrapper_parity.sh"}
    assert want <= names, sorted(want - names)


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_every_script_is_executable(path):
    assert path.stat().st_mode & stat.S_IXUSR, f"{path.name} is not executable"


@pytest.mark.parametrize("path", SH, ids=lambda p: p.name)
def test_every_shell_script_has_the_strict_header(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "#!/usr/bin/env bash"
    code = [line for line in lines[1:] if line.strip() and not line.lstrip().startswith("#")]
    assert code[0] == "set -euo pipefail", f"{path.name}: the first command must be 'set -euo pipefail', got {code[0]!r}"


@pytest.mark.parametrize("path", PY, ids=lambda p: p.name)
def test_every_python_script_has_a_python3_shebang(path):
    assert path.read_text(encoding="utf-8").splitlines()[0] == "#!/usr/bin/env python3"


@pytest.mark.parametrize("path", SH, ids=lambda p: p.name)
def test_scripts_use_only_documented_environment_variables(path):
    extra = env_refs(path.read_text(encoding="utf-8")) - documented_env() - BASH_BUILTINS
    assert not extra, f"{path.name} uses environment variables not documented in run_order.md: {sorted(extra)}"


def test_every_documented_variable_is_used_and_the_python_tools_read_no_environment():
    used = set().union(*(env_refs(p.read_text(encoding="utf-8")) for p in SH))
    assert documented_env() - used == set(), "run_order.md documents variables no script reads"
    for p in PY:
        assert "os.environ" not in p.read_text(encoding="utf-8") and "getenv" not in p.read_text(encoding="utf-8")
    assert env_refs((GPU / "env.example").read_text()) <= documented_env() | BASH_BUILTINS


@pytest.mark.parametrize("path", SCRIPTS + [GPU / "env.example", RUN_ORDER], ids=lambda p: p.name)
def test_no_secrets_and_tokens_are_only_tested_for_presence(path):
    text = path.read_text(encoding="utf-8")
    for pat in SECRET_PATTERNS:
        assert not re.search(pat, text, re.M), f"{path.name} matches secret pattern {pat!r}"
    for line in text.splitlines():
        if "HF_TOKEN" in line and path.suffix == ".sh" and not line.lstrip().startswith("#"):
            assert re.search(r'\[\[ -[nz] "\$\{HF_TOKEN:-\}" \]\]', line), f"{path.name}: HF_TOKEN used beyond a presence test: {line!r}"
    assert not re.search(r"^\s*export HF_TOKEN=", text, re.M)


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_only_setup_and_download_scripts_download(path):
    if re.search(r"setup|download", path.name):
        return
    text = "\n".join(line for line in path.read_text(encoding="utf-8").splitlines() if not line.lstrip().startswith("#"))
    for pat in DOWNLOADS:
        assert not re.search(pat, text), f"{path.name} downloads or installs ({pat!r}) but its name says neither"


def test_downloads_are_where_the_names_say():
    assert re.search(r'"\$hf" download\b', (GPU / "R1_download_checkpoint.sh").read_text())
    assert re.search(r'"\$hf" download\b', (GPU / "R1b_download_datasets.sh").read_text())
    r0 = (GPU / "R0_setup.sh").read_text()
    for pkg in ("tqdm", "imageio", "pandas"):
        assert re.search(rf"uv pip install --python \"\$\(arena_py\)\" .*\b{pkg}\b", r0), pkg


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck is not installed")
def test_shellcheck_passes():
    r = subprocess.run(["shellcheck", "-x", *map(str, SH)], cwd=GPU, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_every_script_path_in_verify_on_gpu_exists():
    paths = set(re.findall(r"scripts/gpu/([A-Za-z0-9_.]+\.(?:sh|py|md))", VERIFY_MD.read_text(encoding="utf-8")))
    assert paths, "VERIFY_ON_GPU.md names no runbook script: the scan is broken"
    missing = sorted(p for p in paths if not (GPU / p).is_file())
    assert not missing, f"docs/VERIFY_ON_GPU.md names scripts that do not exist: {missing}"


def test_run_order_lists_every_script_and_only_existing_ones():
    text = RUN_ORDER.read_text(encoding="utf-8")
    named = set(re.findall(r"`([A-Za-z0-9_]+\.sh)\b", text)) | set(re.findall(r"(collect_measurements\.py)", text))
    assert {p.name for p in SCRIPTS} - {"_common.sh", "runbook_tools.py"} <= named
    assert not {n for n in named if not (GPU / n).is_file()}
    for section in ("## The first three commands", "## Decisions", "## Cost", "## Environment"):
        assert section in text
    for decision in ("DROID", "Mimic usable", "Mode gap", "Instruction string", "merged dataset trains"):
        assert decision in text


def test_every_runbook_key_and_tool_command_the_scripts_use_exists(capsys):
    rb = tools.load_runbook(RUNBOOK)
    def is_command(c: str) -> bool:
        with pytest.raises(SystemExit) as e:          # argparse: --help exits 0, an unknown command exits 2
            tools.main([c, "--help"])
        return e.value.code == 0

    for p in SH:
        text = p.read_text(encoding="utf-8")
        for key in re.findall(r"rb_get ([a-z_.]+)", text):
            tools.dotted(rb, key)
        for c in re.findall(r"(?m)(?:^\s*|\$\(|\|\|\s*|&&\s*|<\(\s*)tool ([a-z0-9-]+)", text):
            assert is_command(c), f"{p.name}: tool {c} is not a runbook_tools command"


def test_the_collector_names_only_listed_verify_ids():
    ids = set(collect.verify_ids(VERIFY_MD))
    assert set(collect.SETTLES) <= ids
    scripts = {Path(p).stem for p in map(str, SCRIPTS)}
    assert {s for s, _ in collect.SETTLES.values()} <= scripts


# --------------------------------------------------------------------------- #
# the scripts themselves: refusals, GPU check, R4 / R7 end to end
# --------------------------------------------------------------------------- #

def bare_env(tmp_path, **kw) -> dict:
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "MEASUREMENTS_DIR": str(tmp_path / "meas"), "RUN_TS": "t0"}
    env.update({k: str(v) for k, v in kw.items()})
    return env


def run(script: str, env: dict, *args: str, timeout: float = 120) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(GPU / script), *args], env=env, capture_output=True, text=True, timeout=timeout)


@pytest.mark.parametrize("path", [p for p in SH if p.name not in ("_common.sh", "00_check_gpu.sh")], ids=lambda p: p.name)
def test_scripts_refuse_to_run_without_their_variables_and_name_them(tmp_path, path):
    r = run(path.name, bare_env(tmp_path))
    assert r.returncode != 0
    m = re.search(r"required environment variable\(s\) not set: ([A-Z_0-9 ]+)", r.stderr)
    assert m, r.stderr
    named = m[1].split()
    assert named and set(named) <= documented_env(), named
    assert not (tmp_path / "meas").exists(), "a refused script must not create its measurement directory"


def fake_nvidia_smi(tmp_path, line: str) -> Path:
    b = tmp_path / "fakebin"
    b.mkdir(exist_ok=True)
    f = b / "nvidia-smi"
    f.write_text(f'#!/bin/sh\ncase "$*" in *query-gpu*) echo "{line}";; *) echo fake;; esac\n')
    f.chmod(0o755)
    return b


@pytest.mark.parametrize("line, ok, why", [
    ("0, NVIDIA A100-SXM4-80GB, 8.0, 81920, 550.54", False, "no RT cores"),
    ("0, NVIDIA H100 PCIe, 9.0, 81559, 550.54", False, "no RT cores"),
    ("0, NVIDIA H200, 9.0, 143771, 550.54", False, "no RT cores"),
    ("0, Tesla T4, 7.5, 15360, 550.54", False, "pre-Ampere"),
    ("0, Quadro RTX 8000, 7.5, 49152, 550.54", False, "pre-Ampere"),
    ("0, NVIDIA RTX 6000 Ada Generation, 8.9, 49140, 550.54", True, "GPU OK"),
    ("0, NVIDIA L4, 8.9, 23034, 550.54", True, "may not fit together"),
    ("0, NVIDIA A10, 8.6, 23028, 550.54", True, "GPU OK"),
    ("0, NVIDIA RTX PRO 6000 Blackwell Server Edition, 12.0, 97887, 580.178.04", True, "GPU OK"),
])
def test_the_gpu_check(tmp_path, line, ok, why):
    icd = tmp_path / "nvidia_icd.json"
    icd.write_text("{}")
    env = bare_env(tmp_path, VK_ICD_FILENAMES=icd)
    env["PATH"] = f"{fake_nvidia_smi(tmp_path, line)}:{env['PATH']}"
    r = run("00_check_gpu.sh", env)
    assert (r.returncode == 0) is ok, r.stdout + r.stderr
    assert why in r.stdout + r.stderr
    assert re.search(r"VRAM \d+ MiB", r.stdout)
    out = tmp_path / "meas" / "t0" / "00_check_gpu"
    assert f"output dir: {out}" in r.stdout
    res = json.loads((out / "result.json").read_text())
    assert res["status"] == ("ok" if ok else "failed") and res["gpus"][0]["ok"] is ok
    if ok:
        again = run("00_check_gpu.sh", env)
        assert again.returncode == 0 and "already complete" in again.stdout
        forced = run("00_check_gpu.sh", {**env, "FORCE": "1"})
        assert forced.returncode == 0 and list((tmp_path / "meas" / "t0").glob("00_check_gpu.old-*"))


def test_the_gpu_check_needs_the_vulkan_icd_and_says_which_package(tmp_path):
    env = bare_env(tmp_path, VK_ICD_FILENAMES=tmp_path / "missing_icd.json")
    env["PATH"] = f"{fake_nvidia_smi(tmp_path, '0, NVIDIA RTX PRO 6000 Blackwell Server Edition, 12.0, 97887, 580.178.04')}:{env['PATH']}"
    r = run("00_check_gpu.sh", env)
    assert r.returncode != 0
    assert "Vulkan ICD not found" in r.stderr and "libnvidia-gl-580" in r.stderr and "VK_ICD_FILENAMES" in r.stderr
    res = json.loads((tmp_path / "meas" / "t0" / "00_check_gpu" / "result.json").read_text())
    assert res["status"] == "failed" and res["vulkan_ok"] is False and res["gpus"][0]["ok"] is True
    assert not (tmp_path / "meas" / "t0" / "00_check_gpu" / "DONE").exists()


def bash_fn(code: str, **env) -> str:
    r = subprocess.run(["bash", "-c", f'source "{GPU}/_common.sh"; {code}'], env={"PATH": "/usr/bin:/bin", **env},
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def test_vulkan_icd_missing_lists_only_absent_files_and_defaults_to_the_standard_path(tmp_path):
    present = tmp_path / "a.json"
    present.write_text("{}")
    assert bash_fn("vulkan_icd_missing", VK_ICD_FILENAMES=str(present)) == ""
    assert bash_fn("vulkan_icd_missing", VK_ICD_FILENAMES=f"{present}:{tmp_path / 'b.json'}") == str(tmp_path / "b.json")
    assert bash_fn("vulkan_icd_files") == "/usr/share/vulkan/icd.d/nvidia_icd.json"
    assert bash_fn("vulkan_icd_advice 580.178.04").startswith("Isaac Sim needs the NVIDIA Vulkan ICD: install libnvidia-gl-580 ")


@pytest.mark.parametrize("cap, expect", [("12.0", True), ("8.9", False), ("9.0", False), ("10.0", False), ("", False)])
def test_cu128_torch_is_chosen_for_compute_capability_12_0_only(cap, expect):
    r = subprocess.run(["bash", "-c", f'source "{GPU}/_common.sh"; needs_cu128_torch "{cap}"'], env={"PATH": "/usr/bin:/bin"})
    assert (r.returncode == 0) is expect


def test_r0_installs_the_pinned_cu128_torch_into_the_gr00t_venv_only_for_that_capability():
    r0 = (GPU / "R0_setup.sh").read_text()
    code = "\n".join(line for line in r0.splitlines() if not line.lstrip().startswith("#"))
    assert re.search(r'if needs_cu128_torch "\$cap"; then', code)
    m = re.search(r'uv pip install --python "\$\(gr00t_py\)" --index-url https://download\.pytorch\.org/whl/cu128 \\\n\s+(.*?) 2>&1', code)
    assert m and m[1] == "torch==2.7.1+cu128 torchvision==0.22.1+cu128", "explicit +cu128 pins: a plain ==2.7.1 is a no-op in uv"
    assert code.index("needs_cu128_torch") > code.index("Isaac-GR00T") and code.index("vulkan_icd_missing") < code.index("uv sync")


def test_the_gpu_check_without_a_driver(tmp_path):
    env = bare_env(tmp_path)
    b = tmp_path / "nodriver"
    b.mkdir()
    for tool_name in ("dirname", "basename"):
        (b / tool_name).symlink_to(shutil.which(tool_name))
    env["PATH"] = str(b)
    r = subprocess.run(["/bin/bash", str(GPU / "00_check_gpu.sh")], env=env, capture_output=True, text=True)
    assert r.returncode != 0 and "nvidia-smi not found" in r.stderr


FAKE_ARENA = '''#!{python}
"""FIXTURE: stands in for Arena's interpreter running policy_runner.py / experiment_runner.py."""
import json, os, sys
argv = sys.argv[1:]
log = open(os.path.join(os.path.dirname(__file__), "calls.log"), "a")
log.write(json.dumps(argv) + "\\n")
with open(os.path.join(os.path.dirname(__file__), "env.log"), "a") as e:
    e.write(json.dumps({{"PYTHONPATH": os.environ.get("PYTHONPATH", "")}}) + "\\n")
fail = os.path.join(os.path.dirname(__file__), "FAIL")
if os.path.exists(fail):
    print("fake arena failing on purpose"); sys.exit(3)
def val(flag, default=None):
    return argv[argv.index(flag) + 1] if flag in argv else default
n = int(val("--num_episodes", 0)); envs = int(val("--num_envs", 1)); seed = int(val("--seed", 0))
out = os.path.join(val("--output_base_dir"), "2026_10_28-10_00_00")
os.makedirs(out, exist_ok=True)
dark = any(a.startswith("light.intensity") for a in argv)
with open(os.path.join(out, "episode_results_rank0.jsonl"), "w") as f:
    for k in range(n):
        rec = {{"env_id": k % envs, "episode_in_env": k // envs, "seed": seed, "episode_length": 300,
               "success": (k % 10 != 0) if not dark else (k % 10 == 0), "language_instruction": val("--language_instruction", "x"),
               "progress": {{"events": []}}}}
        if dark:
            rec["variations"] = {{"light.intensity": [100.0]}}
        f.write(json.dumps(rec) + "\\n")
print("fake arena ran", n, "episodes")
'''


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def box(tmp_path):
    """A fake GPU box: Arena root with a fake interpreter, a vla venv whose python is this one, a backend
    config written by the runbook's own helper, and the fake policy server on GR00T_PORT."""
    arena = tmp_path / "arena"
    (arena / ".venv" / "bin").mkdir(parents=True)
    fake = arena / ".venv" / "bin" / "python"
    fake.write_text(FAKE_ARENA.format(python=sys.executable))
    fake.chmod(0o755)
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").symlink_to(sys.executable)
    port = free_port()
    backend = tmp_path / "backend.yaml"
    assert tools.main(["write-backend-config", "--runbook", str(RUNBOOK), "--example",
                       str(REPO / "config" / "backend.example.yaml"), "--out", str(backend), "--arena-root", str(arena),
                       "--arena-python", str(fake), "--server-python", sys.executable, "--models-dir",
                       str(tmp_path / "models"), "--data-dir", str(tmp_path / "data"), "--repo-root", str(ROOT),
                       "--server-port", str(port + 1)]) == 0
    server = subprocess.Popen([sys.executable, str(FAKE_SERVER), "--port", str(port)], stdout=subprocess.PIPE, text=True)
    assert "listening" in server.stdout.readline()
    env = bare_env(tmp_path, ARENA_ROOT=arena, VLA_VENV=venv, VLA_BACKEND_CONFIG=backend, OMNI_KIT_ACCEPT_EULA="YES",
                   ACCEPT_EULA="Y", GR00T_PORT=port)
    env["PATH"] = f"{venv / 'bin'}:/usr/bin:/bin"
    yield {"env": env, "arena": arena, "tmp": tmp_path, "port": port}
    server.terminate()
    server.wait(10)


def test_r4_end_to_end_on_a_fake_box_and_idempotent(box):
    r = run("R4_nominal_clean.sh", box["env"])
    assert r.returncode == 0, r.stdout + r.stderr
    out = box["tmp"] / "meas" / "t0" / "R4_nominal_clean"
    assert r.stdout.splitlines()[0] == f"output dir: {out}" and r.stdout.splitlines()[-1] == f"output dir: {out}"
    argv = (out / "nominal" / "argv.txt").read_text().splitlines()
    joined = " ".join(argv)
    assert "--num_episodes 50" in joined and "--num_envs 1" in joined and "--seed 0" in joined
    assert "--viz none" in joined and f"--remote_port {box['port']}" in joined
    assert "--policy_type isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy" in joined
    assert argv[argv.index("--policy_type") + 1:].count("put_item_in_fridge_and_close_door") == 1
    res = json.loads((out / "result.json").read_text())
    cell = res["cells"]["nominal"]
    assert res["status"] == "ok" and (out / "DONE").exists()
    assert (cell["success"]["k"], cell["success"]["n"]) == (45, 50)
    w = wilson_interval(45, 50, 0.95)
    assert (cell["success"]["lower"], cell["success"]["upper"]) == (w.lo, w.hi)
    assert cell["timestamped_subdir"] is True and cell["success_types"] == ["bool"]
    calls = (box["arena"] / ".venv" / "bin" / "calls.log").read_text().splitlines()
    again = run("R4_nominal_clean.sh", box["env"])
    assert again.returncode == 0 and "already complete" in again.stdout
    assert (box["arena"] / ".venv" / "bin" / "calls.log").read_text().splitlines() == calls, "a finished run reran Arena"


def test_a_failed_cell_leaves_a_failed_result_and_is_retried(box):
    (box["arena"] / ".venv" / "bin" / "FAIL").write_text("")
    r = run("R7_dark_cell.sh", box["env"])
    out = box["tmp"] / "meas" / "t0" / "R7_dark_cell"
    assert r.returncode != 0 and "Arena exited 3" in r.stderr
    assert json.loads((out / "result.json").read_text())["status"] == "failed"
    assert (out / "dark" / "summary.failed.json").is_file() and not (out / "DONE").exists()
    (box["arena"] / ".venv" / "bin" / "FAIL").unlink()
    r = run("R7_dark_cell.sh", box["env"])
    assert r.returncode == 0, r.stderr
    argv = " ".join((out / "dark" / "argv.txt").read_text().splitlines())
    assert "light.intensity.enabled=true" in argv and "light.intensity.sampler_cfg.low=[100.0]" in argv
    cell = json.loads((out / "result.json").read_text())["cells"]["dark"]
    assert cell["variations_sample"] == {"light.intensity": [100.0]} and cell["success"]["k"] == 5
    assert list(out.glob("dark.failed-*"))


def test_scripts_needing_the_server_refuse_without_it(box):
    env = {**box["env"], "GR00T_PORT": str(free_port())}
    r = run("R5_nominal_incident.sh", env)
    assert r.returncode != 0 and "start it with R2_server.sh" in r.stderr


def test_r6_refuses_the_stock_servers_port(box, tmp_path):
    cfg = yaml.safe_load(Path(box["env"]["VLA_BACKEND_CONFIG"]).read_text())
    cfg["server"]["port"] = box["port"]
    Path(box["env"]["VLA_BACKEND_CONFIG"]).write_text(yaml.safe_dump(cfg))
    r = run("R6_replay_agreement.sh", box["env"])
    assert r.returncode != 0 and "stock server's port" in r.stderr


FAKE_SERVE = '''#!{python}
"""FIXTURE: stands in for the GR00T venv's python running ``-m policy_wrapper.serve``: logs its argv,
cwd and PYTHONPATH, then serves ping like the fake policy server on the requested port."""
import json, os, sys
argv = sys.argv[1:]
with open(os.path.join(os.path.dirname(__file__), "serve_calls.log"), "a") as f:
    f.write(json.dumps({{"argv": argv, "cwd": os.getcwd(), "PYTHONPATH": os.environ.get("PYTHONPATH", "")}}) + "\\n")
os.execv(sys.executable, [sys.executable, {server!r}, "--port", argv[argv.index("--port") + 1]])
'''


@pytest.fixture
def wrapper_box(box):
    """``box`` with a fake serve.py interpreter in the backend config's ``server_python``."""
    serve = box["tmp"] / "gr00t" / "python"
    serve.parent.mkdir()
    serve.write_text(FAKE_SERVE.format(python=sys.executable, server=str(FAKE_SERVER)))
    serve.chmod(0o755)
    path = Path(box["env"]["VLA_BACKEND_CONFIG"])
    cfg = yaml.safe_load(path.read_text())
    cfg["server_python"] = str(serve)
    path.write_text(yaml.safe_dump(cfg))
    return {**box, "serve_log": serve.parent / "serve_calls.log", "pw_port": cfg["server"]["port"]}


def test_wrapper_parity_end_to_end_on_a_fake_box_and_idempotent(wrapper_box):
    box = wrapper_box
    pw = (ROOT / "policy-wrapper").resolve()
    instruction = yaml.safe_load((pw / "config" / "eval.yaml").read_text())["client"]["language_instruction"]
    r = run("wrapper_parity.sh", box["env"], timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    out = box["tmp"] / "meas" / "t0" / "wrapper_parity"
    stock = (out / "stock" / "argv.txt").read_text().splitlines()
    wrap = (out / "wrapper" / "argv.txt").read_text().splitlines()
    for argv in (stock, wrap):
        joined = " ".join(argv)
        assert "--num_episodes 50" in joined and "--num_envs 1" in joined and "--seed 0" in joined
        assert argv[argv.index("--language_instruction") + 1] == instruction
    assert stock[stock.index("--policy_type") + 1].startswith("isaaclab_arena_gr00t.")
    assert stock[stock.index("--remote_port") + 1] == str(box["port"])
    assert wrap[wrap.index("--policy_type") + 1] == \
        "policy_wrapper.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy"
    assert wrap[wrap.index("--policy_config_yaml_path") + 1] == str(pw / "config" / "eval.yaml")
    assert wrap[wrap.index("--remote_port") + 1] == str(box["pw_port"]) != str(box["port"])
    (serve,) = [json.loads(line) for line in box["serve_log"].read_text().splitlines()]
    assert serve["argv"][:2] == ["-m", "policy_wrapper.serve"] and serve["cwd"] == str(pw)
    assert serve["PYTHONPATH"] == str(pw) and serve["argv"][serve["argv"].index("--port") + 1] == str(box["pw_port"])
    assert "--checkpoint-dir" in serve["argv"]
    envs = [json.loads(line)["PYTHONPATH"] for line in (box["arena"] / ".venv" / "bin" / "env.log").read_text().splitlines()]
    assert str(pw) not in envs[0].split(":") and envs[1].split(":")[0] == str(pw)
    assert not tools.main(["ping", "--host", "127.0.0.1", "--port", str(box["pw_port"]), "--timeout-ms", "300"]) == 0, \
        "serve.py was left running"
    rep = json.loads((out / "parity.json").read_text())
    w = wilson_interval(45, 50, 0.95)
    assert rep["stock"]["k"] == rep["wrapper"]["k"] == 45 and (rep["stock"]["lower"], rep["stock"]["upper"]) == (w.lo, w.hi)
    assert rep["agreement"]["k"] == rep["agreement"]["n"] == 50 and rep["parity"] is True and rep["same_instruction"]
    assert len(rep["per_seed"]) == 50 and rep["per_seed"]["0:0"] == {"stock": False, "wrapper": False, "agree": True}
    assert "parity: True" in r.stdout
    res = json.loads((out / "result.json").read_text())
    assert res["status"] == "ok" and res["parity"]["parity"] is True
    again = run("wrapper_parity.sh", box["env"], timeout=300)
    assert again.returncode == 0 and "already complete" in again.stdout
    assert len(box["serve_log"].read_text().splitlines()) == 1


def test_wrapper_parity_reuses_r4_only_with_the_same_instruction(wrapper_box):
    box = wrapper_box
    env = {**box["env"], "RUN_TS": "t0"}
    assert run("R4_nominal_clean.sh", env).returncode == 0
    r = run("wrapper_parity.sh", env, timeout=300)
    assert r.returncode == 0, r.stderr
    assert "R4 logged a different instruction" in r.stdout             # R4 passes no --language_instruction
    assert not (box["tmp"] / "meas" / "t0" / "wrapper_parity" / "stock" / "reused_from.txt").exists()


def test_wrapper_parity_refuses_the_stock_port_and_a_busy_wrapper_port(wrapper_box):
    box = wrapper_box
    path = Path(box["env"]["VLA_BACKEND_CONFIG"])
    cfg = yaml.safe_load(path.read_text())
    cfg["server"]["port"] = box["port"]
    path.write_text(yaml.safe_dump(cfg))
    r = run("wrapper_parity.sh", box["env"])
    assert r.returncode != 0 and "stock server's port" in r.stderr
    cfg["server"]["port"] = box["pw_port"]
    path.write_text(yaml.safe_dump(cfg))
    busy = subprocess.Popen([sys.executable, str(FAKE_SERVER), "--port", str(box["pw_port"])], stdout=subprocess.PIPE,
                            text=True)
    try:
        assert "listening" in busy.stdout.readline()
        r = run("wrapper_parity.sh", {**box["env"], "RUN_TS": "t1"}, timeout=300)
        assert r.returncode != 0 and "already answers" in r.stderr
    finally:
        busy.terminate()
        busy.wait(10)


def summary(outcomes, instruction="x", complete=True):
    from vla_isaac.experiments.tables import proportion

    oc = {f"0:{i}": s for i, s in enumerate(outcomes)}
    return {"outcomes": oc, "success": proportion(sum(outcomes), len(outcomes), 0.95), "complete": complete,
            "language_instructions": [instruction]}


def test_wrapper_parity_report():
    stock = summary([True] * 40 + [False] * 10)
    same = tools.wrapper_parity_report(stock, summary([True] * 38 + [False] * 12))
    assert same["parity"] and same["intervals_overlap"] and not same["differ"]
    assert (same["agreement"]["k"], same["agreement"]["n"]) == (48, 50)
    assert (same["only_stock"], same["only_wrapper"], same["both_success"], same["both_failure"]) == (2, 0, 38, 10)
    assert same["per_seed"]["0:39"] == {"stock": True, "wrapper": False, "agree": False}
    w = wilson_interval(48, 50, 0.95)
    assert (same["agreement"]["lower"], same["agreement"]["upper"]) == (w.lo, w.hi)
    broken = tools.wrapper_parity_report(stock, summary([True] * 10 + [False] * 40))
    assert not broken["parity"] and not broken["intervals_overlap"] and broken["differ"]
    assert "parity: False" in tools.wrapper_parity_line(broken)
    short = tools.wrapper_parity_report(stock, summary([True] * 40 + [False] * 10, complete=False))
    assert not short["parity"] and short["intervals_overlap"]
    other = tools.wrapper_parity_report(stock, summary([True] * 40 + [False] * 10, instruction="y"))
    assert not other["same_instruction"] and "different instructions" in tools.wrapper_parity_line(other)
    none = tools.wrapper_parity_report(stock, {"outcomes": {"1:0": True}, "success": stock["success"]})
    assert none["comparable"] is False and none["parity"] is False


# --------------------------------------------------------------------------- #
# runbook_tools helpers
# --------------------------------------------------------------------------- #

def jsonl(path: Path, records) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


def recs(outcomes, envs=1, seed=0):
    return [{"env_id": k % envs, "episode_in_env": k // envs, "seed": seed, "success": s} for k, s in enumerate(outcomes)]


def test_the_runbook_config_is_strict_and_resolves_paths(tmp_path):
    rb = tools.load_runbook(RUNBOOK)
    for key in ("task_config", "grid", "pipeline", "factor_names"):
        assert Path(rb[key]).is_file(), key
    assert rb["nominal"]["episodes"] == 50
    bad = yaml.safe_load(RUNBOOK.read_text())
    bad["surprise"] = 1
    (tmp_path / "r.yaml").write_text(yaml.safe_dump(bad))
    with pytest.raises(tools.RunbookError, match="surprise"):
        tools.load_runbook(tmp_path / "r.yaml")


def test_summarize_and_compare(tmp_path):
    a = jsonl(tmp_path / "a" / "2026_x" / "episode_results_rank0.jsonl", recs([True] * 8 + [False] * 2))
    b = jsonl(tmp_path / "b" / "episode_results_rebuild0.jsonl", recs([True] * 7 + [False] * 3))
    sa = tools.summarize_run(a.parent.parent, episodes=10, num_envs=1, exit_code=0)
    sb = tools.summarize_run(b.parent, episodes=10, num_envs=1, exit_code=0)
    assert sa["timestamped_subdir"] is True and sb["timestamped_subdir"] is False
    assert sa["success"]["k"] == 8 and sa["complete"]
    c = tools.compare_runs(sa, sb)
    assert c["shared_episodes"] == 10 and c["agreement"] == 0.9 and c["only_first"] == 1 and c["only_second"] == 0
    assert c["fisher_p"] == fisher_exact(8, 10, 7, 10) and c["intervals_overlap"] and not c["differ"]
    short = tools.summarize_run(a.parent.parent, episodes=20, num_envs=1, exit_code=0)
    assert short["complete"] is False
    jsonl(tmp_path / "bad" / "episode_results_rank0.jsonl", [{"env_id": 0, "episode_in_env": 0, "seed": 0, "success": None}])
    assert "not a bool" in tools.summarize_run(tmp_path / "bad", episodes=1, num_envs=1, exit_code=0)["error"]
    log = tmp_path / "x.log"
    log.write_text("error: unrecognized arguments: --viz none\n")
    assert tools.summarize_run(tmp_path / "b", episodes=10, num_envs=1, exit_code=2, log=str(log))[
        "unrecognized_arguments"] == ["--viz none"]


def test_incident_mode_summary_ranks_by_episode_then_env(tmp_path):
    d = jsonl(tmp_path / "r" / "episode_results_rank0.jsonl", recs([True, False] * 30, envs=10)).parent
    s = tools.summarize_run(d, episodes=50, num_envs=10, exit_code=0)
    assert s["success"]["n"] == 50 and s["records"] == 60 and s["success"]["k"] == 25


def test_instruction_strings_line_and_pairs(tmp_path):
    base = tmp_path / "pw"
    (base / "config").mkdir(parents=True)
    (base / "config" / "eval.yaml").write_text(yaml.safe_dump({"client": {"language_instruction": "sauce"}}))
    ds = write_lerobot(tmp_path / "ds", 2, task="ranch dressing")
    backend = tmp_path / "b.yaml"
    cfg = yaml.safe_load((REPO / "config" / "backend.example.yaml").read_text())
    cfg.update(task_config=str(REPO / "config" / "pilot_gr1_fridge.yaml"), workspace_bases={"pw": str(base)},
               datasets={"arena_gr1_ranch_bottle_lerobot": str(ds)})
    backend.write_text(yaml.safe_dump(cfg))
    info = tools.instruction_strings(tools.load_runbook(RUNBOOK), str(backend))
    conv = "Place the ranch dressing bottle on the top shelf of the fridge, and close the fridge door."
    assert info["sources"] == {"eval_yaml": "sauce", "converter_yaml": conv, "tasks_jsonl": "ranch dressing"}
    assert info["distinct"] == ["sauce", conv, "ranch dressing"]
    assert set(info["differ"]) == {"eval_yaml!=converter_yaml", "eval_yaml!=tasks_jsonl", "converter_yaml!=tasks_jsonl"}
    line = tools.instruction_line(info)
    assert "\n" not in line and "eval_yaml='sauce'" in line and "differ: " in line
    assert tools.nominal_task(str(backend), str(REPO / "config" / "finetune_gr1_fridge.yaml")) == "ranch dressing"
    cfg["datasets"] = {}
    backend.write_text(yaml.safe_dump(cfg))
    with pytest.raises(tools.RunbookError, match="R1b"):
        tools.instruction_strings(tools.load_runbook(RUNBOOK), str(backend))


def _write_json(path, obj):
    path.write_text(json.dumps(obj))
    return path


def test_vram_client_tests_checkpoint_and_lerobot_pick(tmp_path):
    csv = tmp_path / "v.csv"
    csv.write_text("2026/10/28 10:00:00.000, 0, 1000, 49140\n2026/10/28 10:00:05.000, 0, 21000, 49140\nbad line\n")
    v = tools.vram_summary(csv)
    assert v["peak_used_mib"] == 21000 and v["gpus"]["0"]["mean_used_mib"] == 11000 and v["samples"] == 2
    blocked = tools.client_tests_report("E   ModuleNotFoundError: No module named 'isaaclab.app'\n1 error in 0.1s", 2)
    assert blocked["blocked_by_isaac"] and blocked["isaac_modules_missing"] == ["isaaclab"] and not blocked["imports_without_isaac"]
    passed = tools.client_tests_report("12 passed, 1 skipped in 3.2s", 0)
    assert passed["imports_without_isaac"] and passed["counts"] == {"passed": 12, "skipped": 1}
    other = tools.client_tests_report("No module named 'gr00t'\n1 error", 2)
    assert not other["blocked_by_isaac"] and other["missing_modules"] == ["gr00t"]
    assert passed["settled"] and not blocked["settled"] and not other["settled"]
    assert not tools.client_tests_report("/x/python: No module named pytest", 1)["settled"]
    assert not tools.client_tests_report("1 error in 0.3s\nERROR collecting tests", 2)["settled"]
    res = tools.write_result(tmp_path, "check_client_tests", run_ts="t", jsons={"gr00t": str(_write_json(tmp_path / "g.json", {"settled": False}))})
    assert json.loads(res.read_text())["status"] == "not_settled"
    ck = tools.checkpoint_report(REPO / "tests" / "fixtures" / "checkpoint_gr1")
    assert all(ck["files"].values()) and isinstance(ck["tune_top_llm_layers"], int)
    assert set(ck["statistics_shape"]) == {"gr1"}
    files = ["a/meta/tasks.jsonl", "ranch_bottle_into_fridge/lerobot/meta/tasks.jsonl", "x/y.hdf5"]
    assert tools.pick_lerobot(files, "ranch") == "ranch_bottle_into_fridge/lerobot"
    with pytest.raises(tools.RunbookError, match="LEROBOT_PREFIX"):
        tools.pick_lerobot(files + ["ranch2/meta/tasks.jsonl"], "ranch")


def mimic_run_dir(out: Path, name: str, *, brightness: int, outcomes, seed_content: int, trials: int,
                  frames: bool = True) -> None:
    import h5py
    import numpy as np

    d = out / name
    d.mkdir(parents=True)
    with h5py.File(d / "generated.hdf5", "w") as f:
        f.attrs["created"] = f"{time.time()}"              # a timestamp attribute must not break equality
        for i in range(2):
            if frames:
                f.create_dataset(f"data/demo_{i}/camera_obs/robot_pov_cam_rgb",
                                 data=np.full((4, 8, 8, 3), brightness, dtype=np.uint8))
            f.create_dataset(f"data/demo_{i}/processed_actions", data=np.full((4, 36), seed_content, dtype=np.float32))
    s = a = 0
    lines = []
    for ok in outcomes:
        a += 1
        s += ok
        lines.append(f"{s}/{a} ({100 * s / a:.1f}%) successful demos generated by mimic")
    (d / "mimic.log").write_text("\n".join(["[INFO] start"] + lines) + "\n")
    (d / "trials").write_text(str(trials))
    (d / "exit_code").write_text("0")
    probe = tools.h5_probe(d / "generated.hdf5", "robot_pov_cam_rgb", frame_stride=1)
    (d / "probe.json").write_text(json.dumps(probe))


def test_h5_probe_ignores_attributes_and_finds_variation_keys(tmp_path):
    import h5py

    mimic_run_dir(tmp_path, "a", brightness=200, outcomes=[True], seed_content=1, trials=1)
    time.sleep(0.01)
    mimic_run_dir(tmp_path, "b", brightness=200, outcomes=[True], seed_content=1, trials=1)
    pa, pb = (json.loads((tmp_path / n / "probe.json").read_text()) for n in "ab")
    assert pa["digest"] == pb["digest"] and pa["mean_pixel"] == 200 and pa["demos"] == 2
    with h5py.File(tmp_path / "a" / "generated.hdf5", "a") as f:
        f["data"].attrs["light.intensity variation"] = "[100.0]"
    assert tools.h5_probe(tmp_path / "a" / "generated.hdf5", "robot_pov_cam_rgb")["recorded_variations"]


@pytest.mark.parametrize("case, flag_in_help, expect_mode", [
    ("fixed", True, "fixed"),            # different seeds give the same runs: the flag does not reach the RNG
    ("per_run", True, "per_run"),
    ("no_flag", False, "fixed"),         # no flag: the repeat decides
    ("unstable", True, "stop"),          # the same seed differs: not deterministic either
])
def test_mimic_check_result(tmp_path, case, flag_in_help, expect_mode):
    rb = tools.load_runbook(RUNBOOK)
    mimic_run_dir(tmp_path, "override", brightness=40, outcomes=[True, False], seed_content=9, trials=2)
    mimic_run_dir(tmp_path, "seed_a", brightness=200, outcomes=[True, True, False, True, True], seed_content=1, trials=5)
    mimic_run_dir(tmp_path, "seed_a_repeat", brightness=200, outcomes=[True, True, False, True, True],
                  seed_content=2 if case == "unstable" else 1, trials=5)
    if flag_in_help:
        mimic_run_dir(tmp_path, "seed_b", brightness=200, outcomes=[True, False, True, True, True],
                      seed_content=1 if case == "fixed" else 3, trials=5)
        if case == "fixed":
            shutil.rmtree(tmp_path / "seed_b")
            mimic_run_dir(tmp_path, "seed_b", brightness=200, outcomes=[True, True, False, True, True], seed_content=1, trials=5)
    res = tools.mimic_check_result(tmp_path, rb, flag="--seed", flag_in_help=flag_in_help)
    assert res["override"]["took_effect"] is True and res["override"]["ratio"] == pytest.approx(0.2)
    assert res["override"]["result"] == "pass"
    assert res["seed"]["mode"].startswith(expect_mode)
    assert res["stdout"]["parse_ok"] and res["stdout"]["per_attempt_lines"]
    assert res["guarantee"][0] == {"trials_asked": 2, "attempts": 2, "successes": 1}
    assert res["runs"]["seed_a"]["log"]["outcomes"] == [True, True, False, True, True]


def test_mimic_check_flags_an_override_that_did_not_take_and_a_summary_only_log(tmp_path):
    rb = tools.load_runbook(RUNBOOK)
    mimic_run_dir(tmp_path, "override", brightness=195, outcomes=[True, True], seed_content=9, trials=2)
    mimic_run_dir(tmp_path, "seed_a", brightness=200, outcomes=[True] * 5, seed_content=1, trials=5)
    (tmp_path / "seed_a" / "mimic.log").write_text("Generated 5 demos in total\n")
    res = tools.mimic_check_result(tmp_path, rb, flag="--seed", flag_in_help=False)
    assert res["override"]["took_effect"] is False and res["override"]["result"] == "fail"
    assert res["stdout"]["parse_ok"] is False
    assert res["seed"]["mode"] is None                     # no repeat run: undecided


@pytest.mark.parametrize("which", ["override", "seed_a", "both", "black"])
def test_mimic_check_without_camera_frames_is_inconclusive_not_pass_or_fail(tmp_path, which):
    rb = tools.load_runbook(RUNBOOK)
    mimic_run_dir(tmp_path, "override", brightness=40, outcomes=[True, True], seed_content=9, trials=2,
                  frames=which not in ("override", "both"))
    mimic_run_dir(tmp_path, "seed_a", brightness=0 if which == "black" else 200, outcomes=[True] * 5, seed_content=1,
                  trials=5, frames=which not in ("seed_a", "both"))
    res = tools.mimic_check_result(tmp_path, rb, flag="--seed", flag_in_help=False)
    ov = res["override"]
    assert ov["result"] == "inconclusive" and ov["took_effect"] is None and ov["method"] == "none"
    assert "ratio" not in ov
    assert ("black" in ov["reason"]) if which == "black" else ("no camera frames" in ov["reason"])


def test_the_isaac_import_blocker(tmp_path):
    script = tmp_path / "s.py"
    script.write_text("import json, sys\nprint('ran', sys.argv[1:])\nimport isaacsim\n")
    r = subprocess.run(tools.isaac_blocked_argv([sys.executable, str(script), "--yaml_file", "y"]),
                       capture_output=True, text=True)
    assert "ran ['--yaml_file', 'y']" in r.stdout
    assert r.returncode != 0 and "blocked Isaac module isaacsim (V-converter-noisaac)" in r.stderr


def test_pipeline_variant_caps_steps_and_stays_valid(tmp_path):
    from vla_isaac.data.config import PipelineConfig

    p = tools.pipeline_variant(REPO / "config" / "finetune_gr1_fridge.yaml", tmp_path / "x" / "p.yaml", max_steps=50,
                               timeout_s=600)
    pipe = PipelineConfig.load(p)
    assert all(hw["max_steps"] == 50 and hw["save_steps"] == 50 for hw in pipe.finetune.hardware.values())
    assert pipe.finetune.timeout_s == 600 and pipe.rung_b.converter.template.is_file()


def test_mix_dryrun_and_report(tmp_path):
    nominal = write_lerobot(tmp_path / "nominal", 6, task="t")
    shifted = write_lerobot(tmp_path / "shifted", 2, task="t", tag="s")
    backend = tmp_path / "b.yaml"
    cfg = yaml.safe_load((REPO / "config" / "backend.example.yaml").read_text())
    cfg["datasets"] = {"arena_gr1_ranch_bottle_lerobot": str(nominal)}
    backend.write_text(yaml.safe_dump(cfg))
    rb = tools.load_runbook(RUNBOOK)
    m = tools.mix_dryrun(rb, str(backend), rb["pipeline"], str(shifted), str(tmp_path / "mixed"))
    assert m["selection"]["nominal_count"] == 2 and m["selection"]["shifted_count"] == 2
    assert m["stats_json_present"] is False
    ft = tmp_path / "ft"
    shutil.copytree(REPO / "tests" / "fixtures" / "finetune_run", ft)
    manifest = {"tune_top_llm_layers": {"value": 4}, "result": {"parameters": {"trainable": 10, "total": 100},
                                                               "final_model_config": {"x": 1}, "complete": True}}
    (ft / "vla_finetune_manifest.json").write_text(json.dumps(manifest))
    with open(ft / "finetune.log", "a") as f:
        f.write(f"Generating stats for {tmp_path / 'mixed'}\n")
    assert tools.dryrun_report(tmp_path / "mixed", ft, 0, 50)["stats_json_after"] is False
    (tmp_path / "mixed" / "meta" / "stats.json").write_text("{}")
    rep = tools.dryrun_report(tmp_path / "mixed", ft, 0, 50)
    assert rep["trained"] and rep["stats_json_after"] and rep["stats_generated_line"]
    assert rep["parameters"] == {"trainable": 10, "total": 100}


def test_write_result_collects_cells_and_values(tmp_path):
    (tmp_path / "c1").mkdir()
    (tmp_path / "c1" / "summary.json").write_text(json.dumps({"success": {"k": 1}}))
    (tmp_path / "c2").mkdir()                              # no summary: not a cell
    (tmp_path / "x.json").write_text(json.dumps({"a": 1}))
    p = tools.write_result(tmp_path, "R4_nominal_clean", run_ts="t", jsons={"extra": str(tmp_path / "x.json")},
                           sets={"n": "3", "s": "text"})
    res = json.loads(p.read_text())
    assert res["status"] == "ok" and list(res["cells"]) == ["c1"] and res["extra"] == {"a": 1}
    assert res["n"] == 3 and res["s"] == "text"


# --------------------------------------------------------------------------- #
# collect_measurements.py on fake measurement directories
# --------------------------------------------------------------------------- #

GATE = {"nominal_min": 0.8, "shifted_max": 0.4, "confidence": 0.95, "alpha": 0.05}


def prop(k, n):
    w = wilson_interval(k, n, 0.95)
    return {"k": k, "n": n, "rate": k / n, "lower": w.lo, "upper": w.hi, "confidence": 0.95}


def put(meas: Path, ts: str, script: str, **res) -> Path:
    d = meas / ts / script
    d.mkdir(parents=True)
    body = {"script": script, "run_ts": ts, "status": "ok", **res}
    (d / "result.json").write_text(json.dumps(body))
    return d


def full_box(meas: Path, *, r4=(45, 50), dark=(5, 50), r5=(30, 50), took_effect=True, mode="fixed") -> None:
    result = {True: "pass", False: "fail", None: "inconclusive"}[took_effect]
    put(meas, "t1", "00_check_gpu", gpus=[{"name": "NVIDIA RTX 6000 Ada Generation", "ok": True}])
    put(meas, "t1", "R0_setup", versions={"isaacsim": "5.1.0", "isaaclab": "2.3.0"})
    put(meas, "t1", "R1_download_checkpoint", revision_sha="abc123",
        checkpoint={"files": {"config.json": True}, "tune_top_llm_layers": 4, "statistics_shape": {"gr1": {}}})
    put(meas, "t1", "R1b_download_datasets", tasks=["ranch"], mimic_source="/d/m.hdf5")
    put(meas, "t1", "check_client_tests", vla={"imports_without_isaac": True}, gr00t={"imports_without_isaac": True})
    put(meas, "t1", "R3_list_variations", factor_table_equal=True, regenerated="/m/f.yaml")
    put(meas, "t1", "mimic_override_check", check={
        "override": {"took_effect": took_effect, "result": result, "reason": "no camera frames in run(s) ['override']"},
        "seed": {"mode": mode}, "stdout": {"parse_ok": True},
        "guarantee": [{"trials_asked": 2, "attempts": 2, "successes": 2}]})
    put(meas, "t1", "R4_nominal_clean", cells={"nominal": {"success": prop(*r4), "timestamped_subdir": True,
                                                           "success_types": ["bool"],
                                                           "progress_sample": {"success": {"events": []}, "failure": None}}})
    put(meas, "t1", "R8_vram", vram={"peak_used_mib": 30000})
    put(meas, "t1", "R7_dark_cell", cells={"dark": {"success": prop(*dark), "variations_sample": {"light.intensity": [100.0]}}})
    put(meas, "t1", "R7b_hdr_and_colour_temp", cells={"hdr_home_office": {"success": prop(40, 50)}})
    put(meas, "t1", "instruction_check", strings={"sources": {"eval_yaml": "sauce", "converter_yaml": "ranch",
                                                               "tasks_jsonl": "ranch"}, "distinct": ["sauce", "ranch"]},
        cells={"instruction_0": {"success": prop(16, 20)}, "instruction_1": {"success": prop(18, 20)}})
    put(meas, "t1", "runner_parity", comparison={"agreement": 0.9})
    put(meas, "t1", "wrapper_parity", parity={"parity": True, "stock": prop(45, 50), "wrapper": prop(44, 50)})
    put(meas, "t1", "R5_nominal_incident", cells={"nominal": {"success": prop(*r5)}})
    put(meas, "t1", "R6_replay_agreement", agreement={"pooled_agreement": prop(48, 50)})
    put(meas, "t1", "finetune_dryrun", convert={"ok": True}, dryrun={"trained": True, "stats_json_after": True,
                                                                     "parameters": {"trainable": 10, "total": 100}})


def test_collect_on_an_empty_directory_leaves_everything_open(tmp_path):
    s = collect.summarize(tmp_path, VERIFY_MD, GATE)
    assert s["settled"] == [] and set(s["open"]) == set(collect.verify_ids(VERIFY_MD))
    assert all(d["decision"] == "undecided" for d in s["decisions"].values())
    assert s["verify"]["V-actuator"]["reason"] == "no runbook script measures it"
    assert s["verify"]["V-viz"]["reason"] == "R4_nominal_clean has not run"


def test_collect_on_a_full_box(tmp_path):
    full_box(tmp_path)
    s = collect.summarize(tmp_path, VERIFY_MD, GATE)
    assert set(s["settled"]) == set(collect.SETTLES)
    assert {"V-actuator", "V-ext-env", "V-hdf5-keys", "V-server-seed"} <= set(s["open"])
    assert set(s["settled"]) | set(s["open"]) == set(collect.verify_ids(VERIFY_MD))
    d = s["decisions"]
    assert d["pilot"]["decision"] == "pilot stays" and d["pilot"]["failing_cells"] == ["dark"]
    assert d["mimic"]["decision"] == "usable (reference.mimic.seed.mode: fixed)"
    assert d["mode_gap"]["decision"] == "real" and d["mode_gap"]["fisher_p"] == fisher_exact(45, 50, 30, 50)
    assert d["instruction"]["decision"].startswith("switch eval.yaml to the training string 'ranch'")
    assert d["merged_dataset"]["decision"] == "trains"
    assert s["verify"]["V-hf-rev"]["value"] == "abc123"
    assert s["verify"]["V-var-record"]["value"] == {"light.intensity": [100.0]}
    assert s["measurements"]["vram"] == {"peak_used_mib": 30000}


@pytest.mark.parametrize("kw, expect", [
    ({"r4": (35, 50)}, "switch to DROID"),
    ({"dark": (40, 50)}, "switch to DROID"),           # no shifted cell drops: the family does not exist
    ({"r4": (41, 50)}, "pilot stays"),                  # 0.82 passes; the interval note is added
])
def test_the_pilot_decision(tmp_path, kw, expect):
    full_box(tmp_path, **kw)
    d = collect.summarize(tmp_path, VERIFY_MD, GATE)["decisions"]["pilot"]
    assert d["decision"] == expect
    if kw.get("r4") == (41, 50):
        assert "lower bound" in d["reason"]


def test_an_inconclusive_override_check_is_its_own_mimic_decision_and_settles_the_id_as_inconclusive(tmp_path):
    full_box(tmp_path, took_effect=None)
    s = collect.summarize(tmp_path, VERIFY_MD, GATE)
    d = s["decisions"]["mimic"]
    assert d["decision"] == "inconclusive" and "no camera frames" in d["reason"]
    assert d["override_result"] == "inconclusive"
    assert s["verify"]["V-mimic-override"]["value"] == "inconclusive"


def test_mimic_and_mode_gap_negative_decisions(tmp_path):
    full_box(tmp_path, took_effect=False, r5=(43, 50))
    d = collect.summarize(tmp_path, VERIFY_MD, GATE)["decisions"]
    assert d["mimic"]["decision"] == "not usable"
    assert d["mode_gap"]["decision"] == "not detected"


def test_the_newest_finished_run_wins_and_a_failed_one_leaves_ids_open(tmp_path):
    put(tmp_path, "t1", "R1_download_checkpoint", revision_sha="old")
    put(tmp_path, "t2", "R1_download_checkpoint", revision_sha="new")
    (put(tmp_path, "t3", "R1_download_checkpoint") / "result.json").write_text(json.dumps({"status": "failed"}))
    put(tmp_path, "t3", "R4_nominal_clean", status="failed", exit_code=1)
    s = collect.summarize(tmp_path, VERIFY_MD, GATE)
    assert s["verify"]["V-hf-rev"]["value"] == "new"
    assert s["verify"]["V-viz"]["state"] == "open" and s["verify"]["V-viz"]["reason"] == "R4_nominal_clean failed"
    assert len(s["runs"]) == 4


def test_collect_cli_writes_the_summary(tmp_path, capsys):
    full_box(tmp_path / "m")
    assert collect.main(["--measurements", str(tmp_path / "m"), "--verify", str(VERIFY_MD), "--runbook", str(RUNBOOK)]) == 0
    s = json.loads((tmp_path / "m" / "summary.json").read_text())
    assert s["decisions"]["pilot"]["decision"] == "pilot stays"
    printed = capsys.readouterr().out
    assert "settled" in printed and "open" in printed and "pilot: pilot stays" in printed
