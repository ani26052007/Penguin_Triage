"""Fixtures for Fix F2a: a real base checkout, a WorkspaceManager on it, and the combined case
(code bug + lighting) wired to the code's FILES, so the real HagentRepairer's patches change the
mock success rates and the synthetic logs whatever names they get.

The checkout: ``controller/wrapper.py`` hands raw observations to the policy (the bug the
observation contract finds as unnormalised inputs). Two fixes are planted as file predicates:

    STEP_A   "normalize_inputs(obs)"                  a half fix: nominal 0.4 (ambiguous Rung 1)
    FIXED    "normalize_inputs(obs, self.norm_stats)" the full fix: nominal 0.9

and the lit scene stays at 0.35 with either (lighting alone still hurts).
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Callable

from hagent.config import Limits, Settings
from hagent.env import SUBMIT_MARKER
from hagent.models import bash
from ladder.backend import MockRule
from ladder.workspace import FileContains, WorkspaceManager
from router.schemas import CodeRef
from router.stage1.synthetic import Faults
from scripts.confusion_matrix import LIGHT, _REF_OK
from tests.router_cases import RouterCase, router_cases

WRAPPER = "controller/wrapper.py"
BUG_LINE = "return dict(obs)  # raw camera values go straight to the policy"
STEP_A_LINE = "return normalize_inputs(obs)  # TODO: training stats"
FIXED_LINE = "return normalize_inputs(obs, self.norm_stats)"
STEP_A = FileContains(WRAPPER, "normalize_inputs(obs)")
FIXED = FileContains(WRAPPER, "normalize_inputs(obs, self.norm_stats)")
BUGGY = FileContains(WRAPPER, BUG_LINE)

WRAPPER_SRC = f'''"""Policy wrapper: observation in, action out."""

from controller.norm import normalize_inputs


class PolicyWrapper:
    def __init__(self, policy, norm_stats):
        self.policy, self.norm_stats = policy, norm_stats

    def to_policy_input(self, obs):
        {BUG_LINE}

    def act(self, obs):
        return self.policy(self.to_policy_input(obs))
'''
NORM_SRC = '''def normalize_inputs(obs, stats=None):
    stats = stats or {}
    return {k: (v - stats.get(k, (0.0, 1.0))[0]) / stats.get(k, (0.0, 1.0))[1] for k, v in obs.items()}
'''
EVAL_CONFIG = "config/eval.yaml"
BASE_FILES = {EVAL_CONFIG: "render:\n  preset: eval_fast\n", WRAPPER: WRAPPER_SRC, "controller/norm.py": NORM_SRC, "controller/__init__.py": "",
              "README.md": "controller repo\n", ".gitignore": "secrets.txt\n__pycache__/\n"}
BLOB = "assets/blob.bin"


def git(cwd: Path, *args: str) -> str:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd, check=True,
                          capture_output=True, text=True, env=env).stdout


def make_checkout(root: Path, files: dict[str, str] = BASE_FILES, *, binary: bool = True) -> Path:
    """A git checkout with two commits of history (one names the bug), an ignored secret and
    a binary file."""
    root.mkdir(parents=True)
    git(root, "init", "-q")
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    if binary:
        (root / BLOB).parent.mkdir(parents=True, exist_ok=True)
        (root / BLOB).write_bytes(bytes(range(0, 256)))
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "initial import")
    (root / "README.md").write_text("controller repo\nsee BUG_LEDGER: wrapper skips normalisation\n")
    git(root, "commit", "-q", "-am", "BUG_LEDGER: wrapper skips normalisation")
    (root / "secrets.txt").write_text("api-key\n")                    # ignored, never copied
    return root


def snapshot(root: Path) -> dict[str, bytes]:
    """Every file under ``root`` except ``.git``, as bytes."""
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*"))
            if p.is_file() and ".git" not in p.relative_to(root).parts}


def manager(tmp: Path, checkout: Path) -> WorkspaceManager:
    return WorkspaceManager(tmp / "store", {"c0": checkout})


def hagent_settings(**limits) -> Settings:
    base = dict(max_steps=6, max_cost=0.0, max_wall_s=0, cmd_timeout_s=10, plan_at_step=0, final_review=False,
                ctx_char_limit=0, enable_explorer=False)
    base.update(limits)
    return Settings(limits=Limits(**base))


SUBMIT = bash(f"echo {SUBMIT_MARKER} && git diff HEAD", "Done, submitting.")


def replace_line(old: str, new: str) -> str:
    return bash(f"python3 - <<'PY'\nimport pathlib\np = pathlib.Path('{WRAPPER}')\n"
                f"p.write_text(p.read_text().replace({old!r}, {new!r}))\nPY", "Editing the wrapper.")


def recording(seen: list, reply: str) -> Callable:
    """A scripted reply that first records the messages the agent sees."""
    return lambda messages: seen.append([dict(m) for m in messages]) or reply


def files_combined_case(code_ref: CodeRef, *, severe: bool = False) -> RouterCase:
    """The combined case on ``code_ref`` (a workspace name, e.g. ``mgr.base_ref("c0")``) with
    patched-code rules and logs keyed on the files, not on code names: code with BUG_LINE logs
    unnormalised inputs; code with STEP_A or FIXED logs healthy ones; anything else has no logs."""
    base = next(c for c in router_cases() if c.name == "combined_code_plus_light")
    ref, rest = base.rules[:len(_REF_OK)], base.rules[len(_REF_OK):]
    patched = (MockRule(0.35, files=FIXED, factors_present=(LIGHT,)), MockRule(0.9, files=FIXED),
               MockRule(0.35, files=STEP_A, factors_present=(LIGHT,)), MockRule(0.4, files=STEP_A))
    return replace(base, name="combined_files" + ("_severe" if severe else ""), rules=ref + patched + rest,
                   default_rate=0.10 if severe else base.default_rate, code_ref=code_ref,
                   trace_faults={}, trace_default=None,
                   trace_file_faults=((FIXED, Faults()), (STEP_A, Faults()), (BUGGY, Faults(unnormalized=True))))


def fixed_files(checkout: Path) -> dict[str, bytes]:
    """What the fully fixed code must contain: the base snapshot with the bug line replaced."""
    files = {k: v for k, v in snapshot(checkout).items() if k != "secrets.txt"}
    files[WRAPPER] = files[WRAPPER].replace(BUG_LINE.encode(), FIXED_LINE.encode())
    return files
