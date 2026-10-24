"""No LLM output can change a route, structurally (plan.md 3.2.12: "None of them decides a route").

The modules that decide (decide, router, stage0/1/2, branches, replay) and the behavior
analysis must not import the LLM layer or hagent; the LLM modules must not import ``decide``
or ``router.router``. Files that are not part of this checkout are skipped.
"""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1] / "router"
DECIDING = ["decide.py", "router.py", "stage0.py", "stage2.py", "replay.py", "behavior.py", "behavior_synthetic.py",
            "prereq/behavior_trace.py", "branches/branch_c.py", "branches/branch_g.py",
            *[f"stage1/{p.name}" for p in (ROOT / "stage1").glob("*.py")]]
LLM_SIDE = ["llm.py", "report.py", "config_mapper.py", "factor_search.py"]


def imports(path: Path) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.add(node.module)
            out |= {f"{node.module}.{a.name}" for a in node.names}
    return out


@pytest.mark.parametrize("rel", DECIDING)
def test_deciding_code_does_not_import_the_llm_layer(rel):
    p = ROOT / rel
    if not p.exists():
        pytest.skip(f"{rel} not in this checkout")
    bad = {m for m in imports(p) if m.split(".")[0] == "hagent" or m in (
        "router.llm", "router.report", "router.config_mapper", "router.factor_search")
        or m.startswith(("router.llm.", "router.report.", "router.config_mapper.", "router.factor_search."))}
    assert not bad, f"{rel} imports {sorted(bad)}"


@pytest.mark.parametrize("rel", LLM_SIDE)
def test_llm_modules_do_not_import_the_decision_code(rel):
    bad = {m for m in imports(ROOT / rel) if m in ("router.decide", "router.router", "router.replay",
                                                   "router.stage0", "router.stage2")
           or m.startswith(("router.branches", "router.decide.", "router.router."))}
    assert not bad, f"{rel} imports {sorted(bad)}"
