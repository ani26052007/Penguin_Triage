"""Structure of the Step 7 modules (extends tests/test_no_model_in_routing.py to ``ladder/``).

* Rung 1 decides pass / fail with statistics only: no LLM layer, no hagent, no Rung 2.
* Rung 2 is the one model judgement (plan 4, Rung 2). It sits on the LLM side: it must not import the
  decision code, so nothing it says can reach a route except through the pure gate in the loop's
  hands (``data_gap_allowed``).
* The loop wires them together but imports neither hagent nor the LLM layer itself; models arrive as
  injected callables.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.test_no_model_in_routing import imports

LADDER = Path(__file__).parents[1] / "ladder"
LLM_LAYER = ("router.llm", "router.report", "router.config_mapper", "router.factor_search")
DECISION_CODE = ("router.decide", "router.router", "router.replay", "router.stage0", "router.stage2",
                 "router.branches", "router.behavior")


def hits(path: Path, prefixes) -> set[str]:
    return {m for m in imports(path) if any(m == p or m.startswith(p + ".") for p in prefixes)}


@pytest.mark.parametrize("name", ["rung1.py", "stats.py", "backend.py"])
def test_statistical_modules_have_no_model_and_no_repair_agent(name):
    bad = hits(LADDER / name, (*LLM_LAYER, "hagent", "ladder.rung2", "ladder.repair", "ladder.loop"))
    assert not bad, f"{name} imports {sorted(bad)}"


def test_rung2_does_not_import_the_decision_code_or_the_statistics_that_decide():
    bad = hits(LADDER / "rung2.py", (*DECISION_CODE, "ladder.rung1", "ladder.loop", "ladder.stats"))
    assert not bad, f"rung2.py imports {sorted(bad)}"


def test_loop_imports_neither_hagent_nor_the_llm_layer():
    bad = hits(LADDER / "loop.py", ("hagent", *LLM_LAYER))
    assert not bad, f"loop.py imports {sorted(bad)}"


def test_the_decision_modules_do_not_import_the_ladder_loop_pieces():
    root = Path(__file__).parents[1] / "router"
    for p in ("decide.py", "router.py", "stage2.py", "replay.py"):
        assert not hits(root / p, ("ladder.rung0", "ladder.rung1", "ladder.rung2", "ladder.loop", "ladder.repair")), p
