# vla-router

Router + verification ladder. Depends on the sibling package `hybrid-agent` (hagent).

## Setup and tests

From a clean checkout (Python 3.11+):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e "../hybrid-agent[dev]" -e ".[dev]"   # one command so pip resolves hagent from the local path
cd ../hybrid-agent && pytest && cd ../vla-router && pytest
```

If ROS (or anything else) is sourced in your shell and sets `PYTHONPATH`, run `unset PYTHONPATH` first: its pytest plugins load at collection and fail (e.g. `No module named 'yaml'`).

Optional: `pip install -e ".[scipy]"` enables the scipy cross-check in `tests/test_stats.py`;
without scipy that test is skipped.

## Rung 0 in the loop

Rung 0 (diff sanity, lint, tests, smoke run, optional reviewer) is off by default. Switch it on with
`LoopConfig.rung0`; the loop then builds the hook from the repairer's `workspaces` and `settings`
(`HagentRepairer` has both) and runs it on the rebuilt patched code before any simulator episode:

```python
# rung0-example
from ladder.loop import IncidentLoop, LoopConfig
from ladder.rung0 import Rung0Config

cfg = LoopConfig(router_cfg, rung1_cfg,
                 rung0=Rung0Config(lint_cmd="ruff check .", test_cmd="pytest -q", expected_files=("controller/wrapper.py",)))
loop = IncidentLoop(backend, cfg, repairer=repairer, inputs=inputs, data_ladder=data_ladder,
                    rung0_reviewer=reviewer_model)      # the reviewer is optional; hagent's own approving review is not repeated
```

Pass `rung0=<hook>` instead for a custom hook (not both).
