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
