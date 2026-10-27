# vla-isaac

Simulation side of the repair ladder: an Isaac Lab Arena `RolloutBackend` for vla-router, traces,
reference controller, experiment tools, data pipeline and the GPU runbook. See `docs/IMPL_PLAN.md`
and `docs/BUILD_STATUS.md` at the repo root.

## Setup and tests (GPU-free)

The test environment is the conda env `vla-sim` (a clone of `vla`, so hybrid-agent's own env is
untouched), with CPU-only torch:

```bash
conda create --name vla-sim --clone vla
conda activate vla-sim
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
pip install -e ../vla-router -e ".[test]"        # hagent is already in the cloned env
unset PYTHONPATH                                  # ROS pytest plugins break collection
pytest
```

Nothing here needs Isaac Sim, Arena's Python env, GR00T or a GPU. Modules that only run inside
Isaac's Python on the GPU box live in `vla_isaac/arena_ext/` and are never imported by the tests.
