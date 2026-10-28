# PLAN F0
Confirmed: vla-router/pyproject.toml had no dependencies/build-system; `pytest` failed at collection (`No module named 'hagent'`). The scipy test already used `pytest.importorskip` (no change needed).
1. pyproject: build-system, `dependencies=["hybrid-agent"]`, `dev`/`scipy` extras, packages.find limited to ladder*/router*.
2. vla-router/README.md: exact setup commands.
3. tests/test_packaging.py pins the declaration and the README.
4. Verify in a fresh venv; update DECISIONS.md and ISSUES.md.
