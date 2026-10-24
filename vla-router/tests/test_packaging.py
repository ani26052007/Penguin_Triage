"""F0 (ISSUES Tier 7): the documented setup must declare hagent so `pytest` runs from a clean env."""
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_hagent_declared_as_dependency():
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text())
    deps = meta["project"].get("dependencies", [])
    assert any(d.split("@")[0].strip().lower().startswith("hybrid-agent") for d in deps)


def test_readme_documents_local_install():
    text = (ROOT / "README.md").read_text()
    assert "pip install -e" in text and "hybrid-agent" in text
