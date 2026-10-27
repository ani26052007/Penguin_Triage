"""Hard rule: every ``# VERIFY(<id>)`` marker has a row in docs/VERIFY_ON_GPU.md."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MARK = re.compile(r"VERIFY\((V-[a-z0-9-]+)\)")


def test_every_verify_marker_is_listed():
    listed = set(re.findall(r"^\| (V-[a-z0-9-]+) \|", (ROOT / "docs" / "VERIFY_ON_GPU.md").read_text(), re.M))
    used = {}
    for repo in ("vla-isaac", "policy-wrapper"):
        for p in (ROOT / repo).rglob("*"):
            if p.is_file() and p.suffix in {".py", ".yaml", ".sh", ".md"} and "tests" not in p.parts:
                for m in MARK.findall(p.read_text(encoding="utf-8", errors="replace")):
                    used.setdefault(m, str(p.relative_to(ROOT)))
    assert used, "no markers found: the scan is broken"
    missing = {k: v for k, v in used.items() if k not in listed}
    assert not missing, f"VERIFY markers without a row in docs/VERIFY_ON_GPU.md: {missing}"
