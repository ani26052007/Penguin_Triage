#!/usr/bin/env python3
"""pass@k from n attempts with c successes (unbiased estimator of Chen et al. 2021).

    pass_at_k(n, c, k) = 1 - C(n-c, k) / C(n, k)

It is the chance that at least one of k attempts drawn without replacement from the n is a success. As a
repair rate it assumes a selector that picks the working attempt among the k; no selector (Rung 1) was run.

usage: passk.py N C [K ...]        (default K = 1 3)
"""
from __future__ import annotations

import argparse
import sys
from math import comb


def pass_at_k(n: int, c: int, k: int) -> float:
    if not (0 <= c <= n) or not (1 <= k <= n):
        raise ValueError(f"need 0 <= c <= n and 1 <= k <= n, got n={n} c={c} k={k}")
    return 1.0 - comb(n - c, k) / comb(n, k)


def wilson(c: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for c successes in n trials."""
    if n <= 0 or not (0 <= c <= n):
        raise ValueError(f"need n > 0 and 0 <= c <= n, got n={n} c={c}")
    p = c / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return centre - half, centre + half


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("n", type=int)
    ap.add_argument("c", type=int)
    ap.add_argument("k", type=int, nargs="*", default=[1, 3])
    a = ap.parse_args(argv)
    lo, hi = wilson(a.c, a.n)
    print(f"n={a.n} c={a.c} per-attempt {a.c / a.n:.2f} (Wilson 95% {lo:.2f}-{hi:.2f})")
    for k in a.k:
        print(f"pass@{k} = {pass_at_k(a.n, a.c, k):.2f}  (assumes a selector)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
