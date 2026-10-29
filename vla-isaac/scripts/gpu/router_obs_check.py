#!/usr/bin/env python3
"""Router observation-contract check on real instrument files.

From the 2026-10-29 GPU session; it ran only there, on the box. It was assembled outside IsaacBackend:
the router loop did not drive the run. Only ObservationContractCheck is used, with router code unchanged.

usage: router_obs_check.py NOMINAL_DIR CHECK_DIR [CHECK_DIR ...]

NOMINAL_DIR: instrument directory (env*_ep*.json, as written by policy_wrapper.instrument via
VLA_INSTRUMENT_DIR) of clean episodes; the nominal is built from it. Each CHECK_DIR is checked against that
nominal and printed as "<dir name>: <n> evidence item(s)". Use a held-out clean directory as one CHECK_DIR to
see the false-positive side. Needs vla-router and vla-isaac importable (the vla-isaac venv).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import median

CAMERA = "ego_view"


def _router():
    from router.prereq.trace import EpisodeTrace, InMemoryTraceSource
    from router.schemas import ActorKind, ActorRef, CodeRef, Incident, Mode, SceneSpec, TrainingSpec
    from router.stage1.nominal import NominalStats
    from router.stage1.obs_contract import ObservationContractCheck, metric, observation_metrics, opponent, required_vectors
    from vla_isaac.traces.instrument_reader import read_instrument_dir

    return locals()


def traces(r, directory):
    eps = r["read_instrument_dir"](directory)
    code_key = r["CodeRef"]("pw@injected", {"client": {}}).key()
    return [r["EpisodeTrace"](seed=i, mode=r["Mode"].CLEAN, frames=ep.frames, step_pairs=ep.step_pairs,
                              raw_actions=ep.raw_actions, code_key=code_key)
            for i, (_, ep) in enumerate(sorted(eps.items()))]


def obs_nominal(r, clean, cameras=(CAMERA,), rel_floor=0.05, abs_floor=0.02):   # copied from vla-isaac/tests/bugs_support.py
    metric, opponent = r["metric"], r["opponent"]
    vectors = {}
    for cam in cameras:
        frames = [fs.policy_input for t in clean for fs in t.frames if fs.camera == cam]
        vectors[metric(cam, "opp")] = tuple(median(c) for c in zip(*(opponent(f.channel_means) for f in frames)))
        vectors[metric(cam, "row_profile")] = tuple(median(c) for c in zip(*(f.row_profile for f in frames)))
        vectors[metric(cam, "col_profile")] = tuple(median(c) for c in zip(*(f.col_profile for f in frames)))
    assert set(vectors) == set(r["required_vectors"](cameras))
    samples = r["observation_metrics"](clean, cameras, vectors)
    floors = {k: max(rel_floor * abs(sum(v) / len(v)), abs_floor) for k, v in samples.items()}
    return r["NominalStats"].from_samples(samples, vectors=vectors, min_std=floors)


def check(r, nominal, tr):
    code = r["CodeRef"]("pw@injected", {"client": {}})
    spec = r["TrainingSpec"]("gr00t_n1d6_gr1_ranch_bottle_into_fridge", image_size=(512, 512), channel_order="RGB")
    refs = {"clean": "traces/clean"}
    src = r["InMemoryTraceSource"]({refs["clean"]: list(tr)})
    inc = r["Incident"]("real-gpu", r["SceneSpec"]("gr1_fridge_ranch"), spec, code,
                        r["ActorRef"](r["ActorKind"].POLICY, "ckpt"), (0,), r["Mode"].CLEAN, log_refs=refs)
    return r["ObservationContractCheck"](nominal, src, [CAMERA]).run(inc)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="ObservationContractCheck on instrument directories.")
    p.add_argument("nominal_dir", type=Path, help="clean instrument directory the nominal is built from")
    p.add_argument("check_dirs", type=Path, nargs="+", help="instrument directories to check against it")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    r = _router()
    nominal_traces = traces(r, args.nominal_dir)
    print(f"nominal: {len(nominal_traces)} episode(s) from {args.nominal_dir}")
    nom = obs_nominal(r, nominal_traces)
    for d in args.check_dirs:
        ev = check(r, nom, traces(r, d))
        print(f"{d.name}: {len(ev)} evidence item(s)")
        for e in ev:
            print("  ", e.id, f"{e.strength:.2f}", e.detail[:120])
    return 0


if __name__ == "__main__":
    sys.exit(main())
