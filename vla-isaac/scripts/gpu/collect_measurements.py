#!/usr/bin/env python3
"""Turn every runbook measurement into one ``measurements/summary.json``.

Reads ``<measurements>/<RUN_TS>/<script>/result.json`` (written by the GPU scripts; a failed run has
``status: failed``). For each script the newest finished run is used, else the newest run. The summary
holds:

- ``verify``: every id of ``docs/VERIFY_ON_GPU.md``, either ``settled`` (with the value measured and the
  result file it came from) or ``open`` (with the script that would settle it, if any);
- ``decisions``: the run_order.md decisions (pilot stays or switches to DROID, Mimic usable, mode gap
  real, instruction string, merged dataset trains), each ``undecided`` until its inputs exist;
- ``measurements``: GPU, versions, VRAM;
- ``runs``: every result file found.

Intervals and tests come from vla-router's ``ladder.stats`` (run it in the vla-isaac venv).
Usage: ``collect_measurements.py [--measurements DIR] [--verify docs/VERIFY_ON_GPU.md] [--out FILE]``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

ROOT = Path(__file__).resolve().parents[3]

# which script settles which VERIFY id: id -> (script, function(result) -> value or None when not settled)
Settler = Callable[[Mapping[str, Any]], Any]


def _cell(res: Mapping, name: str) -> Mapping:
    return (res.get("cells") or {}).get(name) or {}


def _some(v: Any) -> Any:
    return v if v not in (None, [], {}) else None


SETTLES: dict[str, tuple[str, Settler]] = {
    "V-viz": ("R4_nominal_clean", lambda r: "accepted (R4 ran with the backend's app flags)"),
    "V-timestamped-dir": ("R4_nominal_clean", lambda r: _cell(r, "nominal").get("timestamped_subdir")),
    "V-jsonl-success": ("R4_nominal_clean", lambda r: _some(_cell(r, "nominal").get("success_types"))),
    "V-progress": ("R4_nominal_clean", lambda r: _some({k: v for k, v in (_cell(r, "nominal").get("progress_sample")
                                                                           or {}).items() if v is not None})),
    "V-var-record": ("R7_dark_cell", lambda r: _cell(r, "dark").get("variations_sample")),
    "V-list-var": ("R3_list_variations", lambda r: {"factor_table_equal": r.get("factor_table_equal"),
                                                    "regenerated": r.get("regenerated")}),
    "V-exp-yaml": ("R6_replay_agreement", lambda r: "experiment-runner YAML ran (R6)"),
    "V-exp-remote-policy": ("R6_replay_agreement", lambda r: "policy-wrapper remote policy ran in the experiment runner (R6)"),
    "V-runner-parity": ("runner_parity", lambda r: _some(r.get("comparison"))),
    "V-wrapper-parity": ("wrapper_parity", lambda r: _some(r.get("parity"))),
    "V-instruction": ("instruction_check", lambda r: _some(_instruction_rates(r))),
    "V-train-instr": ("R1b_download_datasets", lambda r: _some(r.get("tasks"))),
    "V-mimic-source": ("R1b_download_datasets", lambda r: r.get("mimic_source")),
    "V-hf-rev": ("R1_download_checkpoint", lambda r: r.get("revision_sha")),
    "V-ckpt-cfg": ("R1_download_checkpoint", lambda r: _ckpt_cfg(r.get("checkpoint") or {})),
    "V-stats-format": ("R1_download_checkpoint", lambda r: (r.get("checkpoint") or {}).get("statistics_shape")),
    "V-client-tests": ("check_client_tests", lambda r: _some({k: v.get("imports_without_isaac") for k, v in r.items()
                                                              if k in ("vla", "gr00t") and isinstance(v, Mapping)})),
    "V-mimic-override": ("mimic_override_check", lambda r: _mimic(r, "override", "result")),
    "V-mimic-seed": ("mimic_override_check", lambda r: _mimic(r, "seed", "mode")),
    "V-mimic-out": ("mimic_override_check", lambda r: _mimic(r, "stdout", "parse_ok")),
    "V-mimic-guarantee": ("mimic_override_check", lambda r: _some(((r.get("check") or {}).get("guarantee")))),
    "V-converter-noisaac": ("finetune_dryrun", lambda r: (r.get("convert") or {}).get("ok")),
    "V-finetune-log": ("finetune_dryrun", lambda r: _some(((r.get("dryrun") or {}).get("parameters") or {}).get("trainable")
                                                          and (r.get("dryrun") or {}).get("parameters"))),
}


def _ckpt_cfg(c: Mapping) -> Any:
    files = c.get("files") or {}
    if not files:
        return None
    return {"files": files, "tune_top_llm_layers": c.get("tune_top_llm_layers"),
            "num_inference_timesteps": c.get("num_inference_timesteps"), "action_horizon": c.get("action_horizon")}


def _mimic(r: Mapping, section: str, key: str) -> Any:
    v = ((r.get("check") or {}).get(section) or {}).get(key)
    return v


def _instruction_rates(r: Mapping) -> dict:
    """distinct string -> success proportion, from the instruction_<i> cells."""
    distinct = ((r.get("strings") or {}).get("distinct")) or []
    out = {}
    for i, s in enumerate(distinct):
        cell = _cell(r, f"instruction_{i}")
        if cell.get("success"):
            out[s] = cell["success"]
    return out


# --------------------------------------------------------------------------- #
# reading
# --------------------------------------------------------------------------- #

def verify_ids(verify_md: Path) -> list[str]:
    return re.findall(r"^\| (V-[a-z0-9-]+) \|", Path(verify_md).read_text(encoding="utf-8"), re.M)


def read_results(measurements: Path) -> list[dict]:
    runs = []
    for p in sorted(Path(measurements).glob("*/*/result.json")):
        try:
            res = json.loads(p.read_text(encoding="utf-8"))
        except ValueError as e:
            res = {"status": "unreadable", "error": str(e)}
        res.setdefault("script", p.parent.name)
        res.setdefault("run_ts", p.parent.parent.name)
        res["_path"] = str(p)
        runs.append(res)
    return runs


def pick(runs: list[dict]) -> dict[str, dict]:
    """Per script: the newest ``ok`` run, else the newest run."""
    by: dict[str, list[dict]] = {}
    for r in runs:
        by.setdefault(r["script"], []).append(r)
    out = {}
    for script, rs in by.items():
        rs = sorted(rs, key=lambda r: r["run_ts"])
        ok = [r for r in rs if r.get("status") == "ok"]
        out[script] = (ok or rs)[-1]
    return out


# --------------------------------------------------------------------------- #
# decisions
# --------------------------------------------------------------------------- #

def _rate(cell: Mapping) -> Optional[Mapping]:
    s = cell.get("success")
    return s if s and s.get("n") else None


def decide_pilot(chosen: Mapping[str, dict], gate: Mapping) -> dict:
    r4 = chosen.get("R4_nominal_clean")
    nominal = _rate(_cell(r4, "nominal")) if r4 and r4.get("status") == "ok" else None
    if nominal is None:
        return {"decision": "undecided", "reason": "R4 has not finished"}
    shifted = {}
    for script in ("R7_dark_cell", "R7b_hdr_and_colour_temp"):
        res = chosen.get(script)
        if res and res.get("status") == "ok":
            for name, cell in (res.get("cells") or {}).items():
                if _rate(cell):
                    shifted[name] = _rate(cell)
    out = {"nominal": nominal, "shifted": shifted, "nominal_min": gate["nominal_min"], "shifted_max": gate["shifted_max"]}
    if nominal["rate"] < gate["nominal_min"]:
        return {**out, "decision": "switch to DROID",
                "reason": f"R4 nominal {nominal['k']}/{nominal['n']} = {nominal['rate']:.2f} < {gate['nominal_min']}"}
    if not shifted:
        return {**out, "decision": "undecided", "reason": "R4 passes; R7 / R7b have not finished"}
    drops = {k: v for k, v in shifted.items() if v["rate"] <= gate["shifted_max"]}
    if not drops:
        return {**out, "decision": "switch to DROID",
                "reason": f"no dark / HDR / colour cell drops to {gate['shifted_max']} or below: the lighting family does not exist"}
    note = ("" if nominal["lower"] >= gate["nominal_min"]
            else f" (the R4 interval's lower bound {nominal['lower']:.2f} is below {gate['nominal_min']}: report it)")
    return {**out, "decision": "pilot stays", "failing_cells": sorted(drops),
            "reason": f"R4 nominal {nominal['rate']:.2f} >= {gate['nominal_min']} and {sorted(drops)} drop to "
                      f"<= {gate['shifted_max']}{note}"}


def decide_mimic(chosen: Mapping[str, dict]) -> dict:
    res = chosen.get("mimic_override_check")
    if not res or res.get("status") != "ok":
        return {"decision": "undecided", "reason": "mimic_override_check.sh has not finished"}
    check = res.get("check") or {}
    override = check.get("override") or {}
    took = override.get("took_effect")
    result = override.get("result")
    mode = (check.get("seed") or {}).get("mode")
    parse_ok = (check.get("stdout") or {}).get("parse_ok")
    facts = {"override_took_effect": took, "override_result": result, "seed_mode": mode, "stdout_parse_ok": parse_ok}
    if took is False:
        return {**facts, "decision": "not usable", "reason": "the lighting override did not reach the Mimic runs"}
    if result == "inconclusive":
        return {**facts, "decision": "inconclusive",
                "reason": f"the override check is inconclusive: {override.get('reason')}; check the override by hand "
                          f"(V-mimic-override) before using Mimic"}
    if mode and mode.startswith("stop"):
        return {**facts, "decision": "not usable", "reason": mode}
    if parse_ok is False:
        return {**facts, "decision": "not usable until fixed",
                "reason": "the Mimic stdout does not match reference.mimic.output.progress (V-mimic-out): fix the pattern"}
    if took and mode and parse_ok:
        return {**facts, "decision": f"usable (reference.mimic.seed.mode: {mode})", "reason": "all three checks pass"}
    return {**facts, "decision": "undecided", "reason": "a check could not be made (see the facts)"}


def decide_mode_gap(chosen: Mapping[str, dict], gate: Mapping) -> dict:
    r4, r5 = chosen.get("R4_nominal_clean"), chosen.get("R5_nominal_incident")
    c = _rate(_cell(r4, "nominal")) if r4 and r4.get("status") == "ok" else None
    i = _rate(_cell(r5, "nominal")) if r5 and r5.get("status") == "ok" else None
    if c is None or i is None:
        return {"decision": "undecided", "reason": "needs R4 and R5"}
    from ladder.stats import fisher_exact

    p = fisher_exact(c["k"], c["n"], i["k"], i["n"])
    real = p < gate["alpha"]
    return {"decision": "real" if real else "not detected", "clean": c, "incident": i, "gap": c["rate"] - i["rate"],
            "fisher_p": p, "alpha": gate["alpha"], "pairing": "unpaired (IMPL_PLAN D10)",
            "reason": f"two-sided Fisher exact p = {p:.3g} {'<' if real else '>='} {gate['alpha']}"}


def decide_instruction(chosen: Mapping[str, dict]) -> dict:
    res = chosen.get("instruction_check")
    if not res or res.get("status") != "ok":
        return {"decision": "undecided", "reason": "instruction_check.sh has not finished"}
    sources = (res.get("strings") or {}).get("sources") or {}
    rates = _instruction_rates(res)
    ev = sources.get("eval_yaml")
    train = sources.get("tasks_jsonl")
    if ev is None or train is None or ev not in rates or train not in rates:
        return {"decision": "undecided", "rates": rates, "reason": "the eval or the training string has no result"}
    if ev == train:
        return {"decision": f"keep eval.yaml: {ev!r}", "rates": rates, "reason": "eval.yaml already uses the training string"}
    if rates[train]["rate"] >= rates[ev]["rate"]:
        return {"decision": f"switch eval.yaml to the training string {train!r}", "rates": rates,
                "reason": f"training string {rates[train]['rate']:.2f} >= eval string {rates[ev]['rate']:.2f} (IMPL_PLAN Q2)"}
    return {"decision": f"keep eval.yaml: {ev!r}", "rates": rates,
            "reason": f"eval string {rates[ev]['rate']:.2f} > training string {rates[train]['rate']:.2f}"}


def decide_merged_dataset(chosen: Mapping[str, dict]) -> dict:
    res = chosen.get("finetune_dryrun")
    if not res:
        return {"decision": "undecided", "reason": "finetune_dryrun.sh has not run"}
    d = res.get("dryrun") or {}
    if res.get("status") != "ok":
        return {"decision": "does not train", "reason": f"finetune_dryrun.sh failed ({res.get('_path')})", "dryrun": d}
    trains = bool(d.get("trained")) and bool(d.get("stats_json_after"))
    return {"decision": "trains" if trains else "does not train", "dryrun": d,
            "reason": "training finished and computed meta/stats.json" if trains
            else "training did not finish, or meta/stats.json was not computed"}


# --------------------------------------------------------------------------- #
# summary
# --------------------------------------------------------------------------- #

def summarize(measurements: Path, verify_md: Path, gate: Mapping) -> dict:
    runs = read_results(measurements)
    chosen = pick(runs)
    ids = verify_ids(verify_md)
    verify: dict[str, dict] = {}
    for vid in ids:
        if vid not in SETTLES:
            verify[vid] = {"state": "open", "settled_by": None, "reason": "no runbook script measures it"}
            continue
        script, fn = SETTLES[vid]
        res = chosen.get(script)
        if res is None:
            verify[vid] = {"state": "open", "settled_by": script, "reason": f"{script} has not run"}
        elif res.get("status") != "ok":
            verify[vid] = {"state": "open", "settled_by": script, "reason": f"{script} failed", "result": res["_path"]}
        else:
            value = fn(res)
            verify[vid] = ({"state": "settled", "settled_by": script, "value": value, "result": res["_path"]}
                           if value is not None else
                           {"state": "open", "settled_by": script, "reason": f"{script} ran but did not measure it",
                            "result": res["_path"]})
    unknown = sorted(set(SETTLES) - set(ids))
    gpu = chosen.get("00_check_gpu")
    r0 = chosen.get("R0_setup")
    r8 = chosen.get("R8_vram")
    return {
        "measurements_dir": str(measurements),
        "verify": verify,
        "settled": sorted(k for k, v in verify.items() if v["state"] == "settled"),
        "open": sorted(k for k, v in verify.items() if v["state"] == "open"),
        "unknown_ids_in_collector": unknown,
        "decisions": {"pilot": decide_pilot(chosen, gate), "mimic": decide_mimic(chosen),
                      "mode_gap": decide_mode_gap(chosen, gate), "instruction": decide_instruction(chosen),
                      "merged_dataset": decide_merged_dataset(chosen)},
        "measurements": {"gpu": gpu.get("gpus") if gpu else None, "versions": r0.get("versions") if r0 else None,
                         "vram": r8.get("vram") if r8 else None},
        "runs": [{"script": r["script"], "run_ts": r["run_ts"], "status": r.get("status"), "result": r["_path"]}
                 for r in runs],
    }


def load_gate(runbook: Path) -> dict:
    import yaml

    return dict(yaml.safe_load(Path(runbook).read_text(encoding="utf-8"))["gate"])


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="collect_measurements.py", description=__doc__.split("\n")[0])
    ap.add_argument("--measurements", default=str(ROOT / "measurements"))
    ap.add_argument("--verify", default=str(ROOT / "docs" / "VERIFY_ON_GPU.md"))
    ap.add_argument("--runbook", default=str(ROOT / "vla-isaac" / "config" / "runbook_gr1_fridge.yaml"))
    ap.add_argument("--out", help="default: <measurements>/summary.json")
    a = ap.parse_args(argv)
    meas = Path(a.measurements)
    s = summarize(meas, Path(a.verify), load_gate(Path(a.runbook)))
    out = Path(a.out) if a.out else meas / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(s, indent=2, sort_keys=True), encoding="utf-8")
    print(f"settled {len(s['settled'])}: {', '.join(s['settled']) or '-'}")
    print(f"open {len(s['open'])}: {', '.join(s['open']) or '-'}")
    for name, d in s["decisions"].items():
        print(f"{name}: {d['decision']} ({d.get('reason', '')})")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
