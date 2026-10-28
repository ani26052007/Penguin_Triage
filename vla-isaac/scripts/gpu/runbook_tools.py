#!/usr/bin/env python3
"""Helpers the GPU runbook scripts call (``python runbook_tools.py <command> ...``).

Every command is a small, pure-as-possible step: build an Arena argv from the configs, summarise a
run directory, compare two runs, probe a Mimic HDF5, write a result file. The scripts own the
subprocesses; this file owns the parsing and the numbers, so both are tested without a GPU
(``vla-isaac/tests/test_gpu_runbook.py``). vla_isaac / vla-router imports are inside the commands:
``ping``-free commands that only need the standard library run with any Python.

Nothing here downloads, prints or writes a secret.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

ISAAC_MODULES = ("isaacsim", "isaaclab", "isaaclab_tasks", "isaaclab_mimic", "isaaclab_assets", "isaaclab_rl",
                 "omni", "pxr", "carb")


class RunbookError(RuntimeError):
    """A runbook step cannot go on; the message says why."""


# --------------------------------------------------------------------------- #
# configs
# --------------------------------------------------------------------------- #

RUNBOOK_KEYS = {"task_config", "grid", "pipeline", "factor_names", "checkpoint", "dataset", "stock_policy", "cell_timeout_s", "nominal",
                "incident", "dark_cell", "probes", "probe_episodes", "instruction_check", "parity", "mimic_check",
                "finetune_dryrun", "gate", "vram"}


def _yaml(path: Path) -> Any:
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_runbook(path: str | Path) -> dict:
    """The runbook YAML, strict, with its config paths made absolute."""
    path = Path(path).resolve()
    d = _yaml(path)
    if not isinstance(d, dict):
        raise RunbookError(f"{path}: not a mapping")
    unknown, missing = set(d) - RUNBOOK_KEYS, RUNBOOK_KEYS - set(d)
    if unknown or missing:
        raise RunbookError(f"{path}: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}")
    for key in ("task_config", "grid", "pipeline", "factor_names"):
        p = Path(d[key])
        d[key] = str(p if p.is_absolute() else (path.parent / p).resolve())
    return d


def dotted(d: Mapping, key: str) -> Any:
    cur: Any = d
    for part in key.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            raise RunbookError(f"no key {key!r} in the runbook config")
        cur = cur[part]
    return cur


def _condition(pairs: Iterable[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in pairs:
        if "=" not in p:
            raise RunbookError(f"--factor wants name=value, got {p!r}")
        k, v = p.split("=", 1)
        if k in out:
            raise RunbookError(f"factor {k!r} given twice")
        out[k] = v
    return out


def variations_for(task, condition: Mapping[str, str]) -> dict[str, Any]:
    """The Arena override assignments of ``condition`` through the task's factor table."""
    from vla_isaac.data.rung_b_mimic import scene_for
    from vla_isaac.scene.factors import FactorTable

    if not condition:
        return {}
    return FactorTable.load(task.factor_table).assignments(scene_for(task.nominal_scene, condition))


# --------------------------------------------------------------------------- #
# Arena command lines
# --------------------------------------------------------------------------- #

def stock_run_spec(rb: Mapping, task, *, run_name: str, seed: int, num_envs: int, episodes: int, host: str, port: int,
                   device: str, instruction: Optional[str], condition: Mapping[str, str]):
    from vla_isaac.runner.arena_cmd import PolicySpec, RunSpec

    sp = rb["stock_policy"]
    return RunSpec(run_name=run_name, environment=dict(task.environment), seed=seed, num_envs=num_envs,
                   num_episodes=episodes, language_instruction=instruction,
                   policy=PolicySpec(sp["type"], sp["config_yaml"], device, host, port),
                   variations=variations_for(task, condition))


def policy_runner_cmd(rb: Mapping, backend: str, *, python: str, out: str, episodes: int, num_envs: int, seed: int,
                      host: str, port: int, instruction: Optional[str] = None,
                      condition: Mapping[str, str] = {}) -> list[str]:
    """``policy_runner.py`` with Arena's stock GR00T client (docs/PILOT_AND_ORDER.md R4-R7)."""
    from vla_isaac.config import load_backend_config, load_task_config
    from vla_isaac.runner.arena_cmd import policy_runner_argv

    cfg = load_backend_config(backend)
    task = load_task_config(rb["task_config"])
    spec = stock_run_spec(rb, task, run_name="runbook", seed=seed, num_envs=num_envs, episodes=episodes, host=host,
                          port=port, device=cfg.policy_device, instruction=instruction, condition=condition)
    return policy_runner_argv(python, spec, Path(out), app_flags=cfg.app_flags)


def experiment_runner_cmd(rb: Mapping, backend: str, *, python: str, out: str, episodes: int, seed: int, host: str,
                          port: int) -> list[str]:
    """The experiment runner on a one-run YAML with the same stock policy and seed (V-runner-parity)."""
    from vla_isaac.config import load_backend_config, load_task_config
    from vla_isaac.runner.arena_cmd import dump_experiment_yaml, experiment_argv

    cfg = load_backend_config(backend)
    task = load_task_config(rb["task_config"])
    spec = stock_run_spec(rb, task, run_name="parity", seed=seed, num_envs=1, episodes=episodes, host=host, port=port,
                          device=cfg.policy_device, instruction=None, condition={})
    out_dir = Path(out)
    yaml_path = dump_experiment_yaml(spec, out_dir / "experiment.yaml")
    return experiment_argv(python, yaml_path, out_dir / "results", environment=task.environment,
                           app_flags=cfg.app_flags)


def env_args(rb: Mapping) -> list[str]:
    """The env sub-command of ``policy_runner.py`` for the task (R3)."""
    from vla_isaac.config import load_task_config

    env = dict(load_task_config(rb["task_config"]).environment)
    out = [str(env.pop("type"))]
    env.pop("enable_cameras", None)
    for k in sorted(env):
        out += [f"--{k}", str(env[k])]
    return out


def mimic_cmd(rb: Mapping, backend: str, *, trials: int, output_file: str, condition: Mapping[str, str],
              seed_flag: Optional[str] = None, seed: Optional[int] = None) -> list[str]:
    """``generate_dataset.py`` as the RAW reference runs it, with an optional datagen seed flag."""
    from dataclasses import replace

    from vla_isaac.config import load_backend_config, load_task_config
    from vla_isaac.reference.mimic import MimicReference, mimic_argv

    cfg = load_backend_config(backend)
    task = load_task_config(rb["task_config"])
    ref = MimicReference(cfg, task)
    m = task.reference.mimic
    if seed_flag:
        m = replace(m, seed=replace(m.seed, flag=seed_flag))
    return mimic_argv(cfg.arena_python, cfg.arena_root, m, num_trials=trials, input_file=ref.source_dataset(),
                      output_file=Path(output_file), variations=variations_for(task, condition),
                      app_flags=cfg.app_flags, run_seed=seed if seed_flag else None)


# --------------------------------------------------------------------------- #
# run summaries
# --------------------------------------------------------------------------- #

def _records(run_dir: Path):
    from vla_isaac.results.episode_jsonl import read_records

    files = sorted(Path(run_dir).rglob("episode_results_*.jsonl"))
    recs = [r for f in files for r in read_records(f)]
    return files, recs


def summarize_run(run_dir: str | Path, *, episodes: int, num_envs: int, exit_code: int, log: Optional[str] = None,
                  confidence: float = 0.95, extra: Optional[Mapping[str, Any]] = None) -> dict:
    """k/n with the Wilson interval (``ladder.stats`` through ``experiments.tables``) and the facts the
    VERIFY rows ask about: where the JSONL landed, the type of ``success``, a ``variations`` and a
    ``progress`` sample."""
    from vla_isaac.experiments.tables import proportion

    run_dir = Path(run_dir)
    log_text = Path(log).read_text(encoding="utf-8", errors="replace") if log and Path(log).is_file() else ""
    out: dict[str, Any] = {"run_dir": str(run_dir), "exit_code": exit_code, "episodes_requested": episodes,
                           "num_envs": num_envs, "log": log, **dict(extra or {})}
    out["unrecognized_arguments"] = re.findall(r"unrecognized arguments: (.*)", log_text)
    try:
        files, recs = _records(run_dir)
    except Exception as e:                         # a strict-reader refusal is a finding, not a crash
        out.update(files=[], error=f"{type(e).__name__}: {e}")
        return out
    ranked = sorted(recs, key=lambda r: (r.episode_in_env, r.env_id))[:episodes]
    k = sum(r.success for r in ranked)
    out.update(
        files=[str(f) for f in files],
        timestamped_subdir=any(f.parent != run_dir for f in files) if files else None,
        records=len(recs), complete=len(ranked) == episodes,
        success=proportion(k, len(ranked), confidence),
        outcomes={f"{r.env_id}:{r.episode_in_env}": r.success for r in ranked},
        success_types=sorted({type(r.raw["success"]).__name__ for r in recs}),
        variations_sample=next((r.raw["variations"] for r in recs if "variations" in r.raw), None),
        progress_sample={label: next((r.raw.get("progress") for r in recs if r.success is want), None)
                         for label, want in (("success", True), ("failure", False))},
        language_instructions=sorted({str(r.raw.get("language_instruction")) for r in recs}),
    )
    return out


def compare_runs(a: Mapping, b: Mapping, *, alpha: float = 0.05) -> dict:
    """Two summaries of the same seeds: per-episode agreement and whether the rates differ."""
    from ladder.stats import agreement_rate, fisher_exact

    oa, ob = a.get("outcomes") or {}, b.get("outcomes") or {}
    shared = sorted(set(oa) & set(ob))
    if not shared:
        return {"comparable": False, "reason": "no episode (env_id:episode_in_env) in both runs"}
    ia = {i: oa[key] for i, key in enumerate(shared)}
    ib = {i: ob[key] for i, key in enumerate(shared)}
    sa, sb = a["success"], b["success"]
    p = fisher_exact(sa["k"], sa["n"], sb["k"], sb["n"])
    return {"comparable": True, "shared_episodes": len(shared), "agreement": agreement_rate(ia, ib),
            "both_success": sum(ia[i] and ib[i] for i in ia), "both_failure": sum(not ia[i] and not ib[i] for i in ia),
            "only_first": sum(ia[i] and not ib[i] for i in ia), "only_second": sum(ib[i] and not ia[i] for i in ia),
            "first": sa, "second": sb, "fisher_p": p, "differ": p < alpha,
            "intervals_overlap": max(sa["lower"], sb["lower"]) <= min(sa["upper"], sb["upper"])}


# --------------------------------------------------------------------------- #
# instruction strings
# --------------------------------------------------------------------------- #

def instruction_strings(rb: Mapping, backend: str) -> dict:
    """The eval.yaml string, the original dataset's ``meta/tasks.jsonl`` strings and the converter YAML's."""
    from vla_isaac.config import get_dotted, load_backend_config, load_task_config
    from vla_isaac.data.config import PipelineConfig
    from vla_isaac.experiments.grid import GridSpec

    cfg = load_backend_config(backend)
    task = load_task_config(rb["task_config"])
    grid = GridSpec.load(rb["grid"])
    base = cfg.workspace_bases[grid.code_base]
    eval_yaml = Path(base) / task.policy.config_path
    found, eval_str = get_dotted(_yaml(eval_yaml), task.eval_keys.language_instruction)
    if not found:
        raise RunbookError(f"{eval_yaml} has no {task.eval_keys.language_instruction}")
    ds_id = task.training_spec.dataset
    ds = cfg.datasets.get(ds_id)
    tasks_path = Path(ds) / "meta" / "tasks.jsonl" if ds else None
    if tasks_path is None or not tasks_path.is_file():
        raise RunbookError(f"no meta/tasks.jsonl for dataset {ds_id!r} (backend datasets map: {ds}); run "
                           f"R1b_download_datasets.sh first")
    tasks = [json.loads(line)["task"] for line in tasks_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    pipe = PipelineConfig.load(rb["pipeline"])
    conv = _yaml(pipe.rung_b.converter.template)["language_instruction"]
    sources = {"eval_yaml": str(eval_str), "converter_yaml": str(conv)}
    for i, t in enumerate(tasks):
        sources["tasks_jsonl" if len(tasks) == 1 else f"tasks_jsonl[{i}]"] = str(t)
    names = list(sources)
    differ = [f"{x}!={y}" for i, x in enumerate(names) for y in names[i + 1:] if sources[x] != sources[y]]
    distinct: list[str] = []
    for s in sources.values():
        if s not in distinct:
            distinct.append(s)
    return {"sources": sources, "paths": {"eval_yaml": str(eval_yaml), "tasks_jsonl": str(tasks_path),
                                          "converter_yaml": str(pipe.rung_b.converter.template)},
            "differ": differ, "distinct": distinct}


def instruction_line(info: Mapping) -> str:
    parts = [f"{k}={v!r}" for k, v in info["sources"].items()]
    return " | ".join(parts) + " | differ: " + (", ".join(info["differ"]) or "none")


# --------------------------------------------------------------------------- #
# VRAM, client tests, checkpoint
# --------------------------------------------------------------------------- #

def vram_summary(csv_path: str | Path) -> dict:
    """``nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total --format=csv,noheader,nounits`` lines."""
    per_gpu: dict[str, list[float]] = {}
    total: dict[str, float] = {}
    for line in Path(csv_path).read_text(encoding="utf-8").splitlines():
        cols = [c.strip() for c in line.split(",")]
        if len(cols) < 4:
            continue
        try:
            used, tot = float(cols[2]), float(cols[3])
        except ValueError:
            continue
        per_gpu.setdefault(cols[1], []).append(used)
        total[cols[1]] = tot
    if not per_gpu:
        return {"samples": 0}
    return {"samples": sum(len(v) for v in per_gpu.values()),
            "gpus": {g: {"peak_used_mib": max(v), "mean_used_mib": round(sum(v) / len(v), 1), "total_mib": total[g],
                         "samples": len(v)} for g, v in sorted(per_gpu.items())},
            "peak_used_mib": max(max(v) for v in per_gpu.values())}


def client_tests_report(log: str, exit_code: int) -> dict:
    """pytest's verdict on Arena's GR00T client tests, and whether an Isaac module was what failed."""
    counts = {k: int(n) for n, k in re.findall(r"(\d+) (passed|failed|errors?|skipped)", log)}
    missing = re.findall(r"No module named '([\w.]+)'", log)
    top = sorted({m.split(".")[0] for m in missing})
    isaac = sorted(m for m in top if m in ISAAC_MODULES)
    return {"exit_code": exit_code, "counts": counts, "missing_modules": top, "isaac_modules_missing": isaac,
            "imports_without_isaac": exit_code == 0 or (bool(counts.get("passed")) and not isaac and not missing),
            "blocked_by_isaac": bool(isaac)}


def checkpoint_report(ckpt: str | Path) -> dict:
    """The files and fields V-ckpt-cfg and V-stats-format ask about."""
    ckpt = Path(ckpt)
    files = {n: (ckpt / n).is_file() for n in ("config.json", "processor_config.json", "statistics.json",
                                                "embodiment_id.json")}
    out: dict[str, Any] = {"dir": str(ckpt), "files": files}
    if files["config.json"]:
        c = json.loads((ckpt / "config.json").read_text(encoding="utf-8"))
        flat = json.dumps(c)
        for key in ("tune_top_llm_layers", "num_inference_timesteps", "action_horizon"):
            m = re.search(rf'"{key}":\s*([^,}}]+)', flat)
            out[key] = json.loads(m[1]) if m else None
    if files["statistics.json"]:
        d = json.loads((ckpt / "statistics.json").read_text(encoding="utf-8"))
        out["statistics_shape"] = {e: {m: {g: {s: len(v) if isinstance(v, list) else None for s, v in gs.items()}
                                           for g, gs in ms.items() if isinstance(gs, dict)}
                                       for m, ms in es.items() if isinstance(ms, dict)}
                                   for e, es in d.items() if isinstance(es, dict)}
    return out


def pick_lerobot(files: Sequence[str], hint: str) -> str:
    """The one LeRobot directory (holding ``meta/tasks.jsonl``) whose path contains ``hint``."""
    roots = sorted({f[: -len("meta/tasks.jsonl")].rstrip("/") for f in files if f.endswith("meta/tasks.jsonl")})
    hits = [r for r in roots if hint in r]
    if len(hits) != 1:
        raise RunbookError(f"expected one LeRobot directory with {hint!r} among {roots or 'none'}, found {hits or 'none'}; "
                           f"set LEROBOT_PREFIX to the right one")
    return hits[0]


# --------------------------------------------------------------------------- #
# Mimic check
# --------------------------------------------------------------------------- #

def h5_probe(hdf5: str | Path, camera: str, *, frame_stride: int = 10) -> dict:
    """Per-file facts for the Mimic check: demo count, mean camera pixel (frames from
    ``data/demo_<i>/camera_obs/<camera>``), a content digest of every dataset (attributes left out,
    so a timestamp attribute does not make two equal runs differ), and any key or attribute that
    names a variation."""
    import h5py
    import numpy as np

    out: dict[str, Any] = {"path": str(hdf5)}
    digest = hashlib.sha256()
    found: list[str] = []
    means: list[float] = []
    with h5py.File(hdf5, "r") as f:
        names: list[str] = []
        f.visit(names.append)
        for name in sorted(names):
            obj = f[name]
            if "variation" in name.lower():
                found.append(name)
            for a in obj.attrs:
                if "variation" in str(a).lower():
                    found.append(f"{name}@{a}={obj.attrs[a]!r}")
            if isinstance(obj, h5py.Dataset):
                digest.update(name.encode())
                digest.update(np.ascontiguousarray(obj[()]).tobytes())
        for a in f.attrs:
            if "variation" in str(a).lower():
                found.append(f"/@{a}={f.attrs[a]!r}")
        demos = sorted(k for k in f.get("data", {}) if k.startswith("demo_"))
        for d in demos:
            key = f"data/{d}/camera_obs/{camera}"
            if key in f:
                frames = f[key][::frame_stride]
                if len(frames):
                    means.append(float(np.asarray(frames, dtype=np.float64).mean()))
    out.update(demos=len(demos), camera_frames_found=bool(means), mean_pixel=sum(means) / len(means) if means else None,
               digest=digest.hexdigest(), recorded_variations=found)
    return out


def mimic_log_report(log: str, rb: Mapping, trials: int) -> dict:
    """V-mimic-out and V-mimic-guarantee from one Mimic log, through the reference's own parser."""
    from vla_isaac.config import load_task_config
    from vla_isaac.reference.mimic import trial_outcomes

    out_cfg = load_task_config(rb["task_config"]).reference.mimic.output
    lines = [line for line in log.splitlines() if "successful demos" in line or "demos generated" in line]
    rep: dict[str, Any] = {"progress_lines": len(lines), "sample_lines": lines[:3] + (lines[-2:] if len(lines) > 3 else []),
                           "pattern": out_cfg.progress, "trials_asked": trials}
    try:
        outcomes, (s, a) = trial_outcomes(log, out_cfg)
    except Exception as e:
        rep.update(parse_ok=False, error=f"{type(e).__name__}: {e}")
        return rep
    rep.update(parse_ok=len(outcomes) >= trials and len(lines) > 0, outcomes=outcomes, successes=s, attempts=a,
               per_attempt_lines=len(lines) >= a > 0)
    return rep


def mimic_check_result(out: str | Path, rb: Mapping, *, flag: Optional[str], flag_in_help: bool) -> dict:
    """Read the runs of ``mimic_override_check.sh`` (``<out>/<run>/{probe.json,mimic.log}``) and decide."""
    out = Path(out)
    mc = rb["mimic_check"]

    def run(name: str) -> Optional[dict]:
        d = out / name
        if not (d / "probe.json").is_file():
            return None
        probe = json.loads((d / "probe.json").read_text(encoding="utf-8"))
        trials = int((d / "trials").read_text().strip())
        log = (d / "mimic.log").read_text(encoding="utf-8", errors="replace") if (d / "mimic.log").is_file() else ""
        return {"probe": probe, "log": mimic_log_report(log, rb, trials), "exit_code": int((d / "exit_code").read_text())}

    runs = {n: run(n) for n in ("override", "seed_a", "seed_a_repeat", "seed_b")}
    res: dict[str, Any] = {"runs": runs}
    ov, nom = runs["override"], runs["seed_a"]
    verdict: dict[str, Any] = {"condition": mc["override"], "darker_ratio": mc["darker_ratio"]}
    if ov and nom and ov["probe"]["mean_pixel"] is not None and nom["probe"]["mean_pixel"]:
        ratio = ov["probe"]["mean_pixel"] / nom["probe"]["mean_pixel"]
        verdict.update(method="camera brightness vs the nominal seed_a run", ratio=ratio,
                       took_effect=ratio < mc["darker_ratio"])
    else:
        verdict.update(method="none", took_effect=None, reason="no camera frames in one of the runs")
    verdict["recorded_variations"] = ov["probe"]["recorded_variations"] if ov else None
    res["override"] = verdict

    def same(x: Optional[dict], y: Optional[dict]) -> Optional[bool]:
        if not x or not y:
            return None
        return x["probe"]["digest"] == y["probe"]["digest"] and x["log"].get("outcomes") == y["log"].get("outcomes")

    same_seed = same(runs["seed_a"], runs["seed_a_repeat"])
    diff_seed = same(runs["seed_a"], runs["seed_b"]) if flag_in_help else None
    if same_seed is False:
        mode = "stop: the same seed gives different runs, so fixed mode is not deterministic either"
    elif diff_seed is False:
        mode = "per_run"
    elif same_seed and (diff_seed or not flag_in_help):
        mode = "fixed"
    else:
        mode = None
    res["seed"] = {"flag": flag, "flag_in_help": flag_in_help, "same_seed_identical": same_seed,
                   "different_seed_identical": diff_seed, "mode": mode}
    logs = [r["log"] for r in runs.values() if r]
    res["stdout"] = {"parse_ok": bool(logs) and all(x.get("parse_ok") for x in logs),
                     "per_attempt_lines": bool(logs) and all(x.get("per_attempt_lines") for x in logs),
                     "sample_lines": logs[0]["sample_lines"] if logs else []}
    res["guarantee"] = [{"trials_asked": x["trials_asked"], "attempts": x.get("attempts"),
                         "successes": x.get("successes")} for x in logs]
    return res


# --------------------------------------------------------------------------- #
# data / fine-tune dry run
# --------------------------------------------------------------------------- #

def pipeline_variant(pipeline: str | Path, out: str | Path, *, max_steps: int, timeout_s: float) -> Path:
    """A copy of the pipeline YAML for a short run: every hardware entry's steps capped, the timeout
    shortened, and the converter template made absolute (the copy lives elsewhere)."""
    import yaml

    from vla_isaac.data.config import PipelineConfig

    src = Path(pipeline).resolve()
    d = _yaml(src)
    tpl = Path(d["rung_b"]["converter"]["template"])
    d["rung_b"]["converter"]["template"] = str(tpl if tpl.is_absolute() else (src.parent / tpl).resolve())
    for hw in d["finetune"]["hardware"].values():
        hw["max_steps"] = int(max_steps)
        hw["save_steps"] = int(max_steps)
        hw["save_total_limit"] = 1
    d["finetune"]["timeout_s"] = float(timeout_s)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"# Dry-run copy of {src} (max_steps {max_steps}); written by runbook_tools.py\n"
                   + yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    PipelineConfig.load(out)                      # still a valid pipeline config
    return out


_BLOCKER = """import importlib.abc, runpy, sys
class _NoIsaac(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in {mods!r}:
            raise ModuleNotFoundError('blocked Isaac module ' + name + ' (V-converter-noisaac)')
        return None
sys.meta_path.insert(0, _NoIsaac())
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name='__main__')
"""


def isaac_blocked_argv(argv: Sequence[str]) -> list[str]:
    """``argv`` (python, script, args...) run with every Isaac module import refused."""
    return [argv[0], "-c", _BLOCKER.format(mods=ISAAC_MODULES), *argv[1:]]


def convert_dryrun(rb: Mapping, backend: str, pipeline: str, hdf5: str, work: str, *, instruction: Optional[str]) -> dict:
    """Arena's converter on the Mimic check's HDF5, with Isaac imports blocked (V-converter-noisaac).
    ``instruction`` replaces the converter YAML's string, so the result mixes with the nominal set."""
    import yaml

    from vla_isaac.config import load_backend_config
    from vla_isaac.data.config import PipelineConfig
    from vla_isaac.data.rung_b_mimic import plan_conversion
    from vla_isaac.runner.subprocess_runner import RunInvocation, SubprocessRunner

    cfg = load_backend_config(backend)
    inv = plan_conversion(cfg, PipelineConfig.load(pipeline).rung_b, Path(hdf5), Path(work))
    yaml_path = Path(work) / "converter.yaml"
    if instruction is not None:
        d = _yaml(yaml_path)
        d["language_instruction"] = instruction
        yaml_path.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    blocked = RunInvocation(argv=tuple(isaac_blocked_argv(inv.argv)), cwd=inv.cwd, env=dict(inv.env),
                            timeout_s=inv.timeout_s, log_path=inv.log_path, output_dir=inv.output_dir, spec=inv.spec)
    rep = {"argv": list(blocked.argv[:2]) + ["<isaac import blocker>"] + list(blocked.argv[3:]),
           "output_dir": str(inv.output_dir), "log": str(inv.log_path), "instruction": instruction}
    try:
        SubprocessRunner().run(blocked)
    except Exception as e:
        log = Path(inv.log_path).read_text(encoding="utf-8", errors="replace") if Path(inv.log_path).is_file() else ""
        rep.update(ok=False, error=f"{type(e).__name__}: {e}", blocked_isaac_import="blocked Isaac module" in log)
        return rep
    rep.update(ok=True, blocked_isaac_import=False)
    return rep


def nominal_task(backend: str, pipeline: str) -> str:
    """The nominal LeRobot set's single ``meta/tasks.jsonl`` string (the mix needs one shared string)."""
    from vla_isaac.config import load_backend_config
    from vla_isaac.data.config import PipelineConfig
    from vla_isaac.data.mixer import LeRobotSource

    ds_id = PipelineConfig.load(pipeline).mix.nominal_dataset
    path = load_backend_config(backend).datasets.get(ds_id)
    if path is None or not (Path(path) / "meta" / "tasks.jsonl").is_file():
        raise RunbookError(f"no meta/tasks.jsonl for the nominal dataset {ds_id!r} ({path}); run R1b_download_datasets.sh")
    tasks = LeRobotSource(ds_id, Path(path), {}).tasks()
    if len(tasks) != 1:
        raise RunbookError(f"the nominal dataset has {len(tasks)} task strings {tasks}; the dry run expects one")
    return next(iter(tasks.values()))


def mix_dryrun(rb: Mapping, backend: str, pipeline: str, shifted: str, out: str) -> dict:
    """The mixer on the nominal LeRobot set and one shifted set (the merged dataset the dry run trains on)."""
    from vla_isaac.config import load_backend_config
    from vla_isaac.data.config import PipelineConfig
    from vla_isaac.data.mixer import LeRobotSource, select_mix, write_mixed_dataset

    cfg = load_backend_config(backend)
    pipe = PipelineConfig.load(pipeline)
    nominal_dir = cfg.datasets.get(pipe.mix.nominal_dataset)
    if nominal_dir is None or not Path(nominal_dir).is_dir():
        raise RunbookError(f"nominal dataset {pipe.mix.nominal_dataset!r} is not local ({nominal_dir}); run "
                           f"R1b_download_datasets.sh")
    nom = LeRobotSource(pipe.mix.nominal_dataset, Path(nominal_dir), {})
    shf = LeRobotSource("dryrun_shifted", Path(shifted), dict(rb["finetune_dryrun"]["condition"]))
    sel = select_mix(nom.episodes(), shf.episodes(), pipe.mix)
    path = write_mixed_dataset(sel, [nom, shf], out)
    return {"dataset": str(path), "selection": sel.to_dict(), "stats_json_present": (path / "meta" / "stats.json").exists()}


def dryrun_report(dataset: str | Path, finetune_out: str | Path, exit_code: int, max_steps: int) -> dict:
    """Did training accept the merged dataset, and were its stats computed?"""
    dataset, ft = Path(dataset), Path(finetune_out)
    log_path = ft / "finetune.log"
    log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    manifest_path = ft / "vla_finetune_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    result = manifest.get("result") or {}
    steps = [int(m) for m in re.findall(r"(?:global_step|step)[\"']?\s*[:=]\s*(\d+)", log)]
    return {"exit_code": exit_code, "max_steps": max_steps, "stats_json_after": (dataset / "meta" / "stats.json").is_file(),
            "stats_generated_line": f"Generating stats for {dataset}" in log or "Generating stats for" in log,
            "max_step_seen": max(steps) if steps else None, "parameters": result.get("parameters"),
            "final_model_config": bool(result.get("final_model_config")), "complete": result.get("complete"),
            "missing": result.get("missing"), "tune_top_llm_layers": manifest.get("tune_top_llm_layers"),
            "trained": exit_code == 0 and bool(result.get("complete"))}


# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #

def now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_result(out: str | Path, script: str, *, run_ts: str, jsons: Mapping[str, str] = {},
                 sets: Mapping[str, str] = {}) -> Path:
    """``<out>/result.json``: status ok, every cell's ``summary.json``, plus named JSON files and values."""
    out = Path(out)
    res: dict[str, Any] = {"script": script, "run_ts": run_ts, "status": "ok", "finished": now(), "out": str(out)}
    cells = {d.name: json.loads((d / "summary.json").read_text(encoding="utf-8"))
             for d in sorted(out.iterdir()) if d.is_dir() and (d / "summary.json").is_file()}
    if cells:
        res["cells"] = cells
    for k, p in jsons.items():
        res[k] = json.loads(Path(p).read_text(encoding="utf-8"))
    for k, v in sets.items():
        try:
            res[k] = json.loads(v)
        except ValueError:
            res[k] = v
    path = out / "result.json"
    tmp = path.with_name(".result.json.tmp")
    tmp.write_text(json.dumps(res, indent=2, sort_keys=True), encoding="utf-8")
    tmp.rename(path)
    return path


def _kv(items: Sequence[str]) -> dict[str, str]:
    out = {}
    for it in items or ():
        k, _, v = it.partition("=")
        out[k] = v
    return out


def _dump(obj: Any, path: Optional[str]) -> None:
    text = json.dumps(obj, indent=2, sort_keys=True)
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text, encoding="utf-8")
    else:
        print(text)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="runbook_tools.py", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def cmd(name: str, *, runbook: bool = False, backend: bool = False, **kw) -> argparse.ArgumentParser:
        p = sub.add_parser(name, **kw)
        if runbook:
            p.add_argument("--runbook", required=True)
        if backend:
            p.add_argument("--backend", required=True)
        return p

    p = cmd("get", runbook=True, help="print one runbook value (JSON for non-scalars)")
    p.add_argument("key")
    p = cmd("ping", help="exit 0 when a GR00T server answers ping")
    p.add_argument("--host", required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--timeout-ms", type=int, default=2000)
    p = cmd("check-ports", backend=True, help="the backend's serve.py port must differ from the stock server's")
    p.add_argument("--port", type=int, required=True)
    p = cmd("write-backend-config", runbook=True, help="backend YAML for this box (never overwrites)")
    for a in ("--example", "--out", "--arena-root", "--arena-python", "--server-python", "--models-dir", "--data-dir",
              "--repo-root"):
        p.add_argument(a, required=True)
    p.add_argument("--server-port", type=int, required=True)
    cmd("env-args", runbook=True, help="the task's env sub-command, one argument per line")
    cmd("dataset-paths", runbook=True, backend=True, help="local paths of the LeRobot set and the Mimic source")
    for name in ("policy-runner-argv", "experiment-argv"):
        p = cmd(name, runbook=True, backend=True, help="one argument per line")
        p.add_argument("--python", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--episodes", type=int, required=True)
        p.add_argument("--seed", type=int, required=True)
        p.add_argument("--host", required=True)
        p.add_argument("--port", type=int, required=True)
        if name == "policy-runner-argv":
            p.add_argument("--num-envs", type=int, required=True)
            p.add_argument("--instruction")
            p.add_argument("--factor", action="append", default=[])
            p.add_argument("--probe", help="a named R7b probe (runbook probes)")
    p = cmd("mimic-argv", runbook=True, backend=True, help="one argument per line")
    p.add_argument("--trials", type=int, required=True)
    p.add_argument("--output-file", required=True)
    p.add_argument("--factor", action="append", default=[])
    p.add_argument("--seed-flag")
    p.add_argument("--seed", type=int)
    p = cmd("summarize", runbook=True, help="summary.json of one Arena run directory")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--episodes", type=int, required=True)
    p.add_argument("--num-envs", type=int, required=True)
    p.add_argument("--exit-code", type=int, required=True)
    p.add_argument("--log")
    p.add_argument("--extra", default="{}", help="JSON merged into the summary")
    p.add_argument("--json-out")
    p = cmd("compare", runbook=True, help="two summaries of the same seeds")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--json-out")
    p = cmd("instruction-strings", runbook=True, backend=True, help="print the three instruction sources in one line")
    p.add_argument("--json-out", required=True)
    p = cmd("vram-summary")
    p.add_argument("--csv", required=True)
    p.add_argument("--json-out")
    p = cmd("client-tests")
    p.add_argument("--log", required=True)
    p.add_argument("--exit-code", type=int, required=True)
    p.add_argument("--json-out")
    p = cmd("checkpoint-report")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--json-out")
    p = cmd("pick-lerobot")
    p.add_argument("--files", required=True)
    p.add_argument("--hint", required=True)
    p = cmd("h5-probe")
    p.add_argument("--hdf5", required=True)
    p.add_argument("--camera", required=True)
    p.add_argument("--json-out")
    p = cmd("mimic-check-result", runbook=True)
    p.add_argument("--out", required=True)
    p.add_argument("--flag", default="")
    p.add_argument("--flag-in-help", choices=("true", "false"), required=True)
    p.add_argument("--json-out")
    p = cmd("pipeline-variant")
    p.add_argument("--pipeline", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-steps", type=int, required=True)
    p.add_argument("--timeout-s", type=float, required=True)
    p = cmd("nominal-task", backend=True, help="the one instruction string of the nominal LeRobot set")
    p.add_argument("--pipeline", required=True)
    p = cmd("tune-top", backend=True, help="tune_top_llm_layers of the base checkpoint")
    p.add_argument("--pipeline", required=True)
    p = cmd("convert", runbook=True, backend=True)
    p.add_argument("--pipeline", required=True)
    p.add_argument("--hdf5", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--instruction")
    p.add_argument("--json-out")
    p = cmd("mix", runbook=True, backend=True)
    p.add_argument("--pipeline", required=True)
    p.add_argument("--shifted", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--json-out")
    p = cmd("dryrun-report")
    p.add_argument("--dataset", required=True)
    p.add_argument("--finetune-out", required=True)
    p.add_argument("--exit-code", type=int, required=True)
    p.add_argument("--max-steps", type=int, required=True)
    p.add_argument("--json-out")
    p = cmd("grid-variant", runbook=True, help="the runbook grid with another episode count (R6)")
    p.add_argument("--episodes", type=int, required=True)
    p.add_argument("--out", required=True)
    p = cmd("result", help="write <out>/result.json")
    p.add_argument("--out", required=True)
    p.add_argument("--script", required=True)
    p.add_argument("--run-ts", required=True)
    p.add_argument("--json", action="append", default=[], help="key=path of a JSON file")
    p.add_argument("--set", action="append", default=[], help="key=value (JSON or text)")

    a = ap.parse_args(argv)
    rb = load_runbook(a.runbook) if getattr(a, "runbook", None) else {}
    c = a.cmd
    try:
        if c == "get":
            v = dotted(rb, a.key)
            print(v if isinstance(v, (str, int, float)) and not isinstance(v, bool) else json.dumps(v))
        elif c == "ping":
            from vla_isaac.backend.server import ping

            return 0 if ping(a.host, a.port, a.timeout_ms) else 1
        elif c == "check-ports":
            from vla_isaac.config import load_backend_config

            port = load_backend_config(a.backend).server.port
            if port == a.port:
                raise RunbookError(f"the backend config's server.port ({port}) is the stock server's port (GR00T_PORT); "
                                   f"IsaacBackend starts its own serve.py and refuses a busy port. Use another port")
            print(port)
        elif c == "write-backend-config":
            out = Path(a.out)
            if out.exists():
                print(f"exists, not overwritten: {out}")
                from vla_isaac.config import load_backend_config

                load_backend_config(out)
                return 0
            import yaml

            d = _yaml(Path(a.example))
            task = rb["task_config"]
            ckpt_id = "/".join((rb["checkpoint"]["repo"], rb["checkpoint"]["subfolder"]))
            models, data, repo = Path(a.models_dir), Path(a.data_dir), Path(a.repo_root)
            d.update(arena_root=a.arena_root, gr00t_root=str(Path(a.arena_root) / "submodules" / "Isaac-GR00T"),
                     arena_python=a.arena_python, server_python=a.server_python,
                     output_root=str(data / "vla-isaac-runs"), task_config=task,
                     workspace_store=str(data / "vla-isaac-store"),
                     workspace_bases={"pw": str(repo / "policy-wrapper")},
                     checkpoints={ckpt_id: str(models / "ckpt")},
                     datasets={k: str(data / Path(v).name) for k, v in d.get("datasets", {}).items()})
            d["server"]["port"] = a.server_port
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("# Written by vla-isaac/scripts/gpu/R0_setup.sh from backend.example.yaml\n"
                           + yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
            from vla_isaac.config import load_backend_config

            load_backend_config(out)
            print(out)
        elif c == "dataset-paths":
            from vla_isaac.config import load_backend_config, load_task_config

            cfg, task = load_backend_config(a.backend), load_task_config(rb["task_config"])
            for ds_id in (task.training_spec.dataset, task.reference.mimic.source_dataset):
                if ds_id not in cfg.datasets:
                    raise RunbookError(f"dataset {ds_id!r} is not in the backend config's datasets map")
                print(cfg.datasets[ds_id])
        elif c == "env-args":
            print("\n".join(env_args(rb)))
        elif c == "policy-runner-argv":
            cond = _condition(a.factor)
            if a.probe:
                cond.update(rb["probes"][a.probe])
            print("\n".join(policy_runner_cmd(rb, a.backend, python=a.python, out=a.out, episodes=a.episodes,
                                              num_envs=a.num_envs, seed=a.seed, host=a.host, port=a.port,
                                              instruction=a.instruction, condition=cond)))
        elif c == "experiment-argv":
            print("\n".join(experiment_runner_cmd(rb, a.backend, python=a.python, out=a.out, episodes=a.episodes,
                                                  seed=a.seed, host=a.host, port=a.port)))
        elif c == "mimic-argv":
            print("\n".join(mimic_cmd(rb, a.backend, trials=a.trials, output_file=a.output_file,
                                      condition=_condition(a.factor), seed_flag=a.seed_flag, seed=a.seed)))
        elif c == "summarize":
            _dump(summarize_run(a.run_dir, episodes=a.episodes, num_envs=a.num_envs, exit_code=a.exit_code, log=a.log,
                                confidence=rb["gate"]["confidence"], extra=json.loads(a.extra)), a.json_out)
        elif c == "compare":
            first, second = (json.loads(Path(p).read_text(encoding="utf-8")) for p in (a.a, a.b))
            _dump(compare_runs(first, second, alpha=rb["gate"]["alpha"]), a.json_out)
        elif c == "instruction-strings":
            info = instruction_strings(rb, a.backend)
            _dump(info, a.json_out)
            print(instruction_line(info))
        elif c == "vram-summary":
            _dump(vram_summary(a.csv), a.json_out)
        elif c == "client-tests":
            _dump(client_tests_report(Path(a.log).read_text(encoding="utf-8", errors="replace"), a.exit_code), a.json_out)
        elif c == "checkpoint-report":
            _dump(checkpoint_report(a.ckpt), a.json_out)
        elif c == "pick-lerobot":
            print(pick_lerobot(Path(a.files).read_text(encoding="utf-8").split(), a.hint))
        elif c == "h5-probe":
            _dump(h5_probe(a.hdf5, a.camera), a.json_out)
        elif c == "mimic-check-result":
            _dump(mimic_check_result(a.out, rb, flag=a.flag or None, flag_in_help=a.flag_in_help == "true"), a.json_out)
        elif c == "pipeline-variant":
            print(pipeline_variant(a.pipeline, a.out, max_steps=a.max_steps, timeout_s=a.timeout_s))
        elif c == "nominal-task":
            print(nominal_task(a.backend, a.pipeline))
        elif c == "tune-top":
            from vla_isaac.config import load_backend_config
            from vla_isaac.data.config import PipelineConfig
            from vla_isaac.train.finetune_launcher import read_tune_top_llm_layers

            ft = PipelineConfig.load(a.pipeline).finetune
            local = load_backend_config(a.backend).checkpoints.get(ft.base_checkpoint)
            if local is None:
                raise RunbookError(f"base checkpoint {ft.base_checkpoint!r} is not in the backend checkpoints map")
            top = read_tune_top_llm_layers(Path(local))
            print(json.dumps({"tune_top_llm_layers": top.value, "source": top.source, "reason": top.reason}))
        elif c == "convert":
            _dump(convert_dryrun(rb, a.backend, a.pipeline, a.hdf5, a.work, instruction=a.instruction), a.json_out)
        elif c == "mix":
            _dump(mix_dryrun(rb, a.backend, a.pipeline, a.shifted, a.out), a.json_out)
        elif c == "dryrun-report":
            _dump(dryrun_report(a.dataset, a.finetune_out, a.exit_code, a.max_steps), a.json_out)
        elif c == "grid-variant":
            import yaml

            d = _yaml(Path(rb["grid"]))
            d["episodes"] = a.episodes
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
            from vla_isaac.experiments.grid import GridSpec

            GridSpec.load(a.out)
            print(a.out)
        elif c == "result":
            print(write_result(a.out, a.script, run_ts=a.run_ts, jsons=_kv(a.json), sets=_kv(a.set)))
    except RunbookError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
