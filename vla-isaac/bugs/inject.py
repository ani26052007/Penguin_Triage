#!/usr/bin/env python3
"""Build a policy-wrapper repo with injected bugs, as the repair agent will get it (plan 3.3, 6.4).

    python inject.py inject --base <clean policy-wrapper checkout> --out <new dir> --bug BUG-03 [--bug ...]
                            [--case CASE-01] [--checkpoint-stats <checkpoint statistics.json>] [--embodiment <name>]
    python inject.py audit --repo <injected repo> --base <clean checkout>
    python inject.py list

``inject`` exports the base's HEAD tree (``git archive``: tracked files only, no history), applies each
bug's patch from ``ledger.yaml`` with ``git apply``, writes the bug's generated files, sets every file's and
directory's mtime to the base HEAD's commit time, and commits the result as the single commit of a fresh
repo (empty git template, so no sample hooks; author, committer and date are the base HEAD's, the message is
the base's root commit subject). With no ``--bug`` it builds the clean repo the same way, so a clean and an
injected container look alike. ``--case`` adds a combined case's bugs and prints its scene condition.

``audit`` checks an injected repo for leaks (``audit_repo``). This file, the ledger and the patches stay
outside the agent's container; nothing they contain may reach an injected repo.
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

HERE = Path(__file__).resolve().parent
LEDGER = HERE / "ledger.yaml"
BUG_FIELDS = ("id", "name", "targets", "patch", "tag", "symptom", "verdict", "evidence", "detection", "difficulty",
              "silent", "plan_cases")
TAGS = ("generic", "gr1-specific")
DETECTIONS = ("synthetic", "real-config", "needs-gpu", "synthetic+needs-gpu", "rung0")
DIFFICULTIES = ("easy", "medium", "hard")
VERDICTS = ("code", "infra", "rung0")
# Words no injected repo may contain, in any path, file, commit or git metadata (case-insensitive).
FORBIDDEN_WORDS = ("bug", "inject", "ledger", "todo_bug")


class InjectError(RuntimeError):
    """The ledger, a patch, or the requested injection is wrong; the message says how."""


# --------------------------------------------------------------------------- #
# ledger
# --------------------------------------------------------------------------- #

def patch_files(patch: str) -> list[str]:
    """Repo paths a unified diff changes (``+++ b/<path>``)."""
    return [m.group(1) for m in re.finditer(r"^\+\+\+ b/(\S+)", patch, re.MULTILINE)]


@dataclass(frozen=True)
class Bug:
    id: str
    name: str
    targets: tuple[str, ...]
    patch_path: Path
    raw: Mapping[str, Any]

    @property
    def patch(self) -> str:
        return self.patch_path.read_text(encoding="utf-8")

    @property
    def generated(self) -> Optional[Mapping[str, Any]]:
        return self.raw.get("generated")

    @property
    def silent(self) -> bool:
        return bool(self.raw["silent"])


@dataclass(frozen=True)
class Ledger:
    base_commit: str
    base_tree: str
    bugs: Mapping[str, Bug]
    cases: Mapping[str, Mapping[str, Any]]
    raw: Mapping[str, Any] = field(repr=False)

    def bug(self, bug_id: str) -> Bug:
        if bug_id not in self.bugs:
            raise InjectError(f"no bug {bug_id!r} in the ledger; known: {sorted(self.bugs)}")
        return self.bugs[bug_id]

    def secrets(self) -> list[str]:
        """Strings that would identify an injection: ledger ids, bug names, case ids."""
        return sorted({*self.bugs, *(b.name for b in self.bugs.values()), *self.cases})


def load_ledger(path: str | Path = LEDGER) -> Ledger:
    import yaml

    path = Path(path)
    d = yaml.safe_load(path.read_text(encoding="utf-8"))
    bugs: dict[str, Bug] = {}
    for b in d["bugs"]:
        missing = [f for f in BUG_FIELDS if f not in b]
        if missing:
            raise InjectError(f"ledger entry {b.get('id')!r} lacks {missing}")
        if b["id"] in bugs:
            raise InjectError(f"duplicate ledger id {b['id']!r}")
        if b["tag"] not in TAGS or b["detection"] not in DETECTIONS or b["difficulty"] not in DIFFICULTIES \
                or b["verdict"] not in VERDICTS or not isinstance(b["silent"], bool):
            raise InjectError(f"ledger entry {b['id']!r} has a bad tag, detection, difficulty, verdict or silent")
        if "needs-gpu" in b["detection"] and not str(b.get("verify", "")).startswith("V-"):
            raise InjectError(f"ledger entry {b['id']!r} needs a GPU run but names no VERIFY id (verify: V-...)")
        patch_path = path.parent / b["patch"]
        if not patch_path.is_file():
            raise InjectError(f"ledger entry {b['id']!r}: no patch file {patch_path}")
        changed = set(patch_files(patch_path.read_text(encoding="utf-8")))
        gen = {b["generated"]["path"]} if b.get("generated") else set()
        if set(b["targets"]) != changed | gen:
            raise InjectError(f"ledger entry {b['id']!r}: targets {sorted(b['targets'])} are not the patch's files "
                              f"{sorted(changed)} plus the generated {sorted(gen)}")
        bugs[b["id"]] = Bug(b["id"], b["name"], tuple(b["targets"]), patch_path, b)
    cases = {c["id"]: c for c in d.get("cases", [])}
    for c in cases.values():
        for bug_id in c["bugs"]:
            if bug_id not in bugs:
                raise InjectError(f"case {c['id']!r} names unknown bug {bug_id!r}")
    return Ledger(str(d["base_commit"]), str(d.get("base_tree", "")), bugs, cases, d)


# --------------------------------------------------------------------------- #
# the base
# --------------------------------------------------------------------------- #

def _git(repo: Path, *args: str, env: Optional[Mapping[str, str]] = None, data: Optional[bytes] = None) -> bytes:
    r = subprocess.run(["git", "-C", str(repo), *args], input=data, capture_output=True, env=env)
    if r.returncode != 0:
        raise InjectError(f"git {' '.join(args)} in {repo}: {r.stderr.decode(errors='replace').strip()}")
    return r.stdout


@dataclass(frozen=True)
class BaseIdentity:
    commit: str
    author_name: str
    author_email: str
    date: str                    # author date, ISO 8601 with offset, as git prints it
    epoch: int                   # committer time (the mtime every file gets)
    message: str
    committer_name: str
    committer_email: str
    committer_date: str


def base_identity(base: Path) -> BaseIdentity:
    fmt = "%H%x00%an%x00%ae%x00%aI%x00%ct%x00%cn%x00%ce%x00%cI"
    commit, name, email, date, epoch, cname, cemail, cdate = \
        _git(base, "log", "-1", f"--format={fmt}", "HEAD").decode().strip().split("\0")
    root = _git(base, "rev-list", "--max-parents=0", "HEAD").decode().split()[-1]
    subject = _git(base, "log", "-1", "--format=%s", root).decode().strip()
    return BaseIdentity(commit, name, email, date, int(epoch), subject, cname, cemail, cdate)


def export_tree(base: Path, dest: Path) -> None:
    """The tracked files of the base's HEAD, without its history."""
    tar = _git(base, "archive", "--format=tar", "HEAD")
    with tarfile.open(fileobj=io.BytesIO(tar)) as t:
        t.extractall(dest, filter="tar") if hasattr(tarfile, "tar_filter") else t.extractall(dest)


# --------------------------------------------------------------------------- #
# generated files
# --------------------------------------------------------------------------- #

LOCATION_KEYS = ("mean", "min", "max", "q01", "q99", "median")


def transform_statistics(stats: Mapping[str, Any], spec: Mapping[str, Any], embodiment: Optional[str] = None) -> dict:
    """A copy of a checkpoint ``statistics.json`` (``{embodiment: {state|action: {group: {stat: [...]}}}}``)
    with the action statistics changed as ``spec["transform"]`` says. Keys starting with ``_`` are dropped.

    With ``embodiment`` set, only that embodiment is kept and it must have left_arm / right_arm action groups.
    Without it, embodiments lacking those groups (the checkpoint file also holds embodiments with other
    action groups) are copied unchanged, and the call raises if no embodiment could be transformed."""
    out = {k: copy.deepcopy(v) for k, v in stats.items() if not k.startswith("_")}
    if embodiment is not None:
        if embodiment not in out:
            raise InjectError(f"no embodiment {embodiment!r} in the statistics; known: {sorted(out)}")
        out = {embodiment: out[embodiment]}
    kind = spec["transform"]
    changed = 0
    for emb in out.values():
        action = emb.get("action") or {}
        if not {"left_arm", "right_arm"} <= set(action):
            if embodiment is not None:
                raise InjectError(f"statistics have no left_arm / right_arm action groups: {sorted(action)}")
            continue
        changed += 1
        if kind == "swap_arm_groups":
            action["left_arm"], action["right_arm"] = action["right_arm"], action["left_arm"]
        elif kind == "shift_arm_location":
            frac = float(spec["fraction"])
            for group in ("left_arm", "right_arm"):
                g = action[group]
                delta = [frac * (hi - lo) for lo, hi in zip(g["min"], g["max"])]
                for key in LOCATION_KEYS:
                    if key in g:
                        g[key] = [x + d for x, d in zip(g[key], delta)]
        else:
            raise InjectError(f"unknown statistics transform {kind!r}")
    if not changed:
        raise InjectError("no embodiment in the statistics has left_arm / right_arm action groups")
    return out


def write_generated(dest: Path, bug: Bug, checkpoint_stats: Optional[Path], embodiment: Optional[str] = None) -> None:
    spec = bug.generated
    if spec["kind"] != "norm_stats":
        raise InjectError(f"{bug.id}: unknown generated kind {spec['kind']!r}")
    if checkpoint_stats is None:
        raise InjectError(f"{bug.id} needs --checkpoint-stats (the base checkpoint's statistics.json)")
    stats = json.loads(Path(checkpoint_stats).read_text(encoding="utf-8"))
    path = dest / spec["path"]
    if path.exists():
        raise InjectError(f"{bug.id}: {spec['path']} already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stats_json(transform_statistics(stats, spec, embodiment)), encoding="utf-8")


def stats_json(d: Any, indent: int = 0) -> str:
    """JSON with nested objects indented and each list of numbers on one line (a short file, so a fix
    that deletes it stays within Rung 0's diff limit)."""
    if not isinstance(d, dict):
        return json.dumps(d)
    pad = "  " * (indent + 1)
    body = ",\n".join(f"{pad}{json.dumps(k)}: {stats_json(v, indent + 1)}" for k, v in d.items())
    return "{\n" + body + "\n" + "  " * indent + "}" + ("\n" if indent == 0 else "")


# --------------------------------------------------------------------------- #
# inject
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Injected:
    repo: Path
    bugs: tuple[str, ...]
    diff: str                                   # the injected change, for Rung 0's diff check
    condition: Mapping[str, str]                # scene condition of a combined case
    launcher_env: Mapping[str, str]             # environment the incident-mode launcher must set


def _git_env(ident: BaseIdentity, home: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_AUTHOR_NAME=ident.author_name, GIT_AUTHOR_EMAIL=ident.author_email, GIT_AUTHOR_DATE=ident.date,
               GIT_COMMITTER_NAME=ident.committer_name, GIT_COMMITTER_EMAIL=ident.committer_email,
               GIT_COMMITTER_DATE=ident.committer_date, GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
               HOME=str(home), TZ="UTC")
    return env


def set_mtimes(root: Path, epoch: int) -> None:
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        if ".git" in Path(dirpath).relative_to(root).parts:
            continue
        for f in filenames:
            os.utime(Path(dirpath) / f, (epoch, epoch), follow_symlinks=False)
        os.utime(dirpath, (epoch, epoch))


def inject(base: str | Path, out: str | Path, bug_ids: Sequence[str] = (), *, case: Optional[str] = None,
           checkpoint_stats: Optional[str | Path] = None, ledger: Optional[Ledger] = None,
           embodiment: Optional[str] = None) -> Injected:
    base, out = Path(base).resolve(), Path(out).resolve()
    ledger = ledger or load_ledger()
    if out.exists() and any(out.iterdir()):
        raise InjectError(f"{out} exists and is not empty")
    if out == base or base in out.parents or HERE in out.parents or out == HERE:
        raise InjectError(f"{out} is inside the base checkout or the injection tooling")
    ids = list(bug_ids)
    condition: Mapping[str, str] = {}
    if case is not None:
        if case not in ledger.cases:
            raise InjectError(f"no case {case!r} in the ledger; known: {sorted(ledger.cases)}")
        ids += [b for b in ledger.cases[case]["bugs"] if b not in ids]
        condition = dict(ledger.cases[case].get("condition") or {})
    if len(set(ids)) != len(ids):
        raise InjectError(f"a bug is named twice: {ids}")
    bugs = [ledger.bug(i) for i in ids]
    gens = [b.generated["path"] for b in bugs if b.generated]
    if len(set(gens)) != len(gens):
        raise InjectError(f"two bugs generate the same file: {gens}")
    ident = base_identity(base)
    out.mkdir(parents=True, exist_ok=True)
    export_tree(base, out)
    for b in bugs:
        r = subprocess.run(["git", "apply", "--whitespace=error-all", "-p1", "-C1", str(b.patch_path)], cwd=out,
                           capture_output=True, text=True)
        if r.returncode != 0:
            shutil.rmtree(out)
            raise InjectError(f"{b.id}: the patch does not apply to {base} ({ident.commit[:10]}): {r.stderr.strip()}")
        if b.generated:
            write_generated(out, b, Path(checkpoint_stats) if checkpoint_stats else None, embodiment)
    set_mtimes(out, ident.epoch)
    with tempfile.TemporaryDirectory(prefix="pw-") as home, tempfile.TemporaryDirectory(prefix="tpl-") as tpl:
        env = _git_env(ident, Path(home))
        _git(out, "init", "-q", f"--template={tpl}", "-b", "main", env=env)
        _git(out, "add", "-A", env=env)
        _git(out, "commit", "-q", "-m", ident.message, env=env)
    os.utime(out, (ident.epoch, ident.epoch))
    launcher_env: dict[str, str] = {}
    for b in bugs:
        launcher_env.update(b.raw.get("launcher_env") or {})
    with tempfile.TemporaryDirectory(prefix="pw-diff-") as td:
        diff = diff_against_base(base, out, Path(td))
    return Injected(out, tuple(ids), diff, condition, launcher_env)


def diff_against_base(base: Path, repo: Path, work: Path) -> str:
    """The unified diff from the base's HEAD tree to the injected tree, with ``a/`` / ``b/`` paths."""
    plain = _git_env(base_identity(base), work)
    subprocess.run(["git", "init", "-q", "--bare", str(work / "git")], check=True, capture_output=True, env=plain)
    env = {**plain, "GIT_DIR": str(work / "git"), "GIT_WORK_TREE": str(work / "tree")}
    (work / "tree").mkdir()
    export_tree(base, work / "tree")
    subprocess.run(["git", "add", "-A"], cwd=work / "tree", check=True, capture_output=True, env=env)
    subprocess.run(["git", "commit", "-q", "-m", "base"], cwd=work / "tree", check=True, capture_output=True, env=env)
    shutil.rmtree(work / "tree")
    (work / "tree").mkdir()
    (work / "git" / "index").unlink()     # both trees carry the same mtimes: hash every file, trust no stat cache
    export_tree(repo, work / "tree")
    subprocess.run(["git", "add", "-A"], cwd=work / "tree", check=True, capture_output=True, env=env)
    r = subprocess.run(["git", "diff", "--cached", "--no-color", "--no-ext-diff", "--no-renames"], cwd=work / "tree",
                       check=True, capture_output=True, text=True, env=env)
    return r.stdout


# --------------------------------------------------------------------------- #
# audit
# --------------------------------------------------------------------------- #

def _text_hits(text: str, needles: Sequence[str]) -> list[str]:
    low = text.lower()
    return [n for n in needles if n.lower() in low]


def audit_repo(repo: str | Path, base: str | Path, ledger: Optional[Ledger] = None) -> list[str]:
    """Every way the injected repo could point at its injection; empty when there is none.

    One commit, one reflog entry, and no git object that is not reachable from it; no path, file, commit
    or git metadata containing a forbidden word, a ledger id, a bug name, or the tooling's path; every
    file's mtime equal to the base HEAD's commit time; and no blob of the base's version of a changed file."""
    repo, base = Path(repo), Path(base)
    ledger = ledger or load_ledger()
    problems: list[str] = []
    needles = [*FORBIDDEN_WORDS, *ledger.secrets(), "vla-isaac", "vla_isaac", str(HERE)]

    commits = _git(repo, "rev-list", "--all").decode().split()
    if len(commits) != 1:
        problems.append(f"{len(commits)} commits reachable, expected 1")
    for ref in ("HEAD", *_git(repo, "for-each-ref", "--format=%(refname)").decode().split()):
        entries = _git(repo, "reflog", "show", ref).decode().splitlines()
        if len(entries) != 1:
            problems.append(f"{len(entries)} reflog entries for {ref}, expected 1")
    reachable = {line.split()[0] for line in _git(repo, "rev-list", "--objects", "--all").decode().splitlines()}
    stored = set(_git(repo, "cat-file", "--batch-all-objects", "--batch-check=%(objectname)").decode().split())
    if stored - reachable:
        problems.append(f"{len(stored - reachable)} git objects not reachable from the commit")
    counts = dict(line.split(": ") for line in _git(repo, "count-objects", "-v").decode().splitlines())
    if int(counts.get("count", 0)) + int(counts.get("in-pack", 0)) != len(stored):
        problems.append(f"count-objects {counts} disagrees with the {len(stored)} objects listed")

    blobs = _git(repo, "cat-file", "--batch-all-objects", "--batch").decode(errors="replace")
    for n in _text_hits(blobs, needles):
        problems.append(f"git objects contain {n!r}")
    for p in sorted(repo.rglob("*")):
        rel = p.relative_to(repo).as_posix()
        for n in _text_hits(rel, needles):
            problems.append(f"path {rel} contains {n!r}")
        if p.is_file() and not p.is_symlink() and not rel.startswith(".git/objects/") and rel != ".git/index":
            for n in _text_hits(p.read_bytes().decode(errors="replace"), needles):
                problems.append(f"{rel} contains {n!r}")

    ident = base_identity(base)
    for p in sorted(repo.rglob("*")):
        rel = p.relative_to(repo)
        if rel.parts and rel.parts[0] == ".git":
            continue
        if int(p.lstat().st_mtime) != ident.epoch:
            problems.append(f"{rel} mtime {int(p.lstat().st_mtime)} != the base commit time {ident.epoch}")
    commit = _git(repo, "log", "-1", "--format=%an%x00%ae%x00%aI%x00%cn%x00%ce%x00%cI%x00%s").decode().strip().split("\0")
    if commit != [ident.author_name, ident.author_email, ident.date, ident.committer_name, ident.committer_email,
                  ident.committer_date, ident.message]:
        problems.append(f"commit identity {commit} is not the base's")

    for path in _git(repo, "diff", "--name-only", "--no-renames", "4b825dc642cb6eb9a060e54bf8d69288fbee4904", "HEAD") \
            .decode().split():
        try:
            clean = _git(base, "rev-parse", f"HEAD:{path}").decode().strip()
        except InjectError:
            continue                                    # a file the base does not have
        mine = _git(repo, "rev-parse", f"HEAD:{path}").decode().strip()
        if clean != mine and clean in stored:
            problems.append(f"the base's version of {path} is in the git objects")
    return problems


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="inject.py", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("inject", help="build an injected repo")
    p.add_argument("--base", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--bug", action="append", default=[])
    p.add_argument("--case")
    p.add_argument("--checkpoint-stats")
    p.add_argument("--embodiment", help="keep only this embodiment of the checkpoint statistics")
    p.add_argument("--ledger", default=str(LEDGER))
    p = sub.add_parser("audit", help="check an injected repo for leaks")
    p.add_argument("--repo", required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--ledger", default=str(LEDGER))
    p = sub.add_parser("list", help="the ledger's bugs and cases")
    p.add_argument("--ledger", default=str(LEDGER))
    a = ap.parse_args(argv)
    ledger = load_ledger(a.ledger)
    if a.cmd == "list":
        for b in ledger.bugs.values():
            print(f"{b.id}  {b.name:32s} {b.raw['verdict']:6s} silent={b.silent!s:5s} {b.raw['detection']:20s} "
                  f"{', '.join(b.targets)}")
        for c in ledger.cases.values():
            print(f"{c['id']}  bugs {c['bugs']} under {c.get('condition')}")
        return 0
    if a.cmd == "inject":
        res = inject(a.base, a.out, a.bug, case=a.case, checkpoint_stats=a.checkpoint_stats, ledger=ledger,
                     embodiment=a.embodiment)
        problems = audit_repo(res.repo, a.base, ledger)
        print(json.dumps({"repo": str(res.repo), "bugs": res.bugs, "condition": res.condition,
                          "launcher_env": res.launcher_env, "audit": problems or "clean"}, indent=2))
        return 1 if problems else 0
    problems = audit_repo(a.repo, a.base, ledger)
    print("\n".join(problems) if problems else "clean")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
