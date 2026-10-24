"""Code identity and working copies (Fix F2a; ISSUES #6, #7, #8).

A ``CodeRef`` names files that can be rebuilt exactly:

    code_hash = <base>                         the base checkout, unchanged
              | <base>+<sha1(cumulative diff)[:12]>

``<base>`` is a registered base checkout (``WorkspaceManager(bases=...)``). The cumulative diff
is the whole change against that base's snapshot, computed from the files (``cumulative_diff``),
never by concatenating the diffs of the attempts that led there. So the same files always get
the same name, whatever sequence of attempts produced them, and a change that is undone gets
the base name back.

Every cumulative diff is written to a content-addressed store on disk (``PatchStore``), keyed by
the code name. Any ``WorkspaceManager`` on the same store and bases can rebuild any name, and
``LoopResult.final_diff`` is enough to re-register a fix in a new store.

A working copy (``materialize``) is a fresh directory:

    1. the base checkout's files, without its ``.git`` and without what its ``.gitignore`` files
       ignore (``_copy_base``: the one place a working copy is created);
    2. ``git init`` + ONE commit of those files (fixed author and date, so commit ids are
       deterministic), referenced as ``refs/vla/base``. The original history is not copied;
    3. if the code is not the base: the cumulative diff applied and committed on top, so HEAD is
       the code under repair and hagent's ``git diff HEAD`` is exactly one attempt's change;
    4. ``git config vla.base <base>``, so the copy can name its own code (``code_ref_of``,
       Rung 1's CLI default).

The original checkout is only ever read. git runs with the user's global and system config
switched off (``GIT_CONFIG_GLOBAL``/``GIT_CONFIG_NOSYSTEM``), so excludes files, diff options
or hooks on the host cannot change a name.

Readers that only need to look at the files (the mock backend's file rules, the synthetic trace
sources) use ``view(code_ref)``: one cached working copy per name, never to be written to.

TODO(F2b): ``extend`` copies ``eval_config`` from the code it started from (ISSUES #2 is open);
it will come from the patched working copy through a config loader.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tempfile
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

from router.schemas import CodeRef

BASE_REF = "refs/vla/base"
_BASE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")
_CODE_NAME = re.compile(r"^(?P<base>[A-Za-z0-9._-]+)\+(?P<digest>[0-9a-f]{12})$")
_IDENTITY = {"GIT_AUTHOR_NAME": "vla-workspace", "GIT_AUTHOR_EMAIL": "workspace@vla.invalid",
             "GIT_COMMITTER_NAME": "vla-workspace", "GIT_COMMITTER_EMAIL": "workspace@vla.invalid",
             "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+0000", "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+0000"}
# The one diff recipe. Everything that names code goes through ``cumulative_diff``.
_DIFF_ARGS = ("diff", "--no-color", "--no-ext-diff", "--no-textconv", "--binary", "--full-index", "--no-renames",
              "--diff-algorithm=myers", "--src-prefix=a/", "--dst-prefix=b/", BASE_REF)


class WorkspaceError(RuntimeError):
    """Code that cannot be built: unknown name, changed base checkout, corrupt store."""


class UnrebuildablePatch(WorkspaceError):
    """A patch that does not apply to a fresh copy of the code it was written against (for
    example a binary edit in a diff made without ``--binary``)."""


# --------------------------------------------------------------------------- #
# git
# --------------------------------------------------------------------------- #

def _env() -> dict[str, str]:
    return {**os.environ, **_IDENTITY, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C"}


def _git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    p = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, errors="replace", env=_env())
    if check and p.returncode != 0:
        raise WorkspaceError(f"git {' '.join(args[:2])} failed in {cwd}: {p.stderr.strip()[:500]}")
    return p


def _commit(cwd: Path, message: str) -> None:
    _git(cwd, "add", "-A")
    _git(cwd, "commit", "-q", "--no-verify", "--allow-empty", "--no-gpg-sign", "-m", message)


# --------------------------------------------------------------------------- #
# the naming recipe
# --------------------------------------------------------------------------- #

def code_name(base: str, cum_diff: str) -> str:
    """``base`` for an empty diff, else ``base+<sha1(diff)[:12]>``."""
    if not _BASE_NAME.match(base):
        raise WorkspaceError(f"invalid base name {base!r} (letters, digits, '.', '_', '-')")
    if not cum_diff.strip():
        return base
    return f"{base}+{hashlib.sha1(cum_diff.encode('utf-8', 'replace')).hexdigest()[:12]}"


def split_code_name(code_hash: str) -> tuple[str, Optional[str]]:
    """(base, digest) for ``base+digest``; (base, None) for a bare base name."""
    m = _CODE_NAME.match(code_hash)
    if m:
        return m.group("base"), m.group("digest")
    if _BASE_NAME.match(code_hash):
        return code_hash, None
    raise WorkspaceError(f"{code_hash!r} is not a workspace code name (<base> or <base>+<12 hex>)")


def cumulative_diff(workdir: str | Path) -> str:
    """Everything in ``workdir`` that differs from its base snapshot, new files included."""
    workdir = Path(workdir)
    _git(workdir, "add", "-A", "-N")
    return _git(workdir, *_DIFF_ARGS).stdout


def code_ref_of(workdir: str | Path, eval_config: Optional[Mapping] = None) -> CodeRef:
    """The name of the code in a working copy made by ``materialize``, edits included."""
    p = _git(Path(workdir), "config", "--get", "vla.base", check=False)
    base = p.stdout.strip()
    if p.returncode != 0 or not base:
        raise WorkspaceError(f"{workdir} is not a workspace working copy (no vla.base)")
    return CodeRef(code_name(base, cumulative_diff(workdir)), dict(eval_config or {}))


# --------------------------------------------------------------------------- #
# file predicates (shared by the mock backend and the synthetic trace sources)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class FileContains:
    """True when the file ``path`` (relative to the code root) exists and contains ``text``;
    ``present=False`` inverts it. Evaluated on ``WorkspaceManager.view(code_ref)``."""
    path: str
    text: str
    present: bool = True

    def __post_init__(self) -> None:
        p = Path(self.path)
        if p.is_absolute() or ".." in p.parts:
            raise ValueError(f"FileContains path must be relative and inside the code root, got {self.path!r}")

    def __call__(self, root: str | Path) -> bool:
        f = Path(root) / self.path
        found = f.is_file() and self.text in f.read_text(encoding="utf-8", errors="replace")
        return found == self.present

    def to_dict(self) -> dict:
        return {"kind": "file_contains", "path": self.path, "text": self.text, "present": self.present}

    @classmethod
    def from_dict(cls, d: Mapping) -> "FileContains":
        if d.get("kind", "file_contains") != "file_contains":
            raise ValueError(f"unknown file predicate kind {d.get('kind')!r}")
        return cls(d["path"], d["text"], bool(d.get("present", True)))


# --------------------------------------------------------------------------- #
# store
# --------------------------------------------------------------------------- #

class PatchStore:
    """Content-addressed, on disk: ``patches/<code name>.patch`` holds the cumulative diff, and the
    name is the hash of that diff, so a file can be checked against its own name.
    ``bases/<base>.tree`` records the git tree of each base snapshot the first time it is built,
    so a base checkout that changes afterwards is an error, not a silent new meaning for its name."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        (self.root / "patches").mkdir(parents=True, exist_ok=True)
        (self.root / "bases").mkdir(parents=True, exist_ok=True)

    def _write(self, path: Path, text: str) -> None:
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)

    def put(self, code_hash: str, cum_diff: str) -> None:
        base, digest = split_code_name(code_hash)
        if digest is None or code_name(base, cum_diff) != code_hash:
            raise WorkspaceError(f"diff does not hash to {code_hash!r}")
        path = self.root / "patches" / f"{code_hash}.patch"
        if path.exists():
            if path.read_text(encoding="utf-8") != cum_diff:
                raise WorkspaceError(f"store already holds a different diff for {code_hash!r} (hash collision)")
            return
        self._write(path, cum_diff)

    def get(self, code_hash: str) -> str:
        path = self.root / "patches" / f"{code_hash}.patch"
        if not path.is_file():
            raise WorkspaceError(f"code {code_hash!r} is not in the store {self.root}")
        diff = path.read_text(encoding="utf-8")
        base, _ = split_code_name(code_hash)
        if code_name(base, diff) != code_hash:
            raise WorkspaceError(f"store file for {code_hash!r} does not match its name (corrupt store)")
        return diff

    def check_base_tree(self, base: str, tree: str) -> None:
        path = self.root / "bases" / f"{base}.tree"
        if not path.exists():
            self._write(path, tree + "\n")
            return
        recorded = path.read_text(encoding="utf-8").strip()
        if recorded != tree:
            raise WorkspaceError(f"base checkout {base!r} changed since it was first snapshotted "
                                 f"(tree {tree[:12]}, recorded {recorded[:12]}); register it under a new name")


# --------------------------------------------------------------------------- #
# manager
# --------------------------------------------------------------------------- #

class WorkspaceManager:
    """``bases``: base name -> checkout directory (read only). ``store``: the patch store directory
    (created if missing; always written)."""

    def __init__(self, store: str | Path, bases: Mapping[str, str | Path]):
        self.store = PatchStore(store)
        self.bases = {name: Path(p).resolve() for name, p in bases.items()}
        for name, p in self.bases.items():
            code_name(name, "")                                    # validates the name
            if not p.is_dir():
                raise WorkspaceError(f"base checkout {name!r}: {p} is not a directory")
        self._views: dict[str, Path] = {}
        self._scratch: Optional[Path] = None

    # -- names ---------------------------------------------------------------
    def _resolve(self, code_hash: str) -> tuple[str, str]:
        base, digest = split_code_name(code_hash)
        if base not in self.bases:
            raise WorkspaceError(f"unknown base {base!r} for code {code_hash!r}; registered: {sorted(self.bases)}")
        return base, "" if digest is None else self.store.get(code_hash)

    def diff_of(self, code_ref: CodeRef) -> str:
        """The cumulative diff of ``code_ref`` against its base ("" for the base itself)."""
        return self._resolve(code_ref.code_hash)[1]

    def register(self, code_ref: CodeRef, cum_diff: str) -> CodeRef:
        """Add a known cumulative diff (e.g. ``LoopResult.final_diff``) to the store. The name must
        match the diff."""
        base, _ = split_code_name(code_ref.code_hash)
        if code_name(base, cum_diff) != code_ref.code_hash:
            raise WorkspaceError(f"diff does not hash to {code_ref.code_hash!r}")
        if base not in self.bases:
            raise WorkspaceError(f"unknown base {base!r}")
        if cum_diff.strip():
            self.store.put(code_ref.code_hash, cum_diff)
        return code_ref

    # -- working copies --------------------------------------------------------
    def _copy_base(self, src: Path, dest: Path) -> None:
        """The one place a working copy's files come from. No ``.git`` (at any depth), and then
        nothing the tree's own ``.gitignore`` files ignore. F5 (leakage) adds its filters here."""
        shutil.copytree(src, dest, symlinks=True, ignore=shutil.ignore_patterns(".git"), dirs_exist_ok=True)

    def _snapshot_base(self, base: str, dest: Path) -> None:
        self._copy_base(self.bases[base], dest)
        _git(dest, "init", "-q", "--template=", "-b", "vla")
        _git(dest, "add", "-A")
        _git(dest, "clean", "-f", "-d", "-X", "-q")                # drop ignored files from the copy
        _git(dest, "commit", "-q", "--no-verify", "--allow-empty", "--no-gpg-sign", "-m", f"vla: base {base}")
        self.store.check_base_tree(base, _git(dest, "rev-parse", "HEAD^{tree}").stdout.strip())
        _git(dest, "update-ref", BASE_REF, "HEAD")
        _git(dest, "config", "vla.base", base)

    @staticmethod
    def _apply(dest: Path, patch: str) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False, encoding="utf-8") as fh:
            fh.write(patch if patch.endswith("\n") else patch + "\n")
        try:
            p = _git(dest, "apply", "--whitespace=nowarn", fh.name, check=False)
        finally:
            os.unlink(fh.name)
        if p.returncode != 0:
            raise UnrebuildablePatch(p.stderr.strip()[:1000] or "git apply failed")

    def materialize(self, code_ref: CodeRef, dest: Optional[str | Path] = None) -> Path:
        """Build ``code_ref`` in a fresh directory (``dest``, which must not exist or be empty;
        default a new temp dir the caller owns) and return its path."""
        base, diff = self._resolve(code_ref.code_hash)
        if dest is None:
            dest = Path(tempfile.mkdtemp(prefix="ws-"))
        dest = Path(dest)
        if dest.exists() and any(dest.iterdir()):
            raise WorkspaceError(f"{dest} is not empty")
        dest.mkdir(parents=True, exist_ok=True)
        self._snapshot_base(base, dest)
        if diff:
            try:
                self._apply(dest, diff)
            except UnrebuildablePatch as e:
                raise WorkspaceError(f"stored diff of {code_ref.code_hash!r} does not apply: {e}") from e
            _commit(dest, f"vla: code {code_ref.code_hash}")
        return dest

    def extend(self, code_ref: CodeRef, patch: str) -> CodeRef:
        """The code you get by applying ``patch`` (one attempt's diff, written against
        ``code_ref``'s files) to ``code_ref``. The new name comes from diffing the resulting files
        against the base snapshot. Raises ``UnrebuildablePatch`` if the patch does not apply."""
        if not patch.strip():
            return code_ref
        with tempfile.TemporaryDirectory(prefix="ws-extend-") as td:
            work = self.materialize(code_ref, Path(td) / "repo")
            self._apply(work, patch)
            return self.adopt(work, code_ref.eval_config)

    def adopt(self, workdir: str | Path, eval_config: Optional[Mapping] = None) -> CodeRef:
        """Name the code in a working copy (edits included) and store its cumulative diff."""
        ref = code_ref_of(workdir, eval_config)
        diff = cumulative_diff(workdir)
        return self.register(ref, diff)

    def view(self, code_ref: CodeRef) -> Path:
        """A cached working copy of ``code_ref`` for readers. Do not write to it."""
        name = code_ref.code_hash
        if name not in self._views:
            if self._scratch is None:
                self._scratch = Path(tempfile.mkdtemp(prefix="ws-views-"))
                weakref.finalize(self, shutil.rmtree, str(self._scratch), True)
            self._views[name] = self.materialize(code_ref, self._scratch / f"v{len(self._views)}")
        return self._views[name]


__all__ = ["BASE_REF", "FileContains", "PatchStore", "UnrebuildablePatch", "WorkspaceError", "WorkspaceManager",
           "code_name", "code_ref_of", "cumulative_diff", "split_code_name"]
