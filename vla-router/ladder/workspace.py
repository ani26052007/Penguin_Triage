"""Code identity and working copies (Fix F2a; ISSUES #6, #7, #8; F2a review fixes).

A ``CodeRef`` names files that can be rebuilt exactly:

    code_hash = <label>@<tree12>                     a base checkout, as snapshotted
              | <label>@<tree12>+<sha1(diff)[:12]>   that base plus a cumulative diff

``<label>`` is a registered base checkout (``WorkspaceManager(bases={label: dir})``) and
``<tree12>`` the first 12 hex digits of the git tree id of its snapshot (``base_ref(label)``).
The base's content is therefore part of every name: a checkout that changes gets new names, a
name made from the old content is refused by ``materialize`` ("base checkout changed"), and
rollout-cache entries (keyed on ``CodeRef.key()``) for the old content stop matching, whichever
patch store is used. Names from before the F2a review fixes (``c0+<12 hex>``) are rejected when
parsed.

The cumulative diff is the whole change against that base's snapshot, computed from the files
(``cumulative_diff``), never by concatenating the diffs of the attempts that led there. So the
same files always get the same name, whatever sequence of attempts produced them, and a change
that is undone gets the base name back.

Bytes, not text: git output, diffs, the hash input, the store files and ``git apply`` input are
bytes, so a file that is not valid UTF-8 round-trips exactly. Where a diff has to be a ``str``
(``cumulative_diff``, ``diff_of``, the loop's records), it is the lossless ``surrogateescape``
view (``as_text`` / ``as_bytes``), which survives ``json.dumps`` / ``json.loads``.

Every cumulative diff is written to a content-addressed store on disk (``PatchStore``), keyed by
the code name. Any ``WorkspaceManager`` on the same store and bases can rebuild any name, and
``LoopResult.final_diff`` is enough to re-register a fix in a new store.

A working copy (``materialize``) is a fresh directory:

    1. the base checkout's files without its ``.git`` (``_copy_base``: the one place a working
       copy's files come from). ``_snapshot`` then keeps every file the original checkout tracks,
       even one a ``.gitignore`` matches, and drops only untracked ignored files (``git clean -X``);
    2. ``git init`` + ONE commit of those files (fixed author and date, so commit ids are
       deterministic), referenced as ``refs/vla/base``. The original history is not copied;
    3. if the code is not the base: the cumulative diff applied and committed on top, so HEAD is
       the code under repair and hagent's ``git diff HEAD`` is exactly one attempt's change;
    4. ``git config vla.base <label>@<tree12>``, so the copy can name its own code
       (``code_ref_of``, Rung 1's CLI default).

A symbolic link that is absolute or resolves outside the tree is rejected (base checkout:
``WorkspaceError``; patch: ``UnrebuildablePatch(cause="symlink")``): the agent could write
outside its working copy through it, and its target is not part of the name.

The original checkout is only ever read (copied, and ``git ls-files`` for its tracked files). git
runs with every inherited ``GIT_*`` variable removed and with the user's global and system config
switched off, so an inherited ``GIT_DIR`` / ``GIT_WORK_TREE`` cannot point it at the original, and
excludes files, diff options or hooks on the host cannot change a name.

Readers that only need to look at the files (the mock backend's file rules, the synthetic trace
sources) use ``view(code_ref)``: one cached working copy per name, never to be written to.

The eval config of a ``CodeRef`` is read from the files, never carried over from the code it was
built from (ISSUES #2): ``code_ref_of``, ``adopt``, ``extend`` and ``base_ref`` (unless given an
explicit config) call a config loader on the working copy, so a config edit by a repair agent is
what the patched ``CodeRef`` says. ``file_config_loader(path)`` is the loader for a YAML / JSON file
in the repo (YAML needs the optional ``yaml`` extra; the format is TODO(verify)); ``WorkspaceManager(config_loader=...)`` replaces it. A missing or unreadable file is a
``WorkspaceError``, not an empty config.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Union

from router.schemas import CodeRef

BASE_REF = "refs/vla/base"
_LABEL = r"[A-Za-z0-9._-]+"
_LABEL_RE = re.compile(rf"^{_LABEL}$")
_NAME_RE = re.compile(rf"^(?P<label>{_LABEL})@(?P<tree>[0-9a-f]{{12}})(?:\+(?P<digest>[0-9a-f]{{12}}))?$")
_OLD_RECIPE_RE = re.compile(rf"^{_LABEL}(?:\+[0-9a-f]{{12}})+$")
_IDENTITY = {"GIT_AUTHOR_NAME": "vla-workspace", "GIT_AUTHOR_EMAIL": "workspace@vla.invalid",
             "GIT_COMMITTER_NAME": "vla-workspace", "GIT_COMMITTER_EMAIL": "workspace@vla.invalid",
             "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+0000", "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+0000"}
# The one diff recipe. Everything that names code goes through ``_cumulative_diff``.
_DIFF_ARGS = ("diff", "--no-color", "--no-ext-diff", "--no-textconv", "--binary", "--full-index", "--no-renames",
              "--diff-algorithm=myers", "--src-prefix=a/", "--dst-prefix=b/", BASE_REF)
_GITLINK_MODE = b"160000"

Diff = Union[str, bytes]


class WorkspaceError(RuntimeError):
    """Code that cannot be built: unknown or old-recipe name, changed base checkout, corrupt store,
    an eval config that cannot be read."""


ConfigLoader = Callable[[Path], Mapping[str, Any]]
EVAL_CONFIG_PATH = "config/eval.yaml"        # TODO(verify): where the controller repo keeps its eval config


def file_config_loader(path: str = EVAL_CONFIG_PATH) -> ConfigLoader:
    """A loader that reads the eval config from ``path`` (relative to the working copy): JSON for
    ``.json``, YAML otherwise. The file must exist and hold a mapping (an empty YAML file is ``{}``)."""
    def load(workdir: Path) -> Mapping[str, Any]:
        f = Path(workdir) / path
        if not f.is_file():
            raise WorkspaceError(f"no eval config at {path!r} in {workdir}; pass a config_loader that reads "
                                 f"this repo's config")
        try:
            text = f.read_text(encoding="utf-8")
            if f.suffix == ".json":
                data = json.loads(text)
            else:
                try:
                    import yaml
                except ImportError:
                    raise WorkspaceError(f"reading {path!r} needs pyyaml (pip install 'vla-repair-ladder[yaml]'), "
                                         f"or use a .json config or a custom config_loader") from None
                data = yaml.safe_load(text)
        except WorkspaceError:
            raise
        except Exception as e:                       # unreadable file, bad JSON / YAML
            raise WorkspaceError(f"cannot read the eval config {path!r}: {e}") from e
        if data is None:
            return {}
        if not isinstance(data, Mapping):
            raise WorkspaceError(f"eval config {path!r} is not a mapping")
        return data
    return load


load_eval_config: ConfigLoader = file_config_loader()


class UnrebuildablePatch(WorkspaceError):
    """A patch that cannot be rebuilt on a fresh copy of the code it was written against.
    ``cause``: ``binary`` (a binary edit in a diff made without ``--binary``), ``encoding`` (the
    patch text lost bytes that are not valid UTF-8), ``symlink`` (it adds a link that points
    outside the tree), ``does_not_apply`` (anything else ``git apply`` refuses), ``mismatch``
    (it applies, but does not give the files of the working copy it came from),
    ``config_unreadable`` (it applies, but the eval config of the patched files cannot be read)."""

    CAUSES = ("binary", "encoding", "symlink", "does_not_apply", "mismatch", "config_unreadable")

    def __init__(self, cause: str, message: str):
        if cause not in self.CAUSES:
            raise ValueError(f"unknown cause {cause!r}")
        super().__init__(message)
        self.cause = cause


# --------------------------------------------------------------------------- #
# bytes <-> the lossless str view
# --------------------------------------------------------------------------- #

def as_bytes(diff: Diff) -> bytes:
    """A diff as bytes. A ``str`` is read as the ``surrogateescape`` view (``as_text``)."""
    if isinstance(diff, bytes):
        return diff
    try:
        return diff.encode("utf-8", "surrogateescape")
    except UnicodeEncodeError as e:
        raise WorkspaceError(f"diff text is not a lossless view of bytes: {e}") from e


def as_text(diff: bytes) -> str:
    """The lossless ``str`` view of a diff: invalid UTF-8 bytes become lone surrogates."""
    return diff.decode("utf-8", "surrogateescape")


# --------------------------------------------------------------------------- #
# git
# --------------------------------------------------------------------------- #

def _env(extra: Optional[Mapping[str, str]] = None) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(_IDENTITY, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1", GIT_TERMINAL_PROMPT="0",
               GIT_OPTIONAL_LOCKS="0", LC_ALL="C")
    env.update(extra or {})
    return env


def _show(b: bytes) -> str:
    """git's stderr for an error message (display only)."""
    return b.decode("utf-8", "backslashreplace").strip()


def _git(cwd: Path, *args: str, check: bool = True, input: Optional[bytes] = None,
         env: Optional[Mapping[str, str]] = None) -> subprocess.CompletedProcess:
    p = subprocess.run(["git", "-c", "core.fsmonitor=false", *args], cwd=cwd, capture_output=True,
                       input=input, env=_env(env))
    if check and p.returncode != 0:
        raise WorkspaceError(f"git {' '.join(args[:2])} failed in {cwd}: {_show(p.stderr)[:500]}")
    return p


def _out(cwd: Path, *args: str) -> str:
    return _git(cwd, *args).stdout.decode("ascii").strip()


def _commit(cwd: Path, message: str) -> None:
    _git(cwd, "add", "-A")
    _git(cwd, "commit", "-q", "--no-verify", "--allow-empty", "--no-gpg-sign", "-m", message)


# --------------------------------------------------------------------------- #
# the naming recipe
# --------------------------------------------------------------------------- #

def split_code_name(code_hash: str) -> tuple[str, str, Optional[str]]:
    """(label, tree12, digest) for ``label@tree12+digest``; digest None for a base."""
    m = _NAME_RE.match(code_hash)
    if m:
        return m.group("label"), m.group("tree"), m.group("digest")
    if _OLD_RECIPE_RE.match(code_hash):
        raise WorkspaceError(
            f"{code_hash!r} is an old-recipe code name (<base>+sha1 of a diff, from before the F2a review fixes): "
            f"it does not say which base snapshot it was made from and cannot be rebuilt. Name the code with "
            f"WorkspaceManager.base_ref() and extend()")
    if _LABEL_RE.match(code_hash):
        raise WorkspaceError(f"{code_hash!r} is a bare base label, not a code name: use "
                             f"WorkspaceManager.base_ref({code_hash!r})")
    raise WorkspaceError(f"{code_hash!r} is not a workspace code name (<label>@<tree12>[+<diff12>])")


def base_of(code_hash: str) -> str:
    """The base name (``label@tree12``) of any code name."""
    label, tree, _ = split_code_name(code_hash)
    return f"{label}@{tree}"


def code_name(base: str, cum_diff: Diff) -> str:
    """``base`` (a ``label@tree12`` name) for an empty diff, else ``base+<sha1(diff bytes)[:12]>``."""
    _, _, digest = split_code_name(base)
    if digest is not None:
        raise WorkspaceError(f"{base!r} is not a base name (it already carries a diff)")
    data = as_bytes(cum_diff)
    if not data.strip():
        return base
    return f"{base}+{hashlib.sha1(data).hexdigest()[:12]}"


def _cumulative_diff(workdir: Path) -> bytes:
    _git(workdir, "add", "-A", "-N")
    return _git(workdir, *_DIFF_ARGS).stdout


def cumulative_diff(workdir: str | Path) -> str:
    """Everything in ``workdir`` that differs from its base snapshot, new files included (the
    lossless ``str`` view of the bytes)."""
    return as_text(_cumulative_diff(Path(workdir)))


def _named(workdir: Path) -> tuple[str, bytes]:
    p = _git(workdir, "config", "--get", "vla.base", check=False)
    base = p.stdout.decode("ascii", "backslashreplace").strip()
    if p.returncode != 0 or not base:
        raise WorkspaceError(f"{workdir} is not a workspace working copy (no vla.base)")
    diff = _cumulative_diff(workdir)
    return code_name(base, diff), diff


def code_ref_of(workdir: str | Path, config_loader: ConfigLoader = load_eval_config) -> CodeRef:
    """The code in a working copy made by ``materialize``, edits included: its name, and the eval
    config ``config_loader`` reads from the copy's files."""
    return CodeRef(_named(Path(workdir))[0], dict(config_loader(Path(workdir))))


# --------------------------------------------------------------------------- #
# file predicates (shared by the mock backend and the synthetic trace sources)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class FileContains:
    """True when the file ``path`` (relative to the code root) exists and its bytes contain the
    UTF-8 bytes of ``text``; ``present=False`` inverts it. Evaluated on
    ``WorkspaceManager.view(code_ref)``."""
    path: str
    text: str
    present: bool = True

    def __post_init__(self) -> None:
        p = Path(self.path)
        if p.is_absolute() or ".." in p.parts:
            raise ValueError(f"FileContains path must be relative and inside the code root, got {self.path!r}")

    def __call__(self, root: str | Path) -> bool:
        f = Path(root) / self.path
        found = f.is_file() and self.text.encode("utf-8") in f.read_bytes()
        return found == self.present

    def to_dict(self) -> dict:
        return {"kind": "file_contains", "path": self.path, "text": self.text, "present": self.present}

    @classmethod
    def from_dict(cls, d: Mapping) -> "FileContains":
        if d.get("kind", "file_contains") != "file_contains":
            raise ValueError(f"unknown file predicate kind {d.get('kind')!r}")
        return cls(d["path"], d["text"], bool(d.get("present", True)))


# --------------------------------------------------------------------------- #
# tree checks
# --------------------------------------------------------------------------- #

def _escaping_symlinks(root: Path) -> list[str]:
    """Links under ``root`` (outside its top-level ``.git``) that are absolute or resolve outside it."""
    top = os.path.realpath(root)
    bad = []
    for dirpath, dirnames, filenames in os.walk(root):
        if Path(dirpath) == Path(root) and ".git" in dirnames:
            dirnames.remove(".git")
        for name in dirnames + filenames:
            p = os.path.join(dirpath, name)
            if not os.path.islink(p):
                continue
            target = os.readlink(p)
            if os.path.isabs(target) or os.path.commonpath([top, os.path.realpath(p)]) != top:
                bad.append(f"{os.path.relpath(p, root)} -> {target}")
    return sorted(bad)


def _tracked_files(src: Path) -> list[bytes]:
    """Paths (relative to ``src``) that the original checkout's git tracks; [] when ``src`` is not
    in a git repository. Read only: ``ls-files`` with optional locks off."""
    p = _git(src, "ls-files", "-z", "--stage", check=False)
    if p.returncode != 0:
        if b"not a git repository" in p.stderr:
            return []
        raise WorkspaceError(f"cannot list the tracked files of {src}: {_show(p.stderr)[:300]}")
    out: dict[bytes, None] = {}
    for entry in p.stdout.split(b"\0"):
        if entry:
            meta, path = entry.split(b"\t", 1)
            if meta.split(b" ", 1)[0] != _GITLINK_MODE:            # a submodule's files are copied as files
                out[path] = None
    return list(out)


def _cause(patch: bytes, stderr: bytes) -> str:
    if re.search(rb"^Binary files .* differ$", patch, re.M) or b"binary patch" in stderr \
            or b"without full index" in stderr:
        return "binary"
    if "�".encode("utf-8") in patch:
        return "encoding"
    return "does_not_apply"


# --------------------------------------------------------------------------- #
# store
# --------------------------------------------------------------------------- #

class PatchStore:
    """Content-addressed, on disk: ``patches/<code name>.patch`` holds the cumulative diff as
    bytes, and the name is the hash of those bytes, so every read checks the file against its
    own name. The base's identity is in the name itself (``<label>@<tree12>``), not in the store."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        (self.root / "patches").mkdir(parents=True, exist_ok=True)

    def _path(self, code_hash: str) -> Path:
        return self.root / "patches" / f"{code_hash}.patch"

    def _write(self, path: Path, data: bytes) -> None:
        fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            os.replace(tmp, path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def put(self, code_hash: str, cum_diff: bytes) -> None:
        if split_code_name(code_hash)[2] is None or code_name(base_of(code_hash), cum_diff) != code_hash:
            raise WorkspaceError(f"diff does not hash to {code_hash!r}")
        path = self._path(code_hash)
        if path.exists():
            if path.read_bytes() != cum_diff:
                raise WorkspaceError(f"store already holds a different diff for {code_hash!r} (hash collision)")
            return
        self._write(path, cum_diff)

    def get(self, code_hash: str) -> bytes:
        path = self._path(code_hash)
        if not path.is_file():
            raise WorkspaceError(f"code {code_hash!r} is not in the store {self.root}")
        diff = path.read_bytes()
        if code_name(base_of(code_hash), diff) != code_hash:
            raise WorkspaceError(f"store file for {code_hash!r} does not match its name (corrupt store)")
        return diff


# --------------------------------------------------------------------------- #
# manager
# --------------------------------------------------------------------------- #

class WorkspaceManager:
    """``bases``: label -> checkout directory (read only). ``store``: the patch store directory
    (created if missing; always written). Name a checkout's current content with
    ``base_ref(label)``. ``config_loader``: working copy -> eval config (default: ``config/eval.yaml``)."""

    def __init__(self, store: str | Path, bases: Mapping[str, str | Path],
                 config_loader: ConfigLoader = load_eval_config):
        self.config_loader = config_loader
        self.store = PatchStore(store)
        self.bases = {label: Path(p).resolve() for label, p in bases.items()}
        for label, p in self.bases.items():
            if not _LABEL_RE.match(label):
                raise WorkspaceError(f"invalid base label {label!r} (letters, digits, '.', '_', '-')")
            if not p.is_dir():
                raise WorkspaceError(f"base checkout {label!r}: {p} is not a directory")
        self._views: dict[str, Path] = {}
        self._scratch: Optional[Path] = None
        self._slots = 0

    # -- names ---------------------------------------------------------------
    def _parts(self, code_hash: str) -> tuple[str, str, Optional[str]]:
        label, tree, digest = split_code_name(code_hash)
        if label not in self.bases:
            raise WorkspaceError(f"unknown base {label!r} for code {code_hash!r}; registered: {sorted(self.bases)}")
        return label, tree, digest

    def _diff(self, code_hash: str) -> bytes:
        return b"" if self._parts(code_hash)[2] is None else self.store.get(code_hash)

    def base_ref(self, label: str, eval_config: Optional[Mapping] = None) -> CodeRef:
        """The checkout ``label`` as it is now: ``label@<tree12>``, with ``eval_config`` if given,
        else the config the loader reads from the checkout."""
        if label not in self.bases:
            raise WorkspaceError(f"unknown base {label!r}; registered: {sorted(self.bases)}")
        with tempfile.TemporaryDirectory(prefix="ws-base-") as td:
            tree = self._snapshot(label, Path(td) / "repo")
            config = dict(self.config_loader(Path(td) / "repo") if eval_config is None else eval_config)
        return CodeRef(f"{label}@{tree[:12]}", config)

    def base_name(self, code_ref: CodeRef) -> str:
        """The base (``label@tree12``) that ``code_ref`` is built on."""
        label, tree, _ = self._parts(code_ref.code_hash)
        return f"{label}@{tree}"

    def diff_of(self, code_ref: CodeRef) -> str:
        """The cumulative diff of ``code_ref`` against its base ("" for the base itself), as the
        lossless ``str`` view."""
        return as_text(self._diff(code_ref.code_hash))

    def register(self, code_ref: CodeRef, cum_diff: Diff) -> CodeRef:
        """Add a known cumulative diff (e.g. ``LoopResult.final_diff``) to the store. The name must
        match the diff."""
        self._parts(code_ref.code_hash)
        data = as_bytes(cum_diff)
        if code_name(base_of(code_ref.code_hash), data) != code_ref.code_hash:
            raise WorkspaceError(f"diff does not hash to {code_ref.code_hash!r}")
        if data.strip():
            self.store.put(code_ref.code_hash, data)
        return code_ref

    # -- working copies --------------------------------------------------------
    def _copy_base(self, src: Path, dest: Path) -> None:
        """The one place a working copy's files come from: the checkout without ``.git`` (at any
        depth). Which of them are kept is ``_snapshot``'s job. F5 (leakage) adds its filters here."""
        shutil.copytree(src, dest, symlinks=True, ignore=shutil.ignore_patterns(".git"), dirs_exist_ok=True)

    def _snapshot(self, label: str, dest: Path) -> str:
        """Copy ``label``'s checkout into ``dest`` and commit it as one snapshot; return its tree id.
        Kept: every file the original tracks (``git add -f``, even if ignored) and every untracked
        file no ``.gitignore`` matches. Dropped: untracked ignored files (``git clean -X``)."""
        src = self.bases[label]
        tracked = _tracked_files(src)
        self._copy_base(src, dest)
        _git(dest, "init", "-q", "--template=", "-b", "vla")
        _git(dest, "add", "-A")
        present = [t for t in tracked if os.path.lexists(dest / os.fsdecode(t))]
        if present:
            _git(dest, "add", "-f", "--pathspec-from-file=-", "--pathspec-file-nul", input=b"\0".join(present),
                 env={"GIT_LITERAL_PATHSPECS": "1"})
        _git(dest, "clean", "-f", "-d", "-X", "-q")
        bad = _escaping_symlinks(dest)
        if bad:
            raise WorkspaceError(f"base checkout {label!r} has symlinks that are absolute or point outside it "
                                 f"(rejected: a working copy could be written outside itself through them): "
                                 f"{', '.join(bad[:5])}")
        _git(dest, "commit", "-q", "--no-verify", "--allow-empty", "--no-gpg-sign", "-m", f"vla: base {label}")
        return _out(dest, "rev-parse", "HEAD^{tree}")

    @staticmethod
    def _apply(dest: Path, patch: bytes) -> None:
        p = _git(dest, "apply", "--whitespace=nowarn", "-", check=False,
                 input=patch if patch.endswith(b"\n") else patch + b"\n")
        if p.returncode != 0:
            raise UnrebuildablePatch(_cause(patch, p.stderr), _show(p.stderr)[:1000] or "git apply failed")
        bad = _escaping_symlinks(dest)
        if bad:
            raise UnrebuildablePatch("symlink", f"the patch adds symlinks that are absolute or point outside the "
                                                f"tree: {', '.join(bad[:5])}")

    def materialize(self, code_ref: CodeRef, dest: Optional[str | Path] = None) -> Path:
        """Build ``code_ref`` in a fresh directory (``dest``, which must not exist or be empty;
        default a new temp dir the caller owns) and return its path. On failure nothing is left:
        a directory it created is removed, a given empty one is emptied again."""
        name = code_ref.code_hash
        label, tree, _ = self._parts(name)
        diff = self._diff(name)
        if dest is not None and Path(dest).exists() and any(Path(dest).iterdir()):
            raise WorkspaceError(f"{dest} is not empty")
        created = dest is None or not Path(dest).exists()
        dest = Path(tempfile.mkdtemp(prefix="ws-")) if dest is None else Path(dest)
        try:
            dest.mkdir(parents=True, exist_ok=True)
            now = self._snapshot(label, dest)
            if now[:12] != tree:
                raise WorkspaceError(
                    f"base checkout {label!r} changed: {label}@{tree} was snapshotted as tree {tree}, the checkout "
                    f"is now tree {now[:12]}. Code named from the old content cannot be rebuilt; name the current "
                    f"content with base_ref({label!r})")
            _git(dest, "update-ref", BASE_REF, "HEAD")
            _git(dest, "config", "vla.base", f"{label}@{tree}")
            if diff:
                try:
                    self._apply(dest, diff)
                except UnrebuildablePatch as e:
                    raise WorkspaceError(f"stored diff of {name!r} does not apply: {e}") from e
                _commit(dest, f"vla: code {name}")
        except BaseException:
            if created:
                shutil.rmtree(dest, ignore_errors=True)
            else:
                for child in list(dest.iterdir()):
                    shutil.rmtree(child) if child.is_dir() and not child.is_symlink() else child.unlink()
            raise
        return dest

    def extend(self, code_ref: CodeRef, patch: Diff) -> CodeRef:
        """The code you get by applying ``patch`` (one attempt's diff, written against
        ``code_ref``'s files) to ``code_ref``. The new name comes from diffing the resulting files
        against the base snapshot. Raises ``UnrebuildablePatch`` if the patch cannot be rebuilt,
        including when the patched files' eval config cannot be read (``config_unreadable``).

        The eval config: an explicit one on ``code_ref`` (``base_ref(label, eval_config)``) is kept
        until a patch changes the config the loader reads versus the base snapshot; from then on
        the patched files' config is the config."""
        data = as_bytes(patch)
        if not data.strip():
            return code_ref
        with tempfile.TemporaryDirectory(prefix="ws-extend-") as td:
            work = self.materialize(code_ref, Path(td) / "repo")
            self._apply(work, data)
            try:
                config = dict(self.config_loader(work))
                if config == self._base_config(work, Path(td) / "base"):
                    config = dict(code_ref.eval_config)
            except WorkspaceError as e:
                raise UnrebuildablePatch("config_unreadable", f"the patched files' eval config cannot be read: {e}") from e
            name, diff = _named(work)
            return self.register(CodeRef(name, config), diff)

    def _base_config(self, work: Path, dest: Path) -> dict:
        """The eval config of ``work``'s base snapshot (its ``BASE_REF`` commit), read by the loader."""
        idx = dest.parent / "base.index"
        env = {"GIT_INDEX_FILE": str(idx)}
        _git(work, "read-tree", BASE_REF, env=env)
        _git(work, "checkout-index", "-a", f"--prefix={dest}/", env=env)
        idx.unlink(missing_ok=True)
        return dict(self.config_loader(dest))

    def adopt(self, workdir: str | Path) -> CodeRef:
        """Name the code in a working copy (edits included), read its eval config from the copy,
        and store its cumulative diff."""
        name, diff = _named(Path(workdir))
        return self.register(CodeRef(name, dict(self.config_loader(Path(workdir)))), diff)

    def view(self, code_ref: CodeRef) -> Path:
        """A cached working copy of ``code_ref`` for readers. Do not write to it. A failed build
        keeps nothing, so a retry fails with the same error."""
        name = code_ref.code_hash
        if name not in self._views:
            if self._scratch is None:
                self._scratch = Path(tempfile.mkdtemp(prefix="ws-views-"))
                weakref.finalize(self, shutil.rmtree, str(self._scratch), True)
            self._slots += 1
            self._views[name] = self.materialize(code_ref, self._scratch / f"v{self._slots}")
        return self._views[name]


__all__ = ["BASE_REF", "ConfigLoader", "EVAL_CONFIG_PATH", "FileContains", "PatchStore", "UnrebuildablePatch", "WorkspaceError", "WorkspaceManager",
           "as_bytes", "as_text", "base_of", "code_name", "code_ref_of", "cumulative_diff", "file_config_loader", "load_eval_config",
           "split_code_name"]
