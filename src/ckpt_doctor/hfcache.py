"""Read-only inspection of a Hugging Face hub cache (``models--org--name``). No network access."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .model import CkptDoctorUsageError, Severity

Emit = Callable[..., None]

TREES_FORMAT_VERSION = 1
REPO_ID_RE = re.compile(r"^[A-Za-z0-9][\w.-]*(/[\w.-]+)?$")


@dataclass
class CacheRepo:
    repo_dir: Path
    repo_id: str
    commit: str | None = None
    snapshot: Path | None = None
    trees: dict[str, dict[str, Any]] = field(default_factory=dict)


def default_cache_dir(env: dict[str, str] | None = None) -> Path:
    env = os.environ if env is None else env
    if env.get("HF_HUB_CACHE"):
        return Path(env["HF_HUB_CACHE"]).expanduser()
    if env.get("HF_HOME"):
        return Path(env["HF_HOME"]).expanduser() / "hub"
    return Path("~/.cache/huggingface/hub").expanduser()


def looks_like_repo_id(text: str) -> bool:
    return bool(REPO_ID_RE.match(text))


def repo_dir_name(repo_id: str) -> str:
    return "models--" + repo_id.replace("/", "--")


def repo_id_from_dir(name: str) -> str | None:
    if not name.startswith("models--"):
        return None
    return name[len("models--") :].replace("--", "/")


def is_cache_repo_dir(p: Path) -> bool:
    return p.name.startswith("models--") and (p / "snapshots").is_dir()


def read_ref(ref_file: Path) -> str | None:
    try:
        text = ref_file.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text or None


def select_commit(repo: CacheRepo, revision: str | None) -> str:
    """Pick the commit to inspect from ``refs/<revision>`` (default ``main``) or a commit hash."""
    wanted = revision or "main"
    ref = read_ref(repo.repo_dir / "refs" / wanted)
    if ref:
        return ref
    if (repo.repo_dir / "snapshots" / wanted).is_dir():
        return wanted
    snaps = [d.name for d in (repo.repo_dir / "snapshots").iterdir() if d.is_dir()]
    if revision is None and len(snaps) == 1:
        return snaps[0]
    raise CkptDoctorUsageError("err.revision_not_found", revision=wanted, repo_id=repo.repo_id)


def load_trees(folder: Path, commit: str | None, emit: Emit) -> dict[str, dict[str, Any]]:
    """Read ``<folder>/trees/<commit>.json`` (huggingface_hub >= 2 writes these). Empty if absent."""
    if commit is None:
        return {}
    path = folder / "trees" / f"{commit}.json"
    return parse_trees(path, emit)


def parse_trees(path: Path, emit: Emit) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        files = data["files"]
        version = data.get("format_version")
        if not isinstance(files, dict):
            raise ValueError("'files' is not an object")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        emit("CD407", {"detail": str(exc)}, path=path)
        return {}
    if version != TREES_FORMAT_VERSION:
        emit("CD406", {"version": version}, path=path)
        return {}
    return {k: v for k, v in files.items() if isinstance(v, dict)}


def scan_repo(repo: CacheRepo, selected_commit: str, emit: Emit) -> None:
    """Cache-level findings: dangling refs, ``.incomplete`` blobs, broken symlinks."""
    snaps = repo.repo_dir / "snapshots"
    refs = repo.repo_dir / "refs"
    if refs.is_dir():
        for ref_file in sorted(p for p in refs.rglob("*") if p.is_file()):
            commit = read_ref(ref_file)
            if commit and not (snaps / commit).is_dir():
                selected = commit == selected_commit
                emit(
                    "CD403",
                    {"ref": ref_file.relative_to(refs).as_posix(), "commit": commit},
                    path=ref_file,
                    severity=Severity.ERROR if selected else Severity.WARNING,
                )
    blobs = repo.repo_dir / "blobs"
    if blobs.is_dir():
        for inc in sorted(blobs.rglob("*.incomplete")):
            emit("CD401", {"name": inc.name, "size": inc.stat().st_size}, path=inc)
    if repo.snapshot is not None:
        for broken in find_broken_links(repo.snapshot):
            emit("CD402", {"file": broken.relative_to(repo.snapshot).as_posix()}, path=broken)


def find_broken_links(root: Path) -> list[Path]:
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        for name in sorted(dirnames + filenames):
            p = Path(dirpath, name)
            if p.is_symlink() and not p.exists():
                out.append(p)
    return sorted(set(out))


def check_trees(
    root: Path,
    trees: dict[str, dict[str, Any]],
    emit: Emit,
    report_missing: bool,
) -> None:
    """Compare file sizes against the tree listing recorded at download time."""
    for rel, ent in sorted(trees.items()):
        p = root / rel
        if not os.path.lexists(p):
            if report_missing:
                emit("CD404", {"file": rel}, path=p)
            continue
        if not p.exists():
            continue  # broken symlink, reported as CD402
        expected = ent.get("lfs_size") if isinstance(ent.get("lfs_size"), int) else ent.get("size")
        if isinstance(expected, int) and p.is_file() and p.stat().st_size != expected:
            emit("CD405", {"file": rel, "expected": expected, "actual": p.stat().st_size}, path=p)


def blob_name_sha(path: Path) -> str | None:
    """First symlink hop's file name if it is a 64-hex sha256 (the hub names LFS blobs by sha256)."""
    if not path.is_symlink():
        return None
    name = os.path.basename(os.readlink(path))
    return name.lower() if re.fullmatch(r"[0-9a-fA-F]{64}", name) else None


def local_dir_trees(root: Path, emit: Emit) -> dict[str, dict[str, Any]]:
    """For ``hf download --local-dir`` folders: newest ``.cache/huggingface/trees/*.json``."""
    folder = root / ".cache" / "huggingface" / "trees"
    if not folder.is_dir():
        return {}
    candidates = sorted(folder.glob("*.json"), key=lambda p: p.stat().st_mtime)
    return parse_trees(candidates[-1], emit) if candidates else {}
