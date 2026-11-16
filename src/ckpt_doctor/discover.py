"""Turn the user's PATH argument into a concrete thing to check."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from . import hfcache
from .model import CkptDoctorUsageError


@dataclass
class Target:
    kind: str  # "dir" | "file" | "index" | "cache"
    root: Path  # folder holding the model files (snapshot folder for caches)
    base: Path  # finding paths are shown relative to this
    display: str  # as the user typed it
    only: Path | None = None  # single .safetensors or index file
    repo: hfcache.CacheRepo | None = None
    revision: str | None = None
    repo_id: str | None = None


def resolve_target(
    path: str,
    revision: str | None = None,
    cache_dir: str | None = None,
    repo_id: str | None = None,
    env: dict[str, str] | None = None,
) -> Target:
    p = Path(path).expanduser()
    if p.is_file():
        if p.name.endswith(".safetensors.index.json"):
            return Target("index", p.parent, p.parent, path, only=p, repo_id=repo_id)
        if p.name.endswith(".safetensors"):
            return Target("file", p.parent, p.parent, path, only=p, repo_id=repo_id)
        raise CkptDoctorUsageError("err.not_checkable", path=path)
    if p.is_dir():
        rp = p.resolve()
        if hfcache.is_cache_repo_dir(rp):
            return _cache_target(rp, path, revision)
        if rp.parent.name == "snapshots" and hfcache.is_cache_repo_dir(rp.parent.parent):
            return _cache_target(rp.parent.parent, path, rp.name)
        return Target("dir", p, p, path, repo_id=repo_id)
    if hfcache.looks_like_repo_id(path) and not os.path.isabs(path):
        cache = Path(cache_dir).expanduser() if cache_dir else hfcache.default_cache_dir(env)
        repo_dir = cache / hfcache.repo_dir_name(path)
        if not hfcache.is_cache_repo_dir(repo_dir):
            raise CkptDoctorUsageError("err.repo_not_cached", repo_id=path, cache=str(cache))
        return _cache_target(repo_dir, path, revision)
    raise CkptDoctorUsageError("err.path_not_found", path=path)


def _cache_target(repo_dir: Path, display: str, revision: str | None) -> Target:
    repo_id = hfcache.repo_id_from_dir(repo_dir.name) or repo_dir.name
    repo = hfcache.CacheRepo(repo_dir=repo_dir, repo_id=repo_id)
    repo.commit = hfcache.select_commit(repo, revision)
    snap = repo_dir / "snapshots" / repo.commit
    repo.snapshot = snap if snap.is_dir() else None
    return Target(
        "cache", repo.snapshot or repo_dir, repo_dir, display, repo=repo, revision=revision, repo_id=repo_id
    )
