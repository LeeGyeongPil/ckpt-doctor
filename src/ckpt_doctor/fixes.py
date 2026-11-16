"""Build the command that repairs a finding. All command templates live here."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Any

FILE_CODES = {
    "CD102", "CD103", "CD104", "CD105", "CD106", "CD107", "CD108", "CD109",
    "CD110", "CD111", "CD112", "CD114", "CD202", "CD402", "CD404", "CD405", "CD501", "CD503",
}  # fmt: skip


@dataclass(frozen=True)
class FixContext:
    folder: str  # folder as the user typed it (used for --local-dir)
    repo_id: str | None
    in_cache: bool
    cache_dir: str | None = None  # --cache-dir as the user typed it
    revision: str | None = None  # revision being checked (None or "main" = default)


def _repo(ctx: FixContext) -> str:
    return ctx.repo_id or "<repo_id>"


def _cache_args(ctx: FixContext, with_revision: bool) -> list[str]:
    """Options that make an `hf` command act on the cache/revision that was checked."""
    parts: list[str] = []
    if ctx.cache_dir:
        parts += ["--cache-dir", shlex.quote(ctx.cache_dir)]
    if with_revision and ctx.revision and ctx.revision != "main":
        parts += ["--revision", shlex.quote(ctx.revision)]
    return parts


def _download(ctx: FixContext, files: list[str], force: bool = True) -> str:
    parts = ["hf", "download", _repo(ctx), *[shlex.quote(f) for f in files]]
    if not ctx.in_cache:
        parts += ["--local-dir", shlex.quote(ctx.folder)]
    else:
        parts += _cache_args(ctx, with_revision=True)
    if force:
        parts.append("--force-download")
    return " ".join(parts)


def fix_for(code: str, rel: str | None, params: dict[str, Any], ctx: FixContext) -> str | None:
    if code == "CD101" and rel:
        return f'git lfs pull --include "{rel}"'
    if code in FILE_CODES and rel:
        return _download(ctx, [rel], force=code != "CD202")
    if code == "CD201":
        return _download(ctx, list(params.get("files", [])), force=False)
    if code == "CD401":
        return " ".join(["hf cache prune --dry-run", *_cache_args(ctx, with_revision=False)])
    if code == "CD403":
        parts = ["hf", "download", _repo(ctx), "--revision", shlex.quote(str(params["commit"]))]
        return " ".join([*parts, *_cache_args(ctx, with_revision=False)])
    return None
