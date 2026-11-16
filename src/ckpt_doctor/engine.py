"""Orchestration: discover -> headers -> index -> config -> cache -> hash -> fixes."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable

from . import hfcache
from .codes import LAYERS, default_severity, layer_of
from .config_check import check_config
from .discover import Target, resolve_target
from .fixes import FixContext, fix_for
from .hashing import read_manifest, sha256_file, write_manifest
from .index_check import IndexResult, check_index, is_numbered_shard, numbered_groups, shard_name
from .model import Finding, Report, Severity
from .safetensors_header import HeaderInfo, TensorEntry, read_header

Progress = Callable[[str, int, int], None]


class _Collector:
    def __init__(self, target: Target, ctx: FixContext) -> None:
        self.target = target
        self.ctx = ctx
        self.findings: list[Finding] = []

    def _display(self, p: Path | None) -> str | None:
        if p is None:
            return None
        try:
            return p.relative_to(self.target.base).as_posix() or "."
        except ValueError:
            return p.as_posix()

    def _rel_to_root(self, p: Path | None) -> str | None:
        if p is None:
            return None
        try:
            return p.relative_to(self.target.root).as_posix()
        except ValueError:
            return None

    def emit(
        self,
        code: str,
        params: dict[str, Any] | None = None,
        *,
        path: Path | None = None,
        tensor: str | None = None,
        severity: Severity | None = None,
    ) -> None:
        params = params or {}
        fix = fix_for(code, self._rel_to_root(path), params, self.ctx)
        self.findings.append(
            Finding(
                code=code,
                severity=severity or default_severity(code),
                path=self._display(path),
                message_key=f"{code}.msg",
                params=params,
                tensor=tensor,
                fix=fix,
            )
        )


def check_path(
    path: str | os.PathLike[str],
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
    repo_id: str | None = None,
    hash_files: bool = False,
    manifest: str | os.PathLike[str] | None = None,
    write_manifest_to: str | os.PathLike[str] | None = None,
    progress: Progress | None = None,
    env: dict[str, str] | None = None,
) -> Report:
    """Check a model folder, a single .safetensors / index file, or a cached Hugging Face repo."""
    started = time.perf_counter()
    target = resolve_target(os.fspath(path), revision, cache_dir, repo_id, env)
    manifest_entries = read_manifest(Path(manifest)) if manifest else None
    # --local-dir must be the folder holding the file, not the file/index path itself
    fix_folder = target.display
    if target.kind in ("file", "index"):
        fix_folder = os.path.dirname(target.display) or "."
    ctx = FixContext(
        folder=fix_folder, repo_id=target.repo_id, in_cache=target.kind == "cache"
    )
    col = _Collector(target, ctx)
    report = Report(target=target.display)
    stats = report.stats
    root = target.root

    # ---- which files belong to this check -------------------------------------------------
    broken_links: set[Path] = set()
    if root.is_dir():
        broken_links = {p for p in root.iterdir() if p.is_symlink() and not p.exists()}
    if target.kind == "file":
        assert target.only is not None
        files = [target.only]
    elif target.kind == "index":
        files = _files_for_index(target)
    elif target.kind == "cache" and target.repo is not None and target.repo.snapshot is None:
        files = []
    else:
        files = sorted(p for p in root.iterdir() if p.suffix == ".safetensors" and p.exists())

    # ---- cache layer ------------------------------------------------------------------------
    trees: dict[str, dict[str, Any]] = {}
    if target.kind == "cache" and target.repo is not None:
        repo = target.repo
        assert repo.commit is not None
        hfcache.scan_repo(repo, repo.commit, col.emit)
        trees = hfcache.load_trees(repo.repo_dir, repo.commit, col.emit)
        if repo.snapshot is not None:
            hfcache.check_trees(repo.snapshot, trees, col.emit, report_missing=True)
    elif target.kind == "dir":
        trees = hfcache.local_dir_trees(root, col.emit)
        hfcache.check_trees(root, trees, col.emit, report_missing=False)

    # ---- file layer -------------------------------------------------------------------------
    headers: dict[Path, HeaderInfo | None] = {}
    file_errors: set[Path] = set()
    for f in files:
        def emit_file(code: str, params: dict[str, Any] | None = None, tensor: str | None = None, f: Path = f) -> None:
            if default_severity(code) == Severity.ERROR:
                file_errors.add(f)
            col.emit(code, params, path=f, tensor=tensor)

        info, nread = read_header(f, emit_file)
        headers[f] = info
        stats.bytes_read += nread
        stats.files_checked += 1

    # ---- index layer ------------------------------------------------------------------------
    results: list[IndexResult] = []
    referenced: set[Path] = set()
    has_index_problem = False
    if target.kind in ("dir", "cache", "index"):
        index_files = [target.only] if target.kind == "index" and target.only else sorted(
            root.glob("*.safetensors.index.json")
        )
        for ipath in index_files:
            stats.bytes_read += ipath.stat().st_size
            res = check_index(ipath, root, headers, col.emit, broken_links)
            results.append(res)
            referenced |= res.referenced
            has_index_problem |= not res.ok
        if target.kind != "index":
            if results and all(r.ok for r in results):
                for f in files:
                    if f not in referenced:
                        col.emit("CD209", {"file": f.name}, path=f)
            for (prefix, total), present in sorted(numbered_groups(files).items()):
                group = {root / shard_name(prefix, i, total) for i in present}
                if group <= referenced or (root / f"{prefix}.safetensors.index.json").exists():
                    continue
                missing = [shard_name(prefix, i, total) for i in range(1, total + 1) if i not in present]
                col.emit(
                    "CD201",
                    {
                        "found": len(present), "total": total, "prefix": prefix,
                        "files": [*missing, f"{prefix}.safetensors.index.json"],
                    },
                    path=root,
                )  # fmt: skip
                has_index_problem = True

    # ---- config layer -----------------------------------------------------------------------
    names: set[str] = set()
    tensors: dict[str, TensorEntry] = {}
    for info in headers.values():
        if info is not None:
            names |= set(info.tensors)
            tensors.update(info.tensors)
    for res in results:
        names |= res.names
    index_ok_files = {p for r in results if r.ok for p in r.referenced}
    unreadable = [f for f, info in headers.items() if info is None and f not in index_ok_files]
    complete = not unreadable and not has_index_problem
    if target.kind == "file" and is_numbered_shard(target.only.name if target.only else ""):
        complete = False
    if files:
        check_config(
            root / "config.json",
            names,
            tensors,
            complete=complete,
            require_config=target.kind != "file",
            emit=col.emit,
        )
    elif not broken_links and not (target.kind == "cache" and target.repo and target.repo.snapshot is None):
        col.emit("CD001", {})

    # ---- hash layer -------------------------------------------------------------------------
    manifest_written = False
    if hash_files or write_manifest_to:
        digests: dict[str, str] = {}
        todo = [f for f in files if f not in file_errors and f.exists()]
        for i, f in enumerate(todo, 1):
            rel = f.relative_to(root).as_posix()
            baseline = _baseline(rel, f, manifest_entries, trees) if hash_files else None
            if baseline is None and not write_manifest_to:
                col.emit("CD502", {"file": rel}, path=f)
                continue
            if progress:
                progress(rel, i, len(todo))
            digest = sha256_file(f, _count_hashed(stats))
            digests[rel] = digest
            if baseline is None:
                if hash_files:
                    col.emit("CD502", {"file": rel}, path=f)
            elif baseline[1] != digest:
                col.emit(
                    "CD501",
                    {
                        "expected": baseline[1], "actual": digest, "source": baseline[0],
                        "expected_short": baseline[1][:12], "actual_short": digest[:12],
                    },
                    path=f,
                )
        if hash_files and manifest_entries is not None:
            on_disk = {f.relative_to(root).as_posix() for f in files}
            for name in sorted(set(manifest_entries) - on_disk):
                col.emit("CD503", {"file": name}, path=root / name)
        if write_manifest_to and not any(f.severity == Severity.ERROR for f in col.findings):
            write_manifest(Path(write_manifest_to), digests)
            manifest_written = True

    # ---- finish -----------------------------------------------------------------------------
    col.findings.sort(
        key=lambda f: (LAYERS.index(layer_of(f.code)), f.path or "", f.code, f.tensor or "")
    )
    report.findings = col.findings
    stats.elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    report.context = {
        "kind": target.kind,
        "repo_id": target.repo_id,
        "commit": target.repo.commit if target.repo else None,
        "base": str(target.base),
        "manifest_written": manifest_written,
        "manifest_requested": bool(write_manifest_to),
    }
    return report


def _count_hashed(stats: Any) -> Callable[[int], None]:
    def add(n: int) -> None:
        stats.bytes_hashed += n

    return add


def _baseline(
    rel: str, f: Path, manifest: dict[str, str] | None, trees: dict[str, dict[str, Any]]
) -> tuple[str, str] | None:
    """Pick the expected sha256: manifest > tree listing > cache blob name."""
    if manifest and rel in manifest:
        return "manifest", manifest[rel]
    sha = (trees.get(rel) or {}).get("lfs_sha256")
    if isinstance(sha, str) and len(sha) == 64:
        return "trees", sha.lower()
    blob = hfcache.blob_name_sha(f)
    if blob:
        return "blob", blob
    return None


def _files_for_index(target: Target) -> list[Path]:
    assert target.only is not None
    try:
        weight_map = json.loads(target.only.read_text(encoding="utf-8")).get("weight_map", {})
        shards = sorted({v for v in weight_map.values() if isinstance(v, str)})
    except (OSError, ValueError, AttributeError):
        shards = []
    out = []
    for s in shards:
        p = target.root / s
        if ".." not in Path(s).parts and not Path(s).is_absolute() and p.exists():
            out.append(p)
    return out

