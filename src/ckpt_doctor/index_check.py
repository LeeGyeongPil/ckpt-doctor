"""Cross-check ``*.safetensors.index.json`` against the shard headers (no tensor data is read)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Callable

from .safetensors_header import HeaderInfo

Emit = Callable[..., None]

SHARD_RE = re.compile(r"^(?P<prefix>.+)-(?P<idx>\d{5})-of-(?P<total>\d{5})\.safetensors$")


@dataclass
class IndexResult:
    path: Path
    ok: bool = True  # False when the index itself could not be used
    names: set[str] = field(default_factory=set)  # tensor names according to the index
    referenced: set[Path] = field(default_factory=set)


def _unsafe(name: str) -> bool:
    p = PurePosixPath(name.replace("\\", "/"))
    return p.is_absolute() or ".." in p.parts or re.match(r"^[A-Za-z]:", name) is not None


def check_index(
    index_path: Path,
    root: Path,
    headers: dict[Path, HeaderInfo | None],
    emit: Emit,
    broken_links: set[Path],
) -> IndexResult:
    res = IndexResult(path=index_path)
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        emit("CD211", {"detail": str(exc)}, path=index_path)
        res.ok = False
        return res
    weight_map = data.get("weight_map") if isinstance(data, dict) else None
    if not isinstance(weight_map, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in weight_map.items()
    ):
        emit("CD211", {"detail": "weight_map missing or not a string-to-string map"}, path=index_path)
        res.ok = False
        return res

    res.names = set(weight_map)
    safe: dict[str, Path] = {}
    for shard in dict.fromkeys(weight_map.values()):
        if _unsafe(shard):
            emit("CD210", {"shard": shard}, path=index_path)
            continue
        safe[shard] = root / shard

    unusable: set[str] = set()  # shards whose tensors we cannot compare against
    for shard, p in safe.items():
        res.referenced.add(p)
        if not p.exists():
            unusable.add(shard)
            if p not in broken_links:  # a broken cache symlink is reported once, as CD402
                emit("CD202", {"shard": shard}, path=p)
        elif headers.get(p) is None:
            unusable.add(shard)

    usable = {s: headers[p] for s, p in safe.items() if s not in unusable and (headers.get(p) is not None)}
    holders: dict[str, list[str]] = {}
    for shard, info in usable.items():
        for name in info.tensors:
            holders.setdefault(name, []).append(shard)

    clean_tensors = True
    reported_dups: set[str] = set()
    for name, shard in weight_map.items():
        if shard in unusable or shard not in safe:
            continue
        where = holders.get(name, [])
        if len(where) > 1:
            if name not in reported_dups:
                reported_dups.add(name)
                emit("CD206", {"tensor": name, "shards": ", ".join(where)}, path=index_path, tensor=name)
            clean_tensors = False
        elif shard in where:
            continue
        elif where:
            emit("CD205", {"tensor": name, "mapped": shard, "actual": where[0]}, path=safe[shard], tensor=name)
            clean_tensors = False
        else:
            emit("CD203", {"tensor": name, "shard": shard}, path=safe[shard], tensor=name)
            clean_tensors = False

    for shard, info in usable.items():
        for name in info.tensors:
            if name not in weight_map:
                emit("CD204", {"tensor": name, "shard": shard}, path=safe[shard], tensor=name)
                clean_tensors = False

    metadata = data.get("metadata")
    total = metadata.get("total_size") if isinstance(metadata, dict) else None
    if not isinstance(total, int) or isinstance(total, bool):
        emit("CD208", {}, path=index_path)
    elif clean_tensors and not unusable and len(usable) == len(safe):
        actual = 0
        for name, shard in weight_map.items():
            entry = usable[shard].tensors.get(name) if shard in usable else None
            if entry is None or entry.start is None or entry.end is None:
                return res  # a damaged entry is already reported; do not add noise
            actual += entry.end - entry.start
        if actual != total:
            emit("CD207", {"declared": total, "actual": actual}, path=index_path)
    return res


def numbered_groups(files: list[Path]) -> dict[tuple[str, int], set[int]]:
    """Group ``<prefix>-0000k-of-0000n.safetensors`` files: (prefix, n) -> set of present k."""
    groups: dict[tuple[str, int], set[int]] = {}
    for f in files:
        m = SHARD_RE.match(f.name)
        if m:
            groups.setdefault((m["prefix"], int(m["total"])), set()).add(int(m["idx"]))
    return groups


def is_numbered_shard(name: str) -> bool:
    return SHARD_RE.match(name) is not None


def shard_name(prefix: str, idx: int, total: int) -> str:
    return f"{prefix}-{idx:05d}-of-{total:05d}.safetensors"
