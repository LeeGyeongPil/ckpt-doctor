"""Read and validate a .safetensors header without touching the tensor data.

Layout (https://github.com/huggingface/safetensors): ``u64 little-endian N`` + ``N bytes of JSON`` +
data region. Only the first 8 bytes and the JSON header are ever read here.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .lfs import parse_pointer

MAX_HEADER = 100_000_000  # the official limit: 100 MB
HEAD_PROBE = 512  # bytes read first to tell pointer / HTML / real header apart

DTYPE_BITS = {
    "BOOL": 8, "U8": 8, "I8": 8, "F8_E5M2": 8, "F8_E4M3": 8, "F8_E8M0": 8,
    "F4": 4, "F6_E2M3": 6, "F6_E3M2": 6,
    "I16": 16, "U16": 16, "F16": 16, "BF16": 16,
    "I32": 32, "U32": 32, "F32": 32,
    "I64": 64, "U64": 64, "F64": 64,
}  # fmt: skip

HTML_STARTS = (b"<!doctype", b"<html", b"<?xml", b"<head", b"<body", b"<!--")

# emit(code, params, tensor=None) -> None
Emit = Callable[..., None]


@dataclass
class TensorEntry:
    name: str
    dtype: str | None
    shape: list[int] | None
    start: int | None
    end: int | None


@dataclass
class HeaderInfo:
    path: Path
    file_size: int
    header_len: int
    tensors: dict[str, TensorEntry] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def data_len(self) -> int:
        return self.file_size - 8 - self.header_len


class _DuplicateKey(Exception):
    def __init__(self, key: str) -> None:
        self.key = key


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise _DuplicateKey(key)
        out[key] = value
    return out


def human_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1000 or unit == "TB":
            return f"{int(size)} B" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1000
    return f"{n} B"


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def read_header(path: Path, emit: Emit) -> tuple[HeaderInfo | None, int]:
    """Validate one file. Returns (header or None if unusable, bytes read). Problems go to ``emit``."""
    file_size = os.stat(path).st_size
    bytes_read = 0
    if file_size < 8:
        emit("CD103", {"size": file_size})
        return None, 0
    with open(path, "rb") as fh:
        head = fh.read(min(file_size, HEAD_PROBE))
        bytes_read += len(head)

        pointer = parse_pointer(head)
        if pointer is not None:
            expected = human_bytes(pointer.size) if pointer.size is not None else "?"
            if pointer.oid:
                expected += f", oid sha256:{pointer.oid[:12]}..."
            emit("CD101", {"size": file_size, "expected": expected})
            return None, bytes_read
        if head.lstrip().lower().startswith(HTML_STARTS):
            emit("CD102", {"size": file_size})
            return None, bytes_read

        n = int.from_bytes(head[:8], "little")
        if n == 0:
            emit("CD103", {"size": file_size})
            return None, bytes_read
        if n > MAX_HEADER:
            emit("CD104", {"length": n, "limit": MAX_HEADER})
            return None, bytes_read
        if 8 + n > file_size:
            emit("CD105", {"length": n, "available": file_size - 8})
            return None, bytes_read

        if 8 + n <= len(head):
            raw = head[8 : 8 + n]
        else:
            rest = fh.read(8 + n - len(head))
            bytes_read += len(rest)
            raw = head[8:] + rest

    parsed = _parse_json(raw, emit)
    if parsed is None:
        return None, bytes_read
    info = HeaderInfo(path=path, file_size=file_size, header_len=n)
    _check_entries(info, parsed, emit)
    return info, bytes_read


def _parse_json(raw: bytes, emit: Emit) -> dict[str, Any] | None:
    if not raw.startswith(b"{"):
        emit("CD107", {"detail": "header does not start with '{'"})
        return None
    try:
        parsed = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicates)
    except _DuplicateKey as exc:
        emit("CD106", {"key": exc.key})
        return None
    except (UnicodeDecodeError, ValueError) as exc:
        emit("CD107", {"detail": str(exc)})
        return None
    if not isinstance(parsed, dict):
        emit("CD107", {"detail": "header is not a JSON object"})
        return None
    return parsed


def _check_entries(info: HeaderInfo, parsed: dict[str, Any], emit: Emit) -> None:
    ranges: list[tuple[int, int, str]] = []
    for name, value in parsed.items():
        if name == "__metadata__":
            if isinstance(value, dict) and all(
                isinstance(k, str) and isinstance(v, str) for k, v in value.items()
            ):
                info.metadata = value
            else:
                emit("CD112", {"what": "__metadata__"})
            continue
        entry = _parse_entry(name, value, emit)
        info.tensors[name] = entry
        if entry.start is not None and entry.end is not None:
            ranges.append((entry.start, entry.end, name))

    ranges.sort()
    expected = 0
    for start, end, name in ranges:
        if start != expected:
            emit("CD109", {"tensor": name, "offset": start, "expected": expected}, tensor=name)
            break
        expected = end
    declared_end = max((r[1] for r in ranges), default=0)
    actual = info.data_len
    if declared_end > actual:
        emit("CD110", {"missing": declared_end - actual, "declared": declared_end, "actual": actual})
    elif declared_end < actual:
        emit("CD111", {"extra": actual - declared_end, "declared": declared_end, "actual": actual})


def _parse_entry(name: str, value: Any, emit: Emit) -> TensorEntry:
    entry = TensorEntry(name, None, None, None, None)
    if not isinstance(value, dict):
        emit("CD112", {"what": name}, tensor=name)
        return entry
    dtype, shape, offsets = value.get("dtype"), value.get("shape"), value.get("data_offsets")
    if not isinstance(dtype, str):
        emit("CD112", {"what": name}, tensor=name)
    elif dtype not in DTYPE_BITS:
        emit("CD108", {"tensor": name, "dtype": dtype}, tensor=name)
    else:
        entry.dtype = dtype
    if isinstance(shape, list) and all(_is_int(d) and d >= 0 for d in shape):
        entry.shape = shape
    else:
        emit("CD112", {"what": name}, tensor=name)
    if (
        isinstance(offsets, list)
        and len(offsets) == 2
        and all(_is_int(o) for o in offsets)
        and 0 <= offsets[0] <= offsets[1]
    ):
        entry.start, entry.end = offsets
    else:
        emit("CD112", {"what": name}, tensor=name)
    if entry.dtype and entry.shape is not None and entry.start is not None and entry.end is not None:
        expected = math.ceil(math.prod(entry.shape) * DTYPE_BITS[entry.dtype] / 8)
        if entry.end - entry.start != expected:
            emit(
                "CD114",
                {"tensor": name, "declared": entry.end - entry.start, "expected": expected},
                tensor=name,
            )
    return entry
