"""Streaming sha256 and ``sha256sum``-style manifests (the only code that reads tensor data)."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Callable

from .model import CkptDoctorUsageError

CHUNK = 8 * 1024 * 1024
_LINE = re.compile(r"^([0-9a-fA-F]{64}) [ *](.+)$")


def sha256_file(path: Path, on_chunk: Callable[[int], None] | None = None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(CHUNK):
            h.update(chunk)
            if on_chunk:
                on_chunk(len(chunk))
    return h.hexdigest()


def read_manifest(path: Path) -> dict[str, str]:
    """Parse ``<sha256>  <relative path>`` lines. Paths are relative to the checked folder."""
    if not path.is_file():
        raise CkptDoctorUsageError("err.manifest_not_found", path=str(path))
    entries: dict[str, str] = {}
    for no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            raise CkptDoctorUsageError("err.manifest_bad_line", path=str(path), line=no)
        name = m.group(2).replace("\\", "/")
        entries[name.removeprefix("./")] = m.group(1).lower()
    return entries


def write_manifest(path: Path, entries: dict[str, str]) -> None:
    lines = [f"{digest}  {name}" for name, digest in sorted(entries.items())]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
