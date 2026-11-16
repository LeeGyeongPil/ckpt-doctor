"""Git LFS pointer file detection (https://github.com/git-lfs/git-lfs/blob/main/docs/spec.md)."""

from __future__ import annotations

import re
from dataclasses import dataclass

LFS_SPEC_MARKER = b"https://git-lfs.github.com/spec/"
_OID = re.compile(rb"^oid sha256:([0-9a-f]{64})\s*$", re.MULTILINE)
_SIZE = re.compile(rb"^size (\d+)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class LfsPointer:
    oid: str | None
    size: int | None


def parse_pointer(head: bytes) -> LfsPointer | None:
    """Return the parsed pointer if ``head`` (first bytes of a file) is a Git LFS pointer."""
    if not head.startswith(b"version ") or LFS_SPEC_MARKER not in head[:200]:
        return None
    oid = _OID.search(head)
    size = _SIZE.search(head)
    return LfsPointer(
        oid=oid.group(1).decode() if oid else None,
        size=int(size.group(1)) if size else None,
    )
