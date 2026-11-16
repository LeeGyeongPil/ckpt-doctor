"""AC-6: only the 8-byte length and the JSON header are read, however big the file claims to be."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from ckpt_doctor.cli import main

GIB = 1024**3


def test_sparse_2gib_reads_header_only(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    n_elems = GIB  # F16: 2 bytes each -> 2 GiB of declared data
    header = json.dumps(
        {"big.weight": {"dtype": "F16", "shape": [n_elems], "data_offsets": [0, 2 * GIB]}},
        separators=(",", ":"),
    ).encode()
    header += b" " * (-len(header) % 8)
    path = tmp_path / "model.safetensors"
    with open(path, "wb") as fh:
        fh.write(len(header).to_bytes(8, "little") + header)
        fh.truncate(8 + len(header) + 2 * GIB)
    if path.stat().st_blocks * 512 > 50 * 1024 * 1024:
        pytest.skip("filesystem does not support sparse files")

    started = time.perf_counter()
    code = main(["check", str(path), "--json"])
    elapsed = time.perf_counter() - started
    report = json.loads(capsys.readouterr().out)

    assert code == 0
    assert report["findings"] == []
    assert report["stats"]["bytes_read"] < 1024 * 1024
    assert report["stats"]["bytes_hashed"] == 0
    assert elapsed < 2
