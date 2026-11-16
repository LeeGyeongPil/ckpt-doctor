"""AC-7: --hash, manifests, and the baseline order."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from ckpt_doctor import check_path
from ckpt_doctor.cli import main
from ckpt_doctor.hashing import read_manifest
from ckpt_doctor.sample import COMMIT, shard_names
from ckpt_doctor import CkptDoctorUsageError


def _flip_last_byte(path: Path) -> None:
    data = bytearray(path.read_bytes())
    data[-1] ^= 0xFF  # data region only; the header stays valid
    path.write_bytes(bytes(data))


def _write_trees(folder: Path, shards: list[str]) -> None:
    trees = folder / ".cache" / "huggingface" / "trees"
    trees.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in shards:
        raw = (folder / name).read_bytes()
        files[name] = {
            "size": len(raw), "blob_id": "x", "lfs_sha256": hashlib.sha256(raw).hexdigest(), "lfs_size": len(raw),
        }  # fmt: skip
    (trees / f"{COMMIT}.json").write_text(json.dumps({"format_version": 1, "files": files}))


def test_intact_folder_matches_trees_listing(clean: Path) -> None:
    _write_trees(clean, shard_names())
    report = check_path(clean, hash_files=True)
    assert report.findings == []
    assert report.stats.bytes_hashed > 0


def test_one_byte_change_found_only_with_hash(clean: Path) -> None:
    _write_trees(clean, shard_names())
    _flip_last_byte(clean / shard_names()[1])
    assert check_path(clean).findings == []  # header intact: invisible without --hash
    report = check_path(clean, hash_files=True)
    assert report.codes() == {"CD501"}
    finding = report.findings[0]
    assert finding.path == shard_names()[1]
    assert finding.params["source"] == "trees"


def test_manifest_round_trip(clean: Path, tmp_path: Path) -> None:
    manifest = tmp_path / "SHA256SUMS"
    report = check_path(clean, write_manifest_to=manifest)
    assert report.findings == [] and report.context["manifest_written"]
    entries = read_manifest(manifest)
    assert set(entries) == set(shard_names())
    assert check_path(clean, hash_files=True, manifest=manifest).findings == []
    _flip_last_byte(clean / shard_names()[3])
    tampered = check_path(clean, hash_files=True, manifest=manifest)
    assert tampered.codes() == {"CD501"}
    assert tampered.findings[0].params["source"] == "manifest"


def test_manifest_lines_work_with_sha256sum(clean: Path, tmp_path: Path) -> None:
    manifest = tmp_path / "SHA256SUMS"
    check_path(clean, write_manifest_to=manifest)
    first = manifest.read_text().splitlines()[0]
    digest, name = first.split("  ", 1)
    assert hashlib.sha256((clean / name).read_bytes()).hexdigest() == digest


def test_manifest_not_written_when_folder_has_errors(samples: Path, tmp_path: Path) -> None:
    manifest = tmp_path / "SHA256SUMS"
    report = check_path(samples / "truncated", write_manifest_to=manifest)
    assert not manifest.exists()
    assert report.context["manifest_requested"] and not report.context["manifest_written"]


def test_manifest_listing_missing_file(clean: Path, tmp_path: Path) -> None:
    manifest = tmp_path / "SHA256SUMS"
    check_path(clean, write_manifest_to=manifest)
    with open(manifest, "a") as fh:
        fh.write(f"{'0' * 64}  ghost.safetensors\n")
    assert check_path(clean, hash_files=True, manifest=manifest).codes() == {"CD503"}


def test_no_baseline_is_info(clean: Path) -> None:
    report = check_path(clean, hash_files=True)
    assert report.codes() == {"CD502"}
    assert report.exit_code(strict=True) == 0


def test_bad_manifest_is_usage_error(clean: Path, tmp_path: Path) -> None:
    bad = tmp_path / "bad"
    bad.write_text("not a manifest line\n")
    with pytest.raises(CkptDoctorUsageError):
        check_path(clean, hash_files=True, manifest=bad)
    with pytest.raises(CkptDoctorUsageError):
        check_path(clean, hash_files=True, manifest=tmp_path / "missing")


def test_cache_blob_name_is_lowest_priority_baseline(samples: Path, tmp_path: Path) -> None:
    # sample cache: blob name and trees both carry the original sha256; trees wins
    cache = tmp_path / "cache"
    shutil.copytree(samples / "hf-cache", cache, symlinks=True)
    report = check_path("demo/tiny", cache_dir=str(cache), hash_files=True)
    finding = next(f for f in report.findings if f.code == "CD501")
    assert finding.params["source"] == "trees"
    os.remove(cache / "models--demo--tiny" / "trees" / f"{COMMIT}.json")
    report = check_path("demo/tiny", cache_dir=str(cache), hash_files=True)
    finding = next(f for f in report.findings if f.code == "CD501")
    assert finding.params["source"] == "blob"


def test_cli_write_manifest_then_check(clean: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest = tmp_path / "SUMS"
    assert main(["check", str(clean), "--write-manifest", str(manifest), "--lang", "en"]) == 0
    assert "manifest written" in capsys.readouterr().err
    assert main(["check", str(clean), "--hash", "--manifest", str(manifest), "--lang", "en"]) == 0
