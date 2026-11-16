"""AC-2: every file-layer cause gets its own code, with no cross-layer noise."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ckpt_doctor import check_path
from ckpt_doctor.sample import HTML_PAGE, lfs_pointer_text, safetensors_bytes

TENSORS = [("a.weight", "F16", [2, 2]), ("b.weight", "F32", [2])]


def _raw(header_text: str, data: bytes) -> bytes:
    raw = header_text.encode()
    return len(raw).to_bytes(8, "little") + raw + data


def _case_dup_key() -> bytes:
    entry = '{"dtype":"U8","shape":[1],"data_offsets":[0,1]}'
    return _raw(f'{{"a":{entry},"a":{entry}}}', b"\x01")


def _case_hole() -> bytes:
    header = (
        '{"a":{"dtype":"U8","shape":[4],"data_offsets":[0,4]},'
        '"b":{"dtype":"U8","shape":[4],"data_offsets":[8,12]}}'
    )
    return _raw(header, b"\x01" * 12)


GOOD = safetensors_bytes(TENSORS, {"format": "pt"})

CASES = {
    "lfs-pointer": (lfs_pointer_text("m.safetensors", 4_900_000_000), "CD101"),
    "html": (HTML_PAGE, "CD102"),
    "zeros16": (b"\x00" * 16, "CD103"),
    "huge-length": (((200_000_000).to_bytes(8, "little")) + b"x" * 64, "CD104"),
    "truncated": (GOOD[:-6], "CD110"),
    "trailing": (GOOD + b"junk", "CD111"),
    "duplicate-key": (_case_dup_key(), "CD106"),
    "hole": (_case_hole(), "CD109"),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_fault_classification(tmp_path: Path, name: str) -> None:
    data, expected = CASES[name]
    path = tmp_path / "model.safetensors"
    path.write_bytes(data)
    report = check_path(path)
    assert report.codes() == {expected}, [f.message() for f in report.findings]
    assert not report.ok


def test_lfs_pointer_shows_size_and_fix(tmp_path: Path) -> None:
    path = tmp_path / "model.safetensors"
    path.write_bytes(lfs_pointer_text("m", 4_900_000_000))
    (finding,) = check_path(path).findings
    assert "4.9 GB" in finding.message()
    assert finding.params["size"] == path.stat().st_size
    assert finding.fix == 'git lfs pull --include "model.safetensors"'


def test_truncation_reports_exact_missing_bytes(tmp_path: Path) -> None:
    path = tmp_path / "model.safetensors"
    path.write_bytes(GOOD[:-6])
    (finding,) = check_path(path).findings
    assert finding.params["missing"] == 6
    assert "6 bytes missing" in finding.message()


def test_trailing_bytes_counted(tmp_path: Path) -> None:
    path = tmp_path / "model.safetensors"
    path.write_bytes(GOOD + b"junk")
    (finding,) = check_path(path).findings
    assert finding.params["extra"] == 4


def test_valid_file_is_clean(tmp_path: Path) -> None:
    path = tmp_path / "model.safetensors"
    path.write_bytes(GOOD)
    assert check_path(path).findings == []


def test_bad_json_and_entry_problems(tmp_path: Path) -> None:
    path = tmp_path / "bad.safetensors"
    path.write_bytes(_raw("{not json", b""))
    assert check_path(path).codes() == {"CD107"}
    path.write_bytes(_raw("[1,2]", b""))
    assert check_path(path).codes() == {"CD107"}
    unknown = '{"a":{"dtype":"F7","shape":[1],"data_offsets":[0,1]}}'
    path.write_bytes(_raw(unknown, b"\x01"))
    assert check_path(path).codes() == {"CD108"}
    wrong_size = '{"a":{"dtype":"F16","shape":[4],"data_offsets":[0,4]}}'
    path.write_bytes(_raw(wrong_size, b"\x01" * 4))
    assert check_path(path).codes() == {"CD114"}
    bad_meta = json.dumps({"__metadata__": {"format": 1}})
    path.write_bytes(_raw(bad_meta, b""))
    assert check_path(path).codes() == {"CD112"}


def test_header_longer_than_file(tmp_path: Path) -> None:
    path = tmp_path / "cut.safetensors"
    path.write_bytes(GOOD[:30])
    assert check_path(path).codes() == {"CD105"}


def test_tiny_file_under_eight_bytes(tmp_path: Path) -> None:
    path = tmp_path / "tiny.safetensors"
    path.write_bytes(b"abc")
    assert check_path(path).codes() == {"CD103"}


def test_fix_local_dir_is_parent_folder_for_file_and_index_inputs(tmp_path: Path) -> None:
    import json as _json
    from ckpt_doctor.sample import safetensors_bytes

    data = safetensors_bytes([("a", "F16", [4])])
    (tmp_path / "model.safetensors").write_bytes(data[:-2])
    (tmp_path / "model.safetensors.index.json").write_text(
        _json.dumps({"metadata": {"total_size": 8}, "weight_map": {"a": "model.safetensors"}})
    )
    for name in ("model.safetensors", "model.safetensors.index.json"):
        report = check_path(str(tmp_path / name), repo_id="demo/tiny")
        fixes = [f.fix for f in report.findings if f.code == "CD110"]
        assert fixes and all(fx and f"--local-dir {tmp_path}" in fx and not fx.rstrip().endswith(name) for fx in fixes)
        assert all(f"--local-dir {tmp_path}/{name}" not in fx for fx in fixes)
