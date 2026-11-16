"""AC-3: shard index versus shard headers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ckpt_doctor import check_path
from ckpt_doctor.cli import main
from ckpt_doctor.sample import clean_layout, make_index, shard_names, write_clean

INDEX = "model.safetensors.index.json"


def _edit_index(folder: Path, edit) -> None:
    path = folder / INDEX
    data = json.loads(path.read_text())
    edit(data)
    path.write_text(json.dumps(data))


@pytest.mark.parametrize(
    "name,code",
    [
        ("index-extra", "CD203"),
        ("index-dropped", "CD204"),
        ("duplicate-tensor", "CD206"),
        ("missing-shard", "CD202"),
        ("no-index", "CD201"),
    ],
)
def test_sample_faults(samples: Path, name: str, code: str) -> None:
    assert check_path(samples / name).codes() == {code}


def test_no_index_message_shows_shard_count(samples: Path) -> None:
    (finding,) = check_path(samples / "no-index").findings
    assert "3/4" in finding.message()
    assert shard_names()[2] in finding.fix  # the missing shard is named in the fix command


def test_wrong_shard(clean: Path) -> None:
    def edit(data: dict) -> None:
        data["weight_map"]["model.norm.weight"] = shard_names()[0]

    _edit_index(clean, edit)
    report = check_path(clean)
    assert report.codes() == {"CD205"}
    assert report.findings[0].params["actual"] == shard_names()[3]


def test_total_size_mismatch(clean: Path) -> None:
    def edit(data: dict) -> None:
        data["metadata"]["total_size"] += 2

    _edit_index(clean, edit)
    report = check_path(clean)
    assert report.codes() == {"CD207"}
    declared = make_index(clean_layout())["metadata"]["total_size"] + 2
    assert report.findings[0].params["declared"] == declared


def test_index_without_metadata_is_only_a_warning(clean: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _edit_index(clean, lambda d: d.pop("metadata"))
    report = check_path(clean)
    assert report.codes() == {"CD208"}
    assert report.exit_code() == 0
    assert report.exit_code(strict=True) == 1
    assert main(["check", str(clean), "--lang", "en"]) == 0
    assert main(["check", str(clean), "--strict", "--lang", "en"]) == 1


def test_unreferenced_shard_is_info(clean: Path) -> None:
    (clean / "stray.safetensors").write_bytes((clean / shard_names()[0]).read_bytes())
    report = check_path(clean)
    assert report.codes() == {"CD209"}
    assert report.exit_code(strict=True) == 0


def test_unsafe_path_is_not_followed(clean: Path) -> None:
    _edit_index(clean, lambda d: d["weight_map"].update({"x.weight": "../outside.safetensors"}))
    assert "CD210" in check_path(clean).codes()


def test_broken_index_json(clean: Path) -> None:
    (clean / INDEX).write_text("{oops")
    assert check_path(clean).codes() == {"CD211"}
    (clean / INDEX).write_text('{"metadata": {}}')
    assert check_path(clean).codes() == {"CD211"}


def test_complete_numbered_set_without_index(tmp_path: Path) -> None:
    write_clean(tmp_path)
    (tmp_path / INDEX).unlink()
    report = check_path(tmp_path)
    assert report.codes() == {"CD201"}
    assert "4/4" in report.findings[0].message()


def test_index_file_as_input(clean: Path) -> None:
    assert check_path(clean / INDEX).findings == []
