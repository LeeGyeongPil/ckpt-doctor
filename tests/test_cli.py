"""AC-1 and end-to-end CLI behaviour."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from ckpt_doctor import __version__
from ckpt_doctor.cli import main
from ckpt_doctor.sample import SAMPLE_NAMES, load_expected


def test_sample_check_matches_every_expectation(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["sample", str(tmp_path / "s"), "--check", "--lang", "en"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[-1] == "14/14 samples matched"
    clean_row = next(line for line in lines if line.startswith("clean"))
    assert "OK" in clean_row and "CD" not in clean_row


def test_sample_folders_have_expected_json(samples: Path) -> None:
    assert len(SAMPLE_NAMES) == 14
    for name in SAMPLE_NAMES:
        spec = load_expected(samples / name)
        assert "codes" in spec and "check" in spec
    assert load_expected(samples / "clean")["codes"] == []


def test_sample_without_check_just_writes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["sample", str(tmp_path / "s"), "--lang", "en"]) == 0
    out = capsys.readouterr().out
    assert "not real model weights" in out
    assert (tmp_path / "s" / "mixed" / "config.json").exists()


def test_sample_rerun_replaces_folders(tmp_path: Path) -> None:
    for _ in range(2):
        assert main(["sample", str(tmp_path / "s"), "--check"]) == 0


def test_explain_lists_and_rejects(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["explain", "--lang", "en"]) == 0
    assert "CD101" in capsys.readouterr().out
    assert main(["explain", "CD999", "--lang", "en"]) == 2


def test_check_single_shard_and_index_file(samples: Path) -> None:
    assert main(["check", str(samples / "clean" / "model-00001-of-00004.safetensors")]) == 0
    assert main(["check", str(samples / "lfs-pointer" / "model.safetensors.index.json")]) == 1


def test_check_with_repo_id_changes_fix_command(samples: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main(["check", str(samples / "html-page"), "--repo-id", "org/model", "--lang", "en"])
    assert "hf download org/model model-00003-of-00004.safetensors" in capsys.readouterr().out


def test_empty_folder_warns(tmp_path: Path) -> None:
    from ckpt_doctor import check_path

    report = check_path(tmp_path)
    assert report.codes() == {"CD001"}
    assert report.exit_code() == 0 and report.exit_code(strict=True) == 1


def test_version_and_module_entry(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--version"]) == 0
    assert __version__ in capsys.readouterr().out
    proc = subprocess.run(
        [sys.executable, "-m", "ckpt_doctor", "--version"], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0 and __version__ in proc.stdout


def test_no_command_prints_help_and_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 2


def test_stdout_json_is_pure(samples: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main(["check", str(samples / "clean"), "--json"])
    json.loads(capsys.readouterr().out)
