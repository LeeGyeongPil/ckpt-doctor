"""AC-5: Hugging Face cache layer, fully offline."""

from __future__ import annotations

import os
import shlex
import shutil
from pathlib import Path

import pytest

from ckpt_doctor import CkptDoctorUsageError, check_path
from ckpt_doctor.sample import COMMIT


@pytest.fixture
def cache(samples: Path) -> Path:
    return samples / "hf-cache"


def test_repo_id_resolves_via_hf_hub_cache(cache: Path, no_network: None) -> None:
    env = {"HF_HUB_CACHE": str(cache)}
    report = check_path("demo/tiny", env=env)
    assert report.context["commit"] == COMMIT
    assert report.context["repo_id"] == "demo/tiny"
    assert report.codes() == {"CD401", "CD402", "CD403", "CD405"}


def test_findings_point_at_the_right_files(cache: Path, no_network: None) -> None:
    report = check_path("demo/tiny", cache_dir=str(cache))
    by_code = {f.code: f for f in report.findings}
    assert by_code["CD402"].path.endswith("snapshots/%s/tokenizer.json" % COMMIT)
    assert by_code["CD401"].path.endswith(".incomplete")
    assert by_code["CD405"].params["expected"] - by_code["CD405"].params["actual"] == 11
    assert by_code["CD403"].params["ref"] == "dev"
    assert by_code["CD403"].severity.value == "warning"  # not the checked revision
    q = "--cache-dir %s" % shlex.quote(str(cache))
    assert by_code["CD402"].fix == "hf download demo/tiny tokenizer.json %s --force-download" % q


def test_hf_home_fallback(cache: Path, tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    os.symlink(cache, home / "hub")
    assert check_path("demo/tiny", env={"HF_HOME": str(home)}).context["repo_id"] == "demo/tiny"


def test_ref_to_missing_snapshot_is_an_error_when_selected(cache: Path) -> None:
    report = check_path("demo/tiny", cache_dir=str(cache), revision="dev")
    assert report.codes() == {"CD403", "CD401"}
    errors = {f.code for f in report.findings if f.severity.value == "error"}
    assert errors == {"CD403"}


def test_cache_repo_folder_and_snapshot_folder_inputs(cache: Path) -> None:
    repo_dir = cache / "models--demo--tiny"
    from_repo = check_path(repo_dir)
    from_snapshot = check_path(repo_dir / "snapshots" / COMMIT)
    assert from_repo.codes() == from_snapshot.codes() == {"CD401", "CD402", "CD403", "CD405"}


def test_unknown_repo_and_revision_are_usage_errors(cache: Path) -> None:
    with pytest.raises(CkptDoctorUsageError):
        check_path("nobody/nothing", cache_dir=str(cache))
    with pytest.raises(CkptDoctorUsageError):
        check_path("demo/tiny", cache_dir=str(cache), revision="v9")


def test_local_dir_trees_size_mismatch(clean: Path) -> None:
    import json

    trees = clean / ".cache" / "huggingface" / "trees"
    trees.mkdir(parents=True)
    size = (clean / "config.json").stat().st_size
    entry = {"config.json": {"size": size + 1, "blob_id": "x"}, "only-in-listing.bin": {"size": 1, "blob_id": "y"}}
    (trees / f"{COMMIT}.json").write_text(json.dumps({"format_version": 1, "files": entry}))
    report = check_path(clean)
    assert report.codes() == {"CD405"}  # a partial local_dir download is allowed to miss files


def test_unsupported_and_broken_trees(clean: Path) -> None:
    trees = clean / ".cache" / "huggingface" / "trees"
    trees.mkdir(parents=True)
    (trees / "a.json").write_text('{"format_version": 99, "files": {}}')
    assert check_path(clean).codes() == {"CD406"}
    (trees / "a.json").write_text("{nope")
    assert check_path(clean).codes() == {"CD407"}


def test_fix_commands_carry_cache_dir_and_revision(tmp_path: Path, cache: Path) -> None:
    cd = str(tmp_path / "hub 2")  # a space forces quoting
    shutil.copytree(cache, cd, symlinks=True)
    (Path(cd) / "models--demo--tiny" / "refs" / "dev").write_text(COMMIT)
    report = check_path("demo/tiny", cache_dir=cd, revision="dev")
    by_code = {f.code: f for f in report.findings}
    q = shlex.quote(cd)
    assert q.startswith("'")
    assert by_code["CD402"].fix == (
        f"hf download demo/tiny tokenizer.json --cache-dir {q} --revision dev --force-download"
    )
    assert by_code["CD405"].fix == (
        f"hf download demo/tiny config.json --cache-dir {q} --revision dev --force-download"
    )
    assert by_code["CD401"].fix == f"hf cache prune --dry-run --cache-dir {q}"


def test_fix_commands_unchanged_without_cache_dir_or_revision(cache: Path) -> None:
    report = check_path("demo/tiny", env={"HF_HUB_CACHE": str(cache)})
    by_code = {f.code: f for f in report.findings}
    assert by_code["CD402"].fix == "hf download demo/tiny tokenizer.json --force-download"
    assert by_code["CD401"].fix == "hf cache prune --dry-run"
