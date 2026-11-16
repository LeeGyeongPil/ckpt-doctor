"""AC-8: JSON, SARIF, exit codes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ckpt_doctor import check_path
from ckpt_doctor.cli import main
from ckpt_doctor.codes import CODES

SCHEMA = json.loads((Path(__file__).parent.parent / "docs" / "report.schema.json").read_text())


def validate(value: Any, schema: dict[str, Any], where: str = "$") -> None:
    """Tiny JSON Schema subset check (type / required / properties / items / enum / const / pattern)."""
    import re

    types = {"object": dict, "array": list, "string": str, "integer": int, "number": (int, float), "boolean": bool, "null": type(None)}
    if "const" in schema:
        assert value == schema["const"], where
    if "enum" in schema:
        assert value in schema["enum"], where
    if "type" in schema:
        allowed = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        assert any(isinstance(value, types[t]) and not (t in ("integer", "number") and isinstance(value, bool)) for t in allowed), f"{where}: {value!r}"
    if "pattern" in schema:
        assert re.search(schema["pattern"], value), where
    if "minimum" in schema:
        assert value >= schema["minimum"], where
    if isinstance(value, dict):
        for key in schema.get("required", []):
            assert key in value, f"{where}: missing {key}"
        for key, sub in schema.get("properties", {}).items():
            if key in value:
                validate(value[key], sub, f"{where}.{key}")
    if isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            validate(item, schema["items"], f"{where}[{i}]")


def _json_of(capsys: pytest.CaptureFixture[str], args: list[str]) -> tuple[int, dict[str, Any]]:
    code = main(args)
    return code, json.loads(capsys.readouterr().out)


def test_json_conforms_to_schema(samples: Path, capsys: pytest.CaptureFixture[str]) -> None:
    for name in ("clean", "mixed", "missing-shard"):
        code, data = _json_of(capsys, ["check", str(samples / name), "--json"])
        validate(data, SCHEMA)
        assert data["ok"] == (code == 0)
    assert data["summary"]["errors"] == 1


def test_json_keys_do_not_depend_on_language(samples: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _, en = _json_of(capsys, ["check", str(samples / "mixed"), "--json", "--lang", "en"])
    _, ko = _json_of(capsys, ["check", str(samples / "mixed"), "--json", "--lang", "ko"])
    strip = lambda d: [{k: v for k, v in f.items() if k != "message"} for f in d["findings"]]  # noqa: E731
    assert strip(en) == strip(ko)
    assert en["findings"][0]["message"] != ko["findings"][0]["message"]


def test_sarif_structure(samples: Path, tmp_path: Path) -> None:
    out = tmp_path / "report.sarif"
    code = main(["check", str(samples / "mixed"), "--sarif", str(out)])
    assert code == 1
    sarif = json.loads(out.read_text(encoding="utf-8"))
    assert sarif["version"] == "2.1.0"
    run = sarif["runs"][0]
    rule_ids = {r["id"] for r in run["tool"]["driver"]["rules"]}
    assert rule_ids == set(CODES)
    results = run["results"]
    assert {r["ruleId"] for r in results} == {"CD101", "CD102", "CD203", "CD301"}
    for r in results:
        assert r["ruleId"] in rule_ids
        assert r["level"] in ("error", "warning", "note")
        assert r["locations"][0]["physicalLocation"]["artifactLocation"]["uri"]
    uris = {r["locations"][0]["physicalLocation"]["artifactLocation"]["uri"] for r in results}
    assert "model-00002-of-00004.safetensors" in uris


def test_exit_codes(samples: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check", str(samples / "clean")]) == 0
    assert main(["check", str(samples / "lfs-pointer")]) == 1
    assert main(["check", str(tmp_path / "does-not-exist")]) == 2
    assert main(["check", str(tmp_path / "does-not-exist"), "--lang", "ko"]) == 2
    err = capsys.readouterr().err
    assert "does-not-exist" in err
    (tmp_path / "notes.txt").write_text("hi")
    assert main(["check", str(tmp_path / "notes.txt")]) == 2
    assert main(["check", str(samples / "clean"), "--sarif", str(tmp_path / "no" / "dir" / "x.sarif")]) == 2


def test_warning_only_exit_codes(samples: Path) -> None:
    # config.json missing -> warning only
    report = check_path(samples / "clean" / "model-00001-of-00004.safetensors")
    assert report.exit_code(strict=True) == 0


def test_text_output_has_no_color_when_piped(samples: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main(["check", str(samples / "mixed"), "--lang", "en"])
    assert "\033[" not in capsys.readouterr().out
