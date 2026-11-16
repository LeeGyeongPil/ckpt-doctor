"""AC-9: English and Korean output, language precedence, identical key sets."""

from __future__ import annotations

import re
import string
from pathlib import Path

import pytest

from ckpt_doctor.cli import main
from ckpt_doctor.codes import CODES
from ckpt_doctor.i18n import resolve_lang, t
from ckpt_doctor.messages import EN, KO

HANGUL = re.compile(r"[가-힣]")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("CKPT_DOCTOR_LANG", "LC_ALL", "LC_MESSAGES", "LANG"):
        monkeypatch.delenv(name, raising=False)


def test_key_sets_are_identical() -> None:
    assert set(EN) == set(KO)


def test_every_code_has_all_four_texts() -> None:
    for code in CODES:
        for part in ("title", "msg", "cause", "fix"):
            assert EN[f"{code}.{part}"].strip() and KO[f"{code}.{part}"].strip()


def test_placeholders_match_between_languages() -> None:
    def fields(text: str) -> set[str]:
        return {name.split(":")[0] for _, name, _, _ in string.Formatter().parse(text) if name}

    for key in EN:
        assert fields(EN[key]) == fields(KO[key]), key


def test_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    assert resolve_lang() == "en"
    monkeypatch.setenv("LANG", "ko_KR.UTF-8")
    assert resolve_lang() == "ko"
    monkeypatch.setenv("LC_ALL", "en_US.UTF-8")
    assert resolve_lang() == "en"  # LC_ALL beats LANG
    monkeypatch.setenv("CKPT_DOCTOR_LANG", "ko")
    assert resolve_lang() == "ko"  # tool variable beats LC_ALL
    assert resolve_lang("en") == "en"  # --lang beats everything


def test_unsupported_locale_falls_back_to_english(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANG", "ja_JP.UTF-8")
    assert resolve_lang() == "en"
    monkeypatch.setenv("LC_ALL", "C")
    assert resolve_lang() == "en"


def test_lang_flag_switches_check_output(samples: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main(["check", str(samples / "lfs-pointer"), "--lang", "ko"])
    ko = capsys.readouterr().out
    main(["check", str(samples / "lfs-pointer"), "--lang", "en"])
    en = capsys.readouterr().out
    assert HANGUL.search(ko) and not HANGUL.search(en)
    assert "CD101" in ko and "CD101" in en
    assert "git lfs pull --include" in ko and "git lfs pull --include" in en  # commands are not translated


def test_lang_flag_before_command(samples: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main(["--lang", "ko", "check", str(samples / "lfs-pointer")])
    assert HANGUL.search(capsys.readouterr().out)


def test_help_in_both_languages(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--help", "--lang", "ko"]) == 0
    ko = capsys.readouterr().out
    assert main(["--help", "--lang", "en"]) == 0
    en = capsys.readouterr().out
    assert HANGUL.search(ko) and not HANGUL.search(en)
    assert main(["check", "--help", "--lang", "ko"]) == 0
    assert HANGUL.search(capsys.readouterr().out)


def test_explain_in_both_languages(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["explain", "CD101", "--lang", "ko"]) == 0
    ko = capsys.readouterr().out
    assert main(["explain", "cd101", "--lang", "en"]) == 0
    en = capsys.readouterr().out
    assert HANGUL.search(ko) and not HANGUL.search(en)
    assert "CD101" in ko and "CD101" in en
    assert "git lfs pull" in en


def test_environment_variable_selects_language(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("CKPT_DOCTOR_LANG", "ko")
    main(["explain", "CD110"])
    assert HANGUL.search(capsys.readouterr().out)


def test_usage_error_is_localised(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check", str(tmp_path / "nope"), "--lang", "ko"]) == 2
    assert HANGUL.search(capsys.readouterr().err)
    assert t("err.unknown_code", "en", code="X").startswith("unknown finding code")
