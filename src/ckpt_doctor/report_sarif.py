"""SARIF 2.1.0 output for code scanning."""

from __future__ import annotations

import json
from typing import Any

from .codes import CODES
from .i18n import t
from .model import Report, Severity

SARIF_SCHEMA = "https://json.schemastore.org/sarif-2.1.0.json"
_LEVEL = {Severity.ERROR: "error", Severity.WARNING: "warning", Severity.INFO: "note"}


def report_to_sarif(report: Report, lang: str = "en") -> dict[str, Any]:
    from . import __version__

    rules = [
        {
            "id": code,
            "name": code,
            "shortDescription": {"text": t(f"{code}.title", lang)},
            "help": {"text": f"{t(f'{code}.cause', lang)} {t(f'{code}.fix', lang)}"},
            "defaultConfiguration": {"level": _LEVEL[sev]},
        }
        for code, (_layer, sev) in sorted(CODES.items())
    ]
    results = []
    for f in report.findings:
        text = f.message(lang)
        if f.fix:
            text += f" ({t('ui.fix', lang)}: {f.fix})"
        results.append(
            {
                "ruleId": f.code,
                "level": _LEVEL[f.severity],
                "message": {"text": text},
                "locations": [
                    {"physicalLocation": {"artifactLocation": {"uri": f.path or "."}}}
                ],
            }
        )
    return {
        "$schema": SARIF_SCHEMA,
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "ckpt-doctor",
                        "version": __version__,
                        "informationUri": "https://github.com/LeeGyeongPil/ckpt-doctor",
                        "rules": rules,
                    }
                },
                "results": results,
            }
        ],
    }


def render_sarif(report: Report, lang: str = "en") -> str:
    return json.dumps(report_to_sarif(report, lang), indent=2, ensure_ascii=False)
