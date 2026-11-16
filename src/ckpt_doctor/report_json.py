"""JSON report (schema: docs/report.schema.json). Keys never change with the language."""

from __future__ import annotations

import json
from typing import Any

from .codes import layer_of
from .model import Report, Severity

SCHEMA_VERSION = 1


def report_to_dict(report: Report, lang: str = "en") -> dict[str, Any]:
    from . import __version__

    return {
        "schema_version": SCHEMA_VERSION,
        "tool": {"name": "ckpt-doctor", "version": __version__},
        "target": report.target,
        "ok": report.ok,
        "summary": {
            "errors": report.count(Severity.ERROR),
            "warnings": report.count(Severity.WARNING),
            "info": report.count(Severity.INFO),
        },
        "stats": {
            "bytes_read": report.stats.bytes_read,
            "bytes_hashed": report.stats.bytes_hashed,
            "files_checked": report.stats.files_checked,
            "elapsed_ms": report.stats.elapsed_ms,
        },
        "context": {
            "kind": report.context.get("kind"),
            "repo_id": report.context.get("repo_id"),
            "commit": report.context.get("commit"),
        },
        "findings": [
            {
                "code": f.code,
                "severity": f.severity.value,
                "layer": layer_of(f.code),
                "path": f.path,
                "tensor": f.tensor,
                "message": f.message(lang),
                "params": f.params,
                "fix": f.fix,
            }
            for f in report.findings
        ],
    }


def render_json(report: Report, lang: str = "en") -> str:
    return json.dumps(report_to_dict(report, lang), indent=2, ensure_ascii=False)
