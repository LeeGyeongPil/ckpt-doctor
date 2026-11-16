"""Result types shared by every layer of the checker."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class CkptDoctorUsageError(Exception):
    """Bad input (missing path, unknown cache entry, ...). The CLI maps it to exit code 2."""

    def __init__(self, key: str, **params: Any) -> None:
        self.key = key
        self.params = params
        from .i18n import t

        super().__init__(t(key, "en", **params))


class Severity(str, enum.Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class Finding:
    """One problem found. ``message_key`` + ``params`` are rendered per language."""

    code: str
    severity: Severity
    path: str | None
    message_key: str
    params: dict[str, Any] = field(default_factory=dict)
    tensor: str | None = None
    fix: str | None = None

    def message(self, lang: str = "en") -> str:
        from .i18n import t

        return t(self.message_key, lang, **self.params)


@dataclass
class Stats:
    bytes_read: int = 0
    bytes_hashed: int = 0
    files_checked: int = 0
    elapsed_ms: float = 0.0


@dataclass
class Report:
    target: str
    findings: list[Finding] = field(default_factory=list)
    stats: Stats = field(default_factory=Stats)
    context: dict[str, Any] = field(default_factory=dict)

    def count(self, severity: Severity) -> int:
        return sum(1 for f in self.findings if f.severity == severity)

    @property
    def ok(self) -> bool:
        """True when there are no error-level findings."""
        return self.count(Severity.ERROR) == 0

    def exit_code(self, strict: bool = False) -> int:
        if self.count(Severity.ERROR):
            return 1
        if strict and self.count(Severity.WARNING):
            return 1
        return 0

    def codes(self) -> set[str]:
        return {f.code for f in self.findings}
