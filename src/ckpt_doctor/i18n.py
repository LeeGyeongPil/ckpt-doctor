"""Language selection and message lookup (English / Korean)."""

from __future__ import annotations

import os
from typing import Any, Mapping

from .messages import MESSAGES

SUPPORTED = ("en", "ko")
ENV_VAR = "CKPT_DOCTOR_LANG"


def _normalize(value: str) -> str:
    return "ko" if value.strip().lower().startswith("ko") else "en"


def resolve_lang(cli_lang: str | None = None, env: Mapping[str, str] | None = None) -> str:
    """``--lang`` > ``CKPT_DOCTOR_LANG`` > ``LC_ALL`` > ``LC_MESSAGES`` > ``LANG``; anything but ko is en."""
    if cli_lang:
        return _normalize(cli_lang)
    env = os.environ if env is None else env
    for name in (ENV_VAR, "LC_ALL", "LC_MESSAGES", "LANG"):
        value = env.get(name, "")
        if value.strip():
            return _normalize(value)
    return "en"


def t(key: str, lang: str = "en", **params: Any) -> str:
    """Look up ``key`` in ``lang`` and format it with ``params``."""
    table = MESSAGES[lang if lang in MESSAGES else "en"]
    text = table[key]
    return text.format(**params) if params else text
