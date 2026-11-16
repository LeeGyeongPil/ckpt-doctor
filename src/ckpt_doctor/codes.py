"""Finding code table: code -> (layer, default severity)."""

from __future__ import annotations

from .model import Severity

E, W, I = Severity.ERROR, Severity.WARNING, Severity.INFO

LAYERS = ["input", "file", "index", "config", "cache", "hash"]

CODES: dict[str, tuple[str, Severity]] = {
    "CD001": ("input", W),
    # file layer
    "CD101": ("file", E),
    "CD102": ("file", E),
    "CD103": ("file", E),
    "CD104": ("file", E),
    "CD105": ("file", E),
    "CD106": ("file", E),
    "CD107": ("file", E),
    "CD108": ("file", E),
    "CD109": ("file", E),
    "CD110": ("file", E),
    "CD111": ("file", E),
    "CD112": ("file", E),
    "CD114": ("file", E),
    # index layer
    "CD201": ("index", E),
    "CD202": ("index", E),
    "CD203": ("index", E),
    "CD204": ("index", E),
    "CD205": ("index", E),
    "CD206": ("index", E),
    "CD207": ("index", E),
    "CD208": ("index", W),
    "CD209": ("index", I),
    "CD210": ("index", E),
    "CD211": ("index", E),
    # config layer
    "CD301": ("config", E),
    "CD302": ("config", E),
    "CD303": ("config", W),
    "CD304": ("config", W),
    "CD305": ("config", W),
    "CD306": ("config", W),
    "CD390": ("config", I),
    "CD391": ("config", I),
    # cache layer
    "CD401": ("cache", W),
    "CD402": ("cache", E),
    "CD403": ("cache", E),
    "CD404": ("cache", E),
    "CD405": ("cache", E),
    "CD406": ("cache", I),
    "CD407": ("cache", W),
    # hash layer
    "CD501": ("hash", E),
    "CD502": ("hash", I),
    "CD503": ("hash", E),
}


def layer_of(code: str) -> str:
    return CODES[code][0]


def default_severity(code: str) -> Severity:
    return CODES[code][1]
