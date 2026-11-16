"""Human-readable report."""

from __future__ import annotations

from .codes import LAYERS, layer_of
from .i18n import t
from .model import Finding, Report, Severity
from .safetensors_header import human_bytes

_COLORS = {Severity.ERROR: "31", Severity.WARNING: "33", Severity.INFO: "36"}


def _paint(text: str, code: str, color: bool) -> str:
    return f"\033[{code}m{text}\033[0m" if color else text


def render_text(report: Report, lang: str = "en", color: bool = False) -> str:
    from . import __version__

    st = report.stats
    lines = [_paint(f"ckpt-doctor {__version__}", "1", color) + f"  {report.target}"]
    ctx = report.context
    if ctx.get("repo_id") and ctx.get("commit"):
        lines.append(t("ui.repo", lang, repo_id=ctx["repo_id"], commit=str(ctx["commit"])[:12]))
    checked = t(
        "ui.checked", lang, files=st.files_checked, read=human_bytes(st.bytes_read), ms=f"{st.elapsed_ms:.0f}"
    )
    if st.bytes_hashed:
        checked += ", " + t("ui.checked_hash", lang, hashed=human_bytes(st.bytes_hashed))
    lines.append(checked)

    for layer in LAYERS:
        group = [f for f in report.findings if layer_of(f.code) == layer]
        if not group:
            continue
        lines += ["", _paint(t(f"ui.layer.{layer}", lang), "1", color)]
        for f in group:
            lines += _render_finding(f, lang, color)

    lines.append("")
    errors, warnings, infos = (report.count(s) for s in Severity)
    if not report.findings:
        lines.append(_paint(t("ui.result_ok", lang), "32", color))
    else:
        verdict = t("ui.result_failed" if errors else "ui.result_passed", lang)
        summary = t("ui.result", lang, errors=errors, warnings=warnings, infos=infos)
        lines.append(_paint(f"{summary}: {verdict}", "31" if errors else "32", color))
    return "\n".join(lines)


def _render_finding(f: Finding, lang: str, color: bool) -> list[str]:
    sev = _paint(f"{t('ui.sev.' + f.severity.value, lang):<5}", _COLORS[f.severity], color)
    where = f" {f.path}" if f.path else ""
    out = [f"  {sev} {f.code}{where}", f"        {f.message(lang)}"]
    if f.fix:
        out.append(f"        {t('ui.fix', lang)}: {_paint(f.fix, '1', color)}")
    return out
