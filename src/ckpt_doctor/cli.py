"""Command line interface: check / explain / sample."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

from . import __version__
from .codes import CODES
from .engine import check_path
from .i18n import resolve_lang, t
from .model import CkptDoctorUsageError
from .report_json import render_json
from .report_sarif import render_sarif
from .report_text import render_text
from .sample import SAMPLE_NAMES, load_expected, write_samples


def _prescan_lang(argv: Sequence[str]) -> str | None:
    """Find --lang before argparse runs, so --help itself can be translated."""
    for i, arg in enumerate(argv):
        if arg == "--lang" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--lang="):
            return arg.split("=", 1)[1]
    return None


def build_parser(lang: str) -> argparse.ArgumentParser:
    def h(key: str) -> str:
        return t(f"help.{key}", lang)

    def add_lang(p: argparse.ArgumentParser, nested: bool) -> None:
        # nested copies use SUPPRESS so they never overwrite a --lang given before the command
        p.add_argument(
            "--lang", choices=["en", "ko"], help=h("lang"),
            default=argparse.SUPPRESS if nested else None,
        )  # fmt: skip

    parser = argparse.ArgumentParser(
        prog="ckpt-doctor", description=h("description"), epilog=h("epilog"), add_help=False
    )
    parser.add_argument("-h", "--help", action="help", help=t("help.help", lang))
    parser.add_argument("--version", action="version", version=f"ckpt-doctor {__version__}", help=h("version"))
    add_lang(parser, nested=False)
    sub = parser.add_subparsers(dest="command", metavar="{check,explain,sample}")

    def new_cmd(name: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=h(f"cmd.{name}"), description=h(f"cmd.{name}"), add_help=False)
        p.add_argument("-h", "--help", action="help", help=t("help.help", lang))
        add_lang(p, nested=True)
        return p

    check = new_cmd("check")
    check.add_argument("path", help=h("path"))
    check.add_argument("--revision", help=h("revision"))
    check.add_argument("--cache-dir", help=h("cache_dir"))
    check.add_argument("--repo-id", help=h("repo_id"))
    check.add_argument("--hash", action="store_true", help=h("hash"))
    check.add_argument("--manifest", metavar="FILE", help=h("manifest"))
    check.add_argument("--write-manifest", metavar="FILE", help=h("write_manifest"))
    check.add_argument("--json", action="store_true", help=h("json"))
    check.add_argument("--sarif", metavar="FILE", help=h("sarif"))
    check.add_argument("--strict", action="store_true", help=h("strict"))
    check.add_argument("--no-color", action="store_true", help=h("no_color"))

    explain = new_cmd("explain")
    explain.add_argument("code", nargs="?", help=h("code"))

    sample = new_cmd("sample")
    sample.add_argument("dir", help=h("sample_dir"))
    sample.add_argument("--check", action="store_true", help=h("sample_check"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    lang = resolve_lang(_prescan_lang(argv))
    parser = build_parser(lang)
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 2
    lang = resolve_lang(args.lang)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    try:
        if args.command == "check":
            return _cmd_check(args, lang)
        if args.command == "explain":
            return _cmd_explain(args, lang)
        if args.command == "sample":
            return _cmd_sample(args, lang)
    except CkptDoctorUsageError as exc:
        print(f"ckpt-doctor: {t(exc.key, lang, **exc.params)}", file=sys.stderr)
        return 2
    parser.print_help(sys.stderr)
    return 2


def _cmd_check(args: argparse.Namespace, lang: str) -> int:
    show_progress = sys.stderr.isatty() and not args.json

    def progress(rel: str, i: int, n: int) -> None:
        if show_progress:
            print(t("ui.hashing", lang, file=rel, i=i, n=n), file=sys.stderr, flush=True)

    report = check_path(
        args.path,
        revision=args.revision,
        cache_dir=args.cache_dir,
        repo_id=args.repo_id,
        hash_files=args.hash,
        manifest=args.manifest,
        write_manifest_to=args.write_manifest,
        progress=progress,
    )
    if args.json:
        print(render_json(report, lang))
    else:
        color = sys.stdout.isatty() and not os.environ.get("NO_COLOR") and not args.no_color
        print(render_text(report, lang, color))
    if args.sarif:
        try:
            Path(args.sarif).write_text(render_sarif(report, lang) + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"ckpt-doctor: {args.sarif}: {exc}", file=sys.stderr)
            return 2
    if report.context.get("manifest_written"):
        print(t("ui.manifest_written", lang, path=args.write_manifest), file=sys.stderr)
    elif report.context.get("manifest_requested"):
        print(t("ui.manifest_skipped", lang), file=sys.stderr)
    return report.exit_code(args.strict)


def _cmd_explain(args: argparse.Namespace, lang: str) -> int:
    if not args.code:
        print(t("ui.explain.list", lang))
        for code in sorted(CODES):
            print(f"  {code}  {t(f'{code}.title', lang)}")
        return 0
    code = args.code.upper()
    if code not in CODES:
        raise CkptDoctorUsageError("err.unknown_code", code=args.code)
    layer, severity = CODES[code]
    print(t("ui.explain.header", lang, code=code, title=t(f"{code}.title", lang)))
    print(t("ui.explain.meta", lang, layer=t(f"ui.layer.{layer}", lang), severity=t(f"ui.sev.{severity.value}", lang)))
    print(f"\n{t('ui.explain.cause', lang)}\n  {t(f'{code}.cause', lang)}")
    print(f"\n{t('ui.explain.fix', lang)}\n  {t(f'{code}.fix', lang)}")
    return 0


def _cmd_sample(args: argparse.Namespace, lang: str) -> int:
    dest = Path(args.dir)
    try:
        write_samples(dest)
    except OSError as exc:
        raise CkptDoctorUsageError("err.sample_dir", path=args.dir, detail=str(exc)) from exc
    print(t("ui.sample.created", lang, n=len(SAMPLE_NAMES), path=args.dir))
    if not args.check:
        print(t("ui.sample.try", lang, path=f"{args.dir}/mixed"))
        return 0
    matched = 0
    for name in SAMPLE_NAMES:
        folder = dest / name
        spec = load_expected(folder)["check"]
        if spec["path"] == ".":
            report = check_path(str(folder), hash_files=spec["hash"])
        else:
            report = check_path(spec["path"], cache_dir=str(folder / spec["cache_dir"]), hash_files=spec["hash"])
        got = sorted(report.codes())
        expected = sorted(load_expected(folder)["codes"])
        ok = got == expected
        matched += ok
        shown = ", ".join(got) if got else "-"
        status = t("ui.sample.ok" if ok else "ui.sample.bad", lang)
        print(t("ui.sample.row", lang, name=name, status=status, codes=shown))
        if not ok:
            print(t("ui.sample.mismatch", lang, expected=expected, got=got))
    print(t("ui.sample.summary", lang, matched=matched, total=len(SAMPLE_NAMES)))
    return 0 if matched == len(SAMPLE_NAMES) else 1

