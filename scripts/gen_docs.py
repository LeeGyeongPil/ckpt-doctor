"""Regenerate docs/codes.md and docs/codes.ko.md from the message tables.

Run: uv run python scripts/gen_docs.py
"""

import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "src"))
from ckpt_doctor.codes import CODES, LAYERS  # noqa: E402
from ckpt_doctor.messages import MESSAGES  # noqa: E402

HEADS = {
    "en": (
        "# Finding codes",
        "Every finding has a stable code. `ckpt-doctor explain CODE` prints the same text.\n"
        "The severity shown is the default; a few codes lower it in context.",
        ["Code", "Severity", "Meaning", "Why it happens", "How to fix"],
    ),
    "ko": (
        "# 결함 코드",
        "모든 결함에는 바뀌지 않는 코드가 있습니다. `ckpt-doctor explain CODE` 가 같은 내용을 보여 줍니다.\n"
        "표의 심각도는 기본값이며 일부 코드는 상황에 따라 낮아집니다.",
        ["코드", "심각도", "의미", "왜 생기나", "고치는 법"],
    ),
}

for lang, name in (("en", "codes.md"), ("ko", "codes.ko.md")):
    title, intro, cols = HEADS[lang]
    m = MESSAGES[lang]
    out = [title, "", intro, ""]
    for layer in LAYERS:
        out += [f"## {m['ui.layer.' + layer]}", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for code, (lay, sev) in sorted(CODES.items()):
            if lay == layer:
                out.append(
                    f"| `{code}` | {sev.value} | {m[code + '.title']} | {m[code + '.cause']} | {m[code + '.fix']} |"
                )
        out.append("")
    (root / "docs" / name).write_text("\n".join(out), encoding="utf-8")
print("ok")
