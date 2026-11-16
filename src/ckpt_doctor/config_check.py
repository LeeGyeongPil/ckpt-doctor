"""Compare ``config.json`` with what the safetensors headers actually contain."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Iterable

from .model import Severity
from .safetensors_header import TensorEntry

Emit = Callable[..., None]

# First match wins, so "model.layers" is not also counted as a bare "layers" family.
FAMILIES = ["model.layers", "encoder.layer", "transformer.h", "blocks", "layers"]
_FAMILY_RES = [(f, re.compile(r"(?:^|\.)" + re.escape(f) + r"\.(\d+)(?:\.|$)")) for f in FAMILIES]

_BARE_H_RE = re.compile(r"^h\.(\d+)\.")
# Output-head tensor names used by different families (GPT-NeoX: embed_out, ChatGLM: output_layer, ...).
HEAD_NAMES = ("lm_head.weight", "embed_out.weight", "output.weight", "output_layer.weight")

LAYER_KEYS = ("num_hidden_layers", "n_layer", "num_layers", "n_layers")
NESTED = ("text_config", "language_config", "llm_config", "decoder", "text_model")
HEAD_ARCH = re.compile(r"CausalLM|LMHead|ConditionalGeneration|Seq2SeqLM|MaskedLM")
FLOATS = {"F16", "BF16", "F32", "F64"}
DTYPE_NAMES = {
    "float16": "F16", "half": "F16", "bfloat16": "BF16",
    "float32": "F32", "float": "F32", "float64": "F64", "double": "F64",
}  # fmt: skip


def _lookup(cfg: dict[str, Any], keys: Iterable[str]) -> Any:
    sections = [cfg] + [cfg[n] for n in NESTED if isinstance(cfg.get(n), dict)]
    for section in sections:
        for key in keys:
            if key in section:
                return section[key]
    return None


def layer_families(names: Iterable[str]) -> dict[str, set[int]]:
    found: dict[str, set[int]] = {}
    for name in names:
        for family, rx in _FAMILY_RES:
            m = rx.search(name)
            if m:
                found.setdefault(family, set()).add(int(m.group(1)))
                break
    if not found:
        # Canonical GPT-2 names layers "h.N.*" without a prefix; only trusted when nothing else matched.
        for name in names:
            m = _BARE_H_RE.match(name)
            if m:
                found.setdefault("h", set()).add(int(m.group(1)))
    return found


def check_config(
    config_path: Path,
    names: set[str],
    tensors: dict[str, TensorEntry],
    complete: bool,
    require_config: bool,
    emit: Emit,
) -> None:
    """``names`` are all tensor names known (headers + index); ``complete`` says they are exhaustive."""
    if not config_path.exists():
        if require_config:
            emit("CD305", {}, path=config_path)
        return
    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(cfg, dict):
            raise ValueError("config.json is not a JSON object")
    except (OSError, ValueError) as exc:
        emit("CD306", {"detail": str(exc)}, path=config_path)
        return

    if complete and names:
        _check_layers(cfg, config_path, names, emit)
        _check_tie(cfg, config_path, names, emit)
    _check_dtype(cfg, config_path, tensors, emit)


def _check_layers(cfg: dict[str, Any], path: Path, names: set[str], emit: Emit) -> None:
    expected = _lookup(cfg, LAYER_KEYS)
    if not isinstance(expected, int) or isinstance(expected, bool):
        return
    families = layer_families(names)
    if not families:
        emit("CD390", {"expected": expected}, path=path)
        return
    counts: dict[str, int] = {}
    for family, idx in families.items():
        counts[family] = max(idx) + 1
        if len(idx) != counts[family]:
            emit("CD304", {"family": family, "distinct": len(idx), "highest": max(idx)}, path=path)
    if expected in counts.values():
        return
    family = max(counts, key=lambda f: counts[f])
    severity = Severity.ERROR if len(counts) == 1 else Severity.WARNING
    emit("CD301", {"expected": expected, "actual": counts[family], "family": family}, path=path, severity=severity)


def _check_tie(cfg: dict[str, Any], path: Path, names: set[str], emit: Emit) -> None:
    tie = _lookup(cfg, ("tie_word_embeddings",))
    archs = cfg.get("architectures")
    has_head = not archs or (isinstance(archs, list) and any(HEAD_ARCH.search(str(a)) for a in archs))
    has_out_head = any(n == h or n.endswith("." + h) for n in names for h in HEAD_NAMES)
    has_lm_head = any(n == "lm_head.weight" or n.endswith(".lm_head.weight") for n in names)
    if tie is False and has_head and not has_out_head:
        emit("CD302", {}, path=path)
    elif tie is True and has_lm_head:
        emit("CD391", {}, path=path)


def _check_dtype(cfg: dict[str, Any], path: Path, tensors: dict[str, TensorEntry], emit: Emit) -> None:
    raw = _lookup(cfg, ("dtype", "torch_dtype"))
    if not isinstance(raw, str):
        return
    expected = DTYPE_NAMES.get(raw.lower().removeprefix("torch."))
    if expected is None:
        return
    counts: dict[str, int] = {}
    for entry in tensors.values():
        if entry.dtype in FLOATS:
            counts[entry.dtype] = counts.get(entry.dtype, 0) + 1
    total = sum(counts.values())
    if not total:
        return
    main = max(counts, key=lambda d: counts[d])
    if counts[main] / total >= 0.5 and main != expected:
        emit("CD303", {"declared": raw, "expected": expected, "actual": main}, path=path)
