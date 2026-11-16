"""Synthetic practice checkpoints: 1 clean + 13 broken ones, each with an ``expected.json``.

Everything is written with the standard library only. The tensors are tiny, deterministic filler
bytes, NOT real model weights. The faults copy real incidents: LFS pointer text, an HTML error page,
cut-off downloads, a stale shard index, a config that disagrees with the weights, and a damaged
Hugging Face cache.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Callable

from .safetensors_header import DTYPE_BITS

Tensor = tuple[str, str, list[int]]  # name, dtype, shape
TOTAL_SHARDS = 4
SAMPLE_NAMES = [
    "clean", "lfs-pointer", "html-page", "tiny-shard", "truncated", "index-extra", "index-dropped",
    "duplicate-tensor", "missing-shard", "no-index", "config-layers", "untied-no-lmhead", "mixed",
    "hf-cache",
]  # fmt: skip
COMMIT = "0123456789abcdef0123456789abcdef01234567"
GHOST_COMMIT = "fedcba9876543210fedcba9876543210fedcba98"


def nbytes(dtype: str, shape: list[int]) -> int:
    return math.ceil(math.prod(shape) * DTYPE_BITS[dtype] / 8)


def filler(name: str, n: int) -> bytes:
    """Deterministic non-zero filler bytes for a tensor."""
    seed = hashlib.sha256(name.encode()).digest()
    return bytes((seed[i % 32] + i) % 251 + 1 for i in range(n))


def safetensors_bytes(tensors: list[Tensor], metadata: dict[str, str] | None = None) -> bytes:
    """Serialise a valid safetensors file (header padded with spaces to 8 bytes, like the library)."""
    header: dict[str, Any] = {}
    if metadata is not None:
        header["__metadata__"] = metadata
    data = b""
    for name, dtype, shape in tensors:
        chunk = filler(name, nbytes(dtype, shape))
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [len(data), len(data) + len(chunk)]}
        data += chunk
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    return len(raw).to_bytes(8, "little") + raw + data


# ---- the clean checkpoint ----------------------------------------------------------------------


def _layer(i: int) -> list[Tensor]:
    return [
        (f"model.layers.{i}.input_layernorm.weight", "F16", [4]),
        (f"model.layers.{i}.self_attn.q_proj.weight", "F16", [4, 4]),
    ]


def shard_names() -> list[str]:
    return [f"model-{i:05d}-of-{TOTAL_SHARDS:05d}.safetensors" for i in range(1, TOTAL_SHARDS + 1)]


def clean_layout() -> dict[str, list[Tensor]]:
    names = shard_names()
    return {
        names[0]: [("model.embed_tokens.weight", "F16", [8, 4])],
        names[1]: _layer(0),
        names[2]: _layer(1),
        names[3]: [("model.norm.weight", "F16", [4]), ("lm_head.weight", "F16", [8, 4])],
    }


def make_index(layout: dict[str, list[Tensor]]) -> dict[str, Any]:
    weight_map = {t[0]: shard for shard, tensors in layout.items() for t in tensors}
    total = sum(nbytes(t[1], t[2]) for tensors in layout.values() for t in tensors)
    return {"metadata": {"total_size": total}, "weight_map": weight_map}


def make_config(layers: int = 2, tie: bool = False) -> dict[str, Any]:
    return {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "num_hidden_layers": layers,
        "hidden_size": 4,
        "vocab_size": 8,
        "tie_word_embeddings": tie,
        "torch_dtype": "float16",
    }


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def write_folder(
    folder: Path,
    layout: dict[str, list[Tensor]],
    index: dict[str, Any] | None,
    config: dict[str, Any] | None,
) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for shard, tensors in layout.items():
        (folder / shard).write_bytes(safetensors_bytes(tensors, {"format": "pt"}))
    if index is not None:
        write_json(folder / "model.safetensors.index.json", index)
    if config is not None:
        write_json(folder / "config.json", config)


def write_clean(folder: Path) -> None:
    layout = clean_layout()
    write_folder(folder, layout, make_index(layout), make_config())


def lfs_pointer_text(name: str, size: int = 4_981_234_567) -> bytes:
    oid = hashlib.sha256(name.encode()).hexdigest()
    return f"version https://git-lfs.github.com/spec/v1\noid sha256:{oid}\nsize {size}\n".encode()


HTML_PAGE = (
    b"<!DOCTYPE html>\n<html><head><title>401 Unauthorized</title></head>\n"
    b"<body><h1>Invalid credentials in Authorization header</h1></body></html>\n"
)


# ---- the 13 broken samples ----------------------------------------------------------------------


def _s_lfs(d: Path) -> list[str]:
    write_clean(d)
    (d / shard_names()[1]).write_bytes(lfs_pointer_text(shard_names()[1]))
    return ["CD101"]


def _s_html(d: Path) -> list[str]:
    write_clean(d)
    (d / shard_names()[2]).write_bytes(HTML_PAGE)
    return ["CD102"]


def _s_tiny(d: Path) -> list[str]:
    write_clean(d)
    (d / shard_names()[1]).write_bytes(b"\x00" * 16)
    return ["CD103"]


def _s_truncated(d: Path) -> list[str]:
    write_clean(d)
    p = d / shard_names()[3]
    p.write_bytes(p.read_bytes()[:-64])
    return ["CD110"]


def _s_index_extra(d: Path) -> list[str]:
    write_clean(d)
    index = make_index(clean_layout())
    index["weight_map"]["model.rotary_emb.inv_freq"] = shard_names()[3]
    write_json(d / "model.safetensors.index.json", index)
    return ["CD203"]


def _s_index_dropped(d: Path) -> list[str]:
    write_clean(d)
    index = make_index(clean_layout())
    del index["weight_map"]["model.norm.weight"]
    write_json(d / "model.safetensors.index.json", index)
    return ["CD204"]


def _s_duplicate(d: Path) -> list[str]:
    layout = clean_layout()
    index = make_index(layout)
    layout[shard_names()[2]] = layout[shard_names()[2]] + [("model.norm.weight", "F16", [4])]
    write_folder(d, layout, index, make_config())
    return ["CD206"]


def _s_missing_shard(d: Path) -> list[str]:
    write_clean(d)
    (d / shard_names()[2]).unlink()
    return ["CD202"]


def _s_no_index(d: Path) -> list[str]:
    write_clean(d)
    (d / "model.safetensors.index.json").unlink()
    (d / shard_names()[2]).unlink()
    return ["CD201"]


def _s_config_layers(d: Path) -> list[str]:
    layout = clean_layout()
    write_folder(d, layout, make_index(layout), make_config(layers=4))
    return ["CD301"]


def _s_untied(d: Path) -> list[str]:
    layout = clean_layout()
    layout[shard_names()[3]] = [("model.norm.weight", "F16", [4])]
    write_folder(d, layout, make_index(layout), make_config(tie=False))
    return ["CD302"]


def _s_mixed(d: Path) -> list[str]:
    layout = clean_layout()
    write_folder(d, layout, None, make_config(layers=4))
    index = make_index(layout)
    index["weight_map"]["model.rotary_emb.inv_freq"] = shard_names()[3]
    write_json(d / "model.safetensors.index.json", index)
    (d / shard_names()[1]).write_bytes(lfs_pointer_text(shard_names()[1]))
    (d / shard_names()[2]).write_bytes(HTML_PAGE)
    return ["CD101", "CD102", "CD203", "CD301"]


def _s_hf_cache(d: Path) -> list[str]:
    """A hub cache with a flipped data byte, an unfinished download, a broken link and a stale ref."""
    repo = d / "models--demo--tiny"
    snap = repo / "snapshots" / COMMIT
    (repo / "blobs").mkdir(parents=True)
    snap.mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "trees").mkdir()
    layout = [t for tensors in clean_layout().values() for t in tensors]
    weights = safetensors_bytes(layout, {"format": "pt"})
    weights_sha = hashlib.sha256(weights).hexdigest()
    damaged = bytearray(weights)
    damaged[-1] ^= 0xFF  # data region only: the header stays valid
    config = (json.dumps(make_config()) + "\n").encode()
    blobs = {"model.safetensors": (weights_sha, bytes(damaged)), "config.json": ("a" * 40, config)}
    for name, (blob, content) in blobs.items():
        (repo / "blobs" / blob).write_bytes(content)
        os.symlink(f"../../blobs/{blob}", snap / name)
    os.symlink(f"../../blobs/{'b' * 40}", snap / "tokenizer.json")  # blob never arrived
    (repo / "blobs" / f"{'c' * 64}.incomplete").write_bytes(b"\x00" * 128)
    (repo / "refs" / "main").write_text(COMMIT)
    (repo / "refs" / "dev").write_text(GHOST_COMMIT)
    tree = {
        "format_version": 1,
        "files": {
            "config.json": {"size": len(config) + 11, "blob_id": "a" * 40},
            "model.safetensors": {
                "size": len(weights), "blob_id": "d" * 40, "lfs_sha256": weights_sha, "lfs_size": len(weights),
            },
            "tokenizer.json": {"size": 1234, "blob_id": "b" * 40},
        },
    }  # fmt: skip
    write_json(repo / "trees" / f"{COMMIT}.json", tree)
    return ["CD401", "CD402", "CD403", "CD405", "CD501"]


BUILDERS: dict[str, Callable[[Path], list[str]]] = {
    "lfs-pointer": _s_lfs,
    "html-page": _s_html,
    "tiny-shard": _s_tiny,
    "truncated": _s_truncated,
    "index-extra": _s_index_extra,
    "index-dropped": _s_index_dropped,
    "duplicate-tensor": _s_duplicate,
    "missing-shard": _s_missing_shard,
    "no-index": _s_no_index,
    "config-layers": _s_config_layers,
    "untied-no-lmhead": _s_untied,
    "mixed": _s_mixed,
    "hf-cache": _s_hf_cache,
}


def write_samples(dest: str | os.PathLike[str]) -> list[str]:
    """Create all sample folders under ``dest`` (existing sample folders are replaced)."""
    root = Path(dest)
    root.mkdir(parents=True, exist_ok=True)
    for name in SAMPLE_NAMES:
        folder = root / name
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir()
        if name == "clean":
            write_clean(folder)
            codes: list[str] = []
        else:
            codes = BUILDERS[name](folder)
        check: dict[str, Any] = {"path": ".", "hash": False}
        if name == "hf-cache":
            check = {"path": "demo/tiny", "cache_dir": ".", "hash": True}
        write_json(folder / "expected.json", {"codes": sorted(codes), "check": check})
    return list(SAMPLE_NAMES)


def load_expected(folder: Path) -> dict[str, Any]:
    return json.loads((folder / "expected.json").read_text(encoding="utf-8"))
