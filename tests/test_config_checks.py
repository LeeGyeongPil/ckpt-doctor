"""AC-4: config.json versus tensors."""

from __future__ import annotations

import json
from pathlib import Path

from ckpt_doctor import check_path
from ckpt_doctor.sample import Tensor, make_config, safetensors_bytes


def _folder(tmp_path: Path, tensors: list[Tensor], config: dict) -> Path:
    (tmp_path / "model.safetensors").write_bytes(safetensors_bytes(tensors))
    (tmp_path / "config.json").write_text(json.dumps(config))
    return tmp_path


def _llama(layers: int, head: bool = True) -> list[Tensor]:
    out: list[Tensor] = [("model.embed_tokens.weight", "F16", [4, 2])]
    for i in range(layers):
        out.append((f"model.layers.{i}.mlp.weight", "F16", [2, 2]))
    if head:
        out.append(("lm_head.weight", "F16", [4, 2]))
    return out


def test_layer_count_match_is_clean(tmp_path: Path) -> None:
    assert check_path(_folder(tmp_path, _llama(2), make_config(layers=2))).findings == []


def test_layer_mismatch_llama_bert_gpt2(tmp_path: Path) -> None:
    patterns = {
        "llama": "model.layers.{i}.w",
        "bert": "bert.encoder.layer.{i}.w",
        "gpt2": "transformer.h.{i}.w",
    }
    for kind, pattern in patterns.items():
        folder = tmp_path / kind
        folder.mkdir()
        tensors: list[Tensor] = [(pattern.format(i=i), "F16", [2]) for i in range(2)]
        config = {"num_hidden_layers": 4, "torch_dtype": "float16"}
        report = check_path(_folder(folder, tensors, config))
        assert report.codes() == {"CD301"}, kind
        assert report.findings[0].params["actual"] == 2


def test_untied_without_lm_head_is_error(tmp_path: Path) -> None:
    config = make_config(layers=1, tie=False)
    report = check_path(_folder(tmp_path, _llama(1, head=False), config))
    assert report.codes() == {"CD302"}


def test_tied_without_lm_head_is_fine(tmp_path: Path) -> None:
    config = make_config(layers=1, tie=True)
    assert check_path(_folder(tmp_path, _llama(1, head=False), config)).findings == []


def test_tied_with_lm_head_is_info(tmp_path: Path) -> None:
    config = make_config(layers=1, tie=True)
    report = check_path(_folder(tmp_path, _llama(1), config))
    assert report.codes() == {"CD391"}
    assert report.exit_code(strict=True) == 0


def test_int_buffers_do_not_trigger_dtype_warning(tmp_path: Path) -> None:
    tensors: list[Tensor] = [
        ("bert.embeddings.position_ids", "I64", [1, 4]),
        ("bert.encoder.layer.0.w", "F32", [2]),
        ("bert.encoder.layer.0.b", "F32", [2]),
    ]
    config = {"num_hidden_layers": 1, "torch_dtype": "float32"}
    assert check_path(_folder(tmp_path, tensors, config)).findings == []


def test_dtype_mismatch_reads_both_keys(tmp_path: Path) -> None:
    tensors: list[Tensor] = [("model.layers.0.w", "F32", [2]), ("model.layers.0.b", "F32", [2])]
    for key in ("dtype", "torch_dtype"):
        folder = tmp_path / key
        folder.mkdir()
        config = {"num_hidden_layers": 1, key: "bfloat16"}
        report = check_path(_folder(folder, tensors, config))
        assert report.codes() == {"CD303"}, key
        assert report.exit_code() == 0  # warning only


def test_nested_text_config_is_read(tmp_path: Path) -> None:
    config = {"architectures": ["Foo"], "text_config": {"num_hidden_layers": 3}}
    tensors: list[Tensor] = [(f"language_model.model.layers.{i}.w", "F16", [2]) for i in range(2)]
    assert check_path(_folder(tmp_path, tensors, config)).codes() == {"CD301"}


def test_unknown_naming_is_info_only(tmp_path: Path) -> None:
    tensors: list[Tensor] = [("encoder.block.0.w", "F16", [2]), ("decoder.block.0.w", "F16", [2])]
    report = check_path(_folder(tmp_path, tensors, {"num_hidden_layers": 12}))
    assert report.codes() == {"CD390"}
    assert report.exit_code(strict=True) == 0


def test_layer_gap_warns(tmp_path: Path) -> None:
    tensors: list[Tensor] = [("model.layers.0.w", "F16", [2]), ("model.layers.2.w", "F16", [2])]
    report = check_path(_folder(tmp_path, tensors, {"num_hidden_layers": 3}))
    assert report.codes() == {"CD304"}


def test_missing_and_broken_config(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors").write_bytes(safetensors_bytes(_llama(1)))
    assert check_path(tmp_path).codes() == {"CD305"}
    (tmp_path / "config.json").write_text("{bad")
    assert check_path(tmp_path).codes() == {"CD306"}


def test_config_checks_skipped_when_tensor_set_incomplete(samples: Path) -> None:
    # the no-index sample misses a shard; comparing layer counts would be a false alarm
    assert "CD301" not in check_path(samples / "no-index").codes()


def test_gpt_neox_embed_out_head_is_not_flagged(tmp_path: Path) -> None:
    tensors: list[Tensor] = [("gpt_neox.embed_in.weight", "F16", [4, 2])]
    tensors += [(f"gpt_neox.layers.{i}.mlp.weight", "F16", [2, 2]) for i in range(2)]
    tensors.append(("embed_out.weight", "F16", [4, 2]))
    config = make_config(layers=2)
    config.update({"tie_word_embeddings": False, "architectures": ["GPTNeoXForCausalLM"]})
    report = check_path(_folder(tmp_path, tensors, config))
    assert "CD302" not in {f.code for f in report.findings}


def test_bare_h_layer_names_are_counted(tmp_path: Path) -> None:
    tensors: list[Tensor] = [(f"h.{i}.attn.weight", "F16", [2]) for i in range(3)]
    report = check_path(_folder(tmp_path, tensors, make_config(layers=2)))
    codes = {f.code for f in report.findings}
    assert "CD301" in codes and "CD390" not in codes
