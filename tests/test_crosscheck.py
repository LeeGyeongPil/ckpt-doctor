"""AC-10: agree with the official safetensors library (needs: uv run --group crosscheck pytest)."""

from __future__ import annotations

from pathlib import Path

import pytest

safetensors = pytest.importorskip("safetensors")
np = pytest.importorskip("numpy")
from safetensors.numpy import save_file  # noqa: E402

from ckpt_doctor import check_path  # noqa: E402
from ckpt_doctor.sample import HTML_PAGE, lfs_pointer_text  # noqa: E402


def _good(path: Path) -> bytes:
    tensors = {
        "w": np.arange(12, dtype=np.float32).reshape(3, 4),
        "b": np.ones(5, dtype=np.float16),
        "ids": np.arange(4, dtype=np.int64),
    }
    save_file(tensors, str(path), metadata={"format": "pt"})
    return path.read_bytes()


def _raw(header: str, data: bytes) -> bytes:
    raw = header.encode()
    return len(raw).to_bytes(8, "little") + raw + data


ENTRY = '{"dtype":"U8","shape":[4],"data_offsets":[%d,%d]}'


def _faults(good: bytes) -> dict[str, bytes]:
    return {
        "lfs-pointer": lfs_pointer_text("m.safetensors"),
        "html": HTML_PAGE,
        "zeros16": b"\x00" * 16,
        "huge-length": (200_000_000).to_bytes(8, "little") + b"x" * 64,
        "truncated": good[:-9],
        "trailing": good + b"junk",
        "duplicate-key": _raw('{"a":%s,"a":%s}' % (ENTRY % (0, 4), ENTRY % (0, 4)), b"\x01" * 4),
        "hole": _raw('{"a":%s,"b":%s}' % (ENTRY % (0, 4), ENTRY % (8, 12)), b"\x01" * 12),
        "cut-in-header": good[:30],
    }


def _library_raises(path: Path) -> str | None:
    from safetensors import safe_open

    try:
        with safe_open(str(path), framework="numpy") as f:
            list(f.keys())
    except Exception as exc:  # noqa: BLE001 - any failure counts
        return str(exc)
    return None


def test_library_written_file_has_no_findings(tmp_path: Path) -> None:
    path = tmp_path / "model.safetensors"
    _good(path)
    assert _library_raises(path) is None
    assert check_path(path).findings == []


@pytest.mark.parametrize("name", sorted(set(_faults(b"x" * 100)) - {"duplicate-key"}))
def test_error_iff_library_raises(tmp_path: Path, name: str) -> None:
    good = _good(tmp_path / "good.safetensors")
    path = tmp_path / "model.safetensors"
    path.write_bytes(_faults(good)[name])
    library_error = _library_raises(path)
    ours = check_path(path)
    assert library_error is not None, f"library accepted {name}"
    assert not ours.ok, f"ckpt-doctor missed {name}: {library_error}"


def test_same_library_message_distinct_codes(tmp_path: Path) -> None:
    pointer, html = tmp_path / "p.safetensors", tmp_path / "h.safetensors"
    pointer.write_bytes(lfs_pointer_text("m"))
    html.write_bytes(HTML_PAGE)
    assert _library_raises(pointer) == _library_raises(html)  # the library cannot tell them apart
    assert check_path(pointer).codes() == {"CD101"}
    assert check_path(html).codes() == {"CD102"}


def test_duplicate_key_is_stricter_than_library_0_8_0(tmp_path: Path) -> None:
    """Documented divergence: the format spec forbids duplicate keys, 0.8.0 silently accepts them."""
    path = tmp_path / "dup.safetensors"
    path.write_bytes(_faults(b"x" * 100)["duplicate-key"])
    assert _library_raises(path) is None
    assert check_path(path).codes() == {"CD106"}
