"""Shared fixtures. All checkpoints are synthetic and written with the standard library."""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

from ckpt_doctor.sample import write_clean, write_samples


@pytest.fixture
def clean(tmp_path: Path) -> Path:
    folder = tmp_path / "clean"
    write_clean(folder)
    return folder


@pytest.fixture(scope="session")
def samples(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("samples")
    write_samples(root)
    return root


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if anything tries to open a socket."""

    def blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "socket", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
