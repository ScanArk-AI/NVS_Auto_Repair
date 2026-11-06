"""Windows-facing details of the repair side that can be checked on Linux."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def test_attention_backend_windows_avoids_fa3():
    from repair_pipeline.model import default_attention_backend

    assert default_attention_backend("_flash_3", platform="win32") == "native"
    assert default_attention_backend(None, platform="win32") == "native"
    assert default_attention_backend("_flash_3", platform="linux") == "_flash_3"
    assert default_attention_backend(None, platform="linux") is None


@pytest.mark.parametrize("error", [TypeError, RuntimeError, OSError])
def test_load_state_dict_falls_back_without_mmap(monkeypatch, error):
    torch = pytest.importorskip("torch")
    from repair_pipeline import model

    calls = []

    def fake_load(path, **kw):
        calls.append(kw.get("mmap"))
        if kw.get("mmap"):
            raise error("cannot mmap")
        return {"w": torch.zeros(1)}

    monkeypatch.setattr(torch, "load", fake_load)
    loaded = []
    model._load_state_dict(SimpleNamespace(load_state_dict=loaded.append), Path("x.pt"))
    assert calls == [True, None] and list(loaded[0]) == ["w"]


def test_ensure_ninja_finds_scripts_dir(monkeypatch, tmp_path):
    from repair_pipeline import render_backend

    scripts = tmp_path / "Scripts"
    scripts.mkdir()
    exe = scripts / "ninja"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "python"))
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    render_backend._ensure_ninja()
    assert os.environ["PATH"].split(os.pathsep)[0] == str(scripts)


def test_save_frames_explains_locked_file(monkeypatch, tmp_path):
    from repair_pipeline.frames import save_frames

    (tmp_path / "00000.png").write_bytes(b"")

    def locked(self, missing_ok=False):
        raise PermissionError(13, "The process cannot access the file")

    monkeypatch.setattr(Path, "unlink", locked)
    with pytest.raises(PermissionError, match="in use"):
        save_frames(np.zeros((1, 2, 2, 3), np.uint8), tmp_path)
