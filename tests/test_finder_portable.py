"""The finder on Windows: no hard-coded Linux font, no locale-dependent text I/O."""

import ast
from pathlib import Path

import numpy as np
import pytest

FINDER = Path(__file__).resolve().parents[1] / "subprojects" / "repair_view_finder"


def _text_io_without_encoding(path):
    """(line, call) for every text-mode open()/read_text()/write_text() that
    leaves the encoding to the locale (cp1252 / cp936 on Windows)."""
    bad = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""
        kw = {k.arg for k in node.keywords}
        if name in ("read_text", "write_text") and "encoding" not in kw:
            bad.append((node.lineno, name))
        elif name == "open" and isinstance(f, ast.Name) and "encoding" not in kw:
            mode = node.args[1] if len(node.args) > 1 else next(
                (k.value for k in node.keywords if k.arg == "mode"), None)
            if not (isinstance(mode, ast.Constant) and "b" in str(mode.value)):
                bad.append((node.lineno, name))
    return bad


@pytest.mark.parametrize("path", sorted(FINDER.rglob("*.py")), ids=lambda p: str(p.relative_to(FINDER)))
def test_finder_text_io_is_utf8(path):
    assert _text_io_without_encoding(path) == []


def test_text_io_check_catches_a_bare_open(tmp_path):
    src = tmp_path / "x.py"
    src.write_text("open(p)\nopen(p, 'rb')\nopen(p, 'w', encoding='utf-8')\nq.write_text(s)\n",
                   encoding="utf-8")
    assert _text_io_without_encoding(src) == [(1, "open"), (4, "write_text")]


@pytest.fixture
def viz(monkeypatch):
    pytest.importorskip("matplotlib")      # viz.py needs the finder env
    from repair_view_finder import viz
    monkeypatch.setattr(viz, "_font_warned", False)
    monkeypatch.delenv("NVS_REPAIR_FONT", raising=False)
    return viz


def test_no_cjk_font_falls_back_instead_of_raising(viz, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(viz, "CJK_FONTS", ((str(tmp_path / "missing.ttc"), 2),))
    rgb = np.zeros((18, 32, 3), np.uint8)
    out = tmp_path / "top.png"
    viz.overview_image([("#1  camera_0", rgb), ("#2  相机", rgb)], out,
                       title="最该修的2个视角", subtitle="场景 · GAVIS UQ ranking")
    assert out.is_file()
    assert "no CJK font found" in capsys.readouterr().out


def test_bitmap_fallback_font_gets_latin1_text(viz):
    from PIL import ImageFont
    bitmap = ImageFont.ImageFont()
    assert viz._drawable("最该修 #1", bitmap) == "??? #1"


def test_font_env_var_comes_first(viz, monkeypatch, tmp_path):
    tried = []

    def truetype(path, size, index=0, **kw):
        tried.append((path, index))
        raise OSError(path)

    monkeypatch.setattr(viz.ImageFont, "truetype", truetype)
    monkeypatch.setattr(viz.ImageFont, "load_default", lambda *a: "default")
    monkeypatch.setenv("NVS_REPAIR_FONT", r"D:\fonts\my.ttc#3")
    assert viz.cjk_font(12) == "default"
    assert tried[0] == (r"D:\fonts\my.ttc", 3)
    assert (r"C:\Windows\Fonts\msyh.ttc", 0) in tried
    assert tried[1:] == list(viz.CJK_FONTS)
