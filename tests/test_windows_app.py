"""The GUI's logic (settings, form -> Options, mirrors) and a window smoke test.

The window test builds the real ``App`` with fake setup / runner / WSL, pumps
the event loop until the background status check has landed, then destroys
it. It is skipped where no display is available.
"""

import json
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from nvs_repair_gui import __version__, app, runner, setup


# --- settings ----------------------------------------------------------------
def test_settings_round_trip(tmp_path):
    s = app.Settings(scene=r"C:\data\my scene", out=r"D:\out", preset="object-centric-indoor",
                     units_mode="number", units_value="3.5", caption='a "room": white walls',
                     mode="repair", limit="4", copy_to_wsl=True, gh_proxy="https://m/", hf_endpoint="https://h",
                     hf_disable_xet=True)
    path = app.save_settings(s, tmp_path / "nvs_repair" / "settings.json")
    assert app.load_settings(path) == s
    assert json.loads(path.read_text(encoding="utf-8"))["caption"] == 'a "room": white walls'


@pytest.mark.parametrize("content", ["", "not json", "[1, 2]", '{"preset": "nope", "mode": 3, "limit": 5}'])
def test_bad_settings_fall_back_to_defaults(tmp_path, content):
    p = tmp_path / "settings.json"
    p.write_text(content, encoding="utf-8")
    s = app.load_settings(p)
    assert s.preset == "default" and s.mode == "check" and s.limit == ""


def test_missing_settings_file(tmp_path):
    assert app.load_settings(tmp_path / "absent.json") == app.Settings()


def test_settings_path_per_platform(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("APPDATA", str(tmp_path))
    assert app.settings_path() == tmp_path / "nvs_repair" / "settings.json"
    monkeypatch.setattr(sys, "platform", "linux")
    assert app.settings_path().parts[-3:] == (".config", "nvs_repair", "settings.json")


# --- form -> Options ------------------------------------------------------------
@pytest.mark.parametrize("scene, out", [
    (r"C:\data\my_scene", r"C:\data\my_scene_nvs_repair"),
    ("C:\\data\\my scene\\", r"C:\data\my scene_nvs_repair"),
    ("/home/u/scenes/picnic", "/home/u/scenes/picnic_nvs_repair"),
    ("", ""),
])
def test_default_out(scene, out):
    assert app.default_out(scene) == out


@pytest.mark.parametrize("mode, value, expected", [
    ("auto", "", "auto"), ("metric", "", "1"), ("number", "1.674", "1.674"), ("number", " 3.5 ", "3.5")])
def test_units(mode, value, expected):
    assert app.to_options(app.Settings(scene="C:\\s", units_mode=mode, units_value=value)).units == expected


@pytest.mark.parametrize("settings, message", [
    (dict(scene=""), "scene directory"),
    (dict(scene="C:\\s", units_mode="number", units_value="abc"), "must be a number"),
    (dict(scene="C:\\s", units_mode="number", units_value="0"), "greater than 0"),
    (dict(scene="C:\\s", limit="x"), "whole number"),
    (dict(scene="C:\\s", limit="0"), "at least 1"),
    (dict(scene="C:\\s", preset="bogus"), "capture type"),
])
def test_bad_forms(settings, message):
    with pytest.raises(ValueError, match=message):
        app.to_options(app.Settings(**settings))


def test_to_options_full():
    o = app.to_options(app.Settings(scene=" C:\\data\\s ", preset="object-centric", caption="  a room ",
                                    mode="repair", limit="3", copy_to_wsl=True))
    assert isinstance(o, runner.Options)
    assert (o.scene, o.out, o.preset, o.units, o.caption, o.mode, o.limit, o.copy_to_wsl) == (
        "C:\\data\\s", "C:\\data\\s_nvs_repair", "object-centric", "auto", "a room", "repair", 3, True)
    assert app.to_options(app.Settings(scene="C:\\s", out="D:\\o")).limit is None


def test_every_preset_and_mode_has_a_label():
    assert set(app.PRESET_LABELS) == set(runner.PRESETS)
    assert set(app.MODE_LABELS) == set(runner.MODES)


def test_mirrors():
    assert app.mirrors(app.Settings()) == {}
    assert app.mirrors(app.Settings(gh_proxy=" https://g/ ", hf_endpoint="https://h", hf_disable_xet=True)) == {
        "GH_PROXY": "https://g/", "HF_ENDPOINT": "https://h", "HF_HUB_DISABLE_XET": "1"}


def test_source_dir(monkeypatch, tmp_path):
    assert (app.source_dir() / "run.py").is_file()          # repo root when run from source
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "nvs_repair.exe"))
    assert app.source_dir() == tmp_path / "source"


def test_status_text():
    st = setup.Status(wsl=True, distro=True, wsl2=True, user=True, gpu="RTX 4090, 24564 MiB", installed=False)
    assert "GPU: RTX 4090" in app.status_text(st) and "installed missing" in app.status_text(st)


def test_stage_text():
    assert set(app.STAGE_LABELS) == set(runner.STAGES)
    assert app.stage_text(runner.Progress("repair", 3, 22)) == "Repairing 3/22"
    assert app.stage_text(runner.Progress("done")) == "Done"
    assert app.stage_text(runner.Progress("prepare", message="caption")) == "Preparing: caption"


def _native_status(**kw):
    base = dict(driver="572.83", driver_ok=True, gpu="NVIDIA GeForce RTX 4090, 24564 MiB", vcredist=True,
                runtime=True, gavis=False, weights=False, hints=[], installed=False, ready=False)
    return SimpleNamespace(**{**base, **kw})


def test_status_text_native():
    t = app.status_text(_native_status())
    assert "driver 572.83 OK" in t and "VC++ runtime OK" in t and "GAVIS missing" in t and "WSL" not in t
    assert "driver missing" in app.status_text(_native_status(driver="", driver_ok=False))
    assert "too old" in app.status_text(_native_status(driver="551.00", driver_ok=False))


# --- backend choice -----------------------------------------------------------
def _bundle(tmp_path):
    (tmp_path / "runtime").mkdir(parents=True)
    (tmp_path / "runtime" / "python.exe").write_bytes(b"")
    (tmp_path / "runtime" / "pythonw.exe").write_bytes(b"")
    return tmp_path


def test_bundle_root(tmp_path, monkeypatch):
    root = _bundle(tmp_path / "bundle")
    assert app.bundle_root(str(root / "runtime" / "pythonw.exe")) == root.resolve()     # launcher -> pythonw
    assert app.bundle_root(str(root / "nvs_repair.exe")) == root.resolve()
    assert app.bundle_root(str(tmp_path / "venv" / "bin" / "python")) is None
    monkeypatch.setattr(sys, "executable", str(root / "runtime" / "pythonw.exe"))
    assert app.source_dir() == root.resolve() / "app"


def test_make_backend_in_bundle():
    from pathlib import PureWindowsPath

    from nvs_repair_gui import native, native_setup

    executor, setup_module = app.make_backend(PureWindowsPath(r"C:\nvs_repair"))
    assert isinstance(executor, native.Native) and setup_module is native_setup
    assert executor.layout.python == r"C:\nvs_repair\runtime\python.exe" and executor.layout.scenes is None


def test_module_entry_point():
    import importlib.util

    assert importlib.util.find_spec("nvs_repair_gui.__main__") is not None


def test_make_backend_without_bundle(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "executable", str(tmp_path / "python"))
    executor, setup_module = app.make_backend()
    assert executor.layout is runner.WSL_LAYOUT and setup_module is setup


# --- launch.py ------------------------------------------------------------------
def test_launch_version(tmp_path, capsys):
    import launch

    assert launch.main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == __version__
    out = tmp_path / "v.txt"
    assert launch.main(["--version-file", str(out)]) == 0
    assert out.read_text().strip() == __version__


# --- the window ---------------------------------------------------------------
class FakeWSL:
    layout = runner.WSL_LAYOUT

    def __init__(self):
        self.stopped = threading.Event()

    def stop(self):
        self.stopped.set()


class FakeSetup:
    Status = setup.Status

    def __init__(self):
        self.installs = []

    def status(self, wsl):
        return setup.Status(wsl=True, distro=True, wsl2=True, user=True, gpu="Fake GPU, 24576 MiB",
                            installed=True, weights=False, hints=["Open Ubuntu once to create your user"])

    def install(self, wsl, source_dir, on_line, with_weights=True, mirrors=None):
        self.installs.append((source_dir, with_weights, mirrors))
        on_line("installing")
        return 0


class FakeRunner:
    PRESETS, MODES, Options, Progress = runner.PRESETS, runner.MODES, runner.Options, runner.Progress

    def __init__(self):
        self.runs = []

    def run(self, opts, wsl, on_line, on_progress):
        self.runs.append(opts)
        on_line("=== repair:repair")
        on_progress(runner.Progress("repair", 1, 2))
        on_progress(runner.Progress("done"))
        return 0


def _tk_root():
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"no display: {exc}")
    root.withdraw()
    return root


def _pump(root, cond, timeout=5.0):
    end = time.time() + timeout
    while not cond() and time.time() < end:
        root.update()
        time.sleep(0.02)
    return cond()


def test_window_smoke(tmp_path):
    root = _tk_root()
    fs, fr, fw = FakeSetup(), FakeRunner(), FakeWSL()
    try:
        a = app.App(root, setup=fs, runner=fr, executor=fw, settings_file=tmp_path / "s.json")
        assert _pump(root, lambda: a.status is not None and not a.busy)
        assert "Fake GPU" in a.status_label.cget("text")

        a.v["scene"].set(r"C:\data\room")
        assert a.v["out"].get() == r"C:\data\room_nvs_repair"          # follows the scene until edited
        a.v["mode"].set("check")
        a.v["limit"].set("2")
        a.on_run()
        assert _pump(root, lambda: fr.runs and not a.busy)
        assert fr.runs[0].limit == 2 and fr.runs[0].mode == "check"
        assert a.stage.cget("text") == "Done"
        assert "=== repair:repair" in a.log.get("1.0", "end")

        a.on_install()
        assert _pump(root, lambda: fs.installs and not a.busy)
        assert fs.installs[0][1] is False                               # check mode: no weights
        assert (tmp_path / "s.json").is_file()
    finally:
        root.destroy()


def test_small_gpu_hint():
    small = setup.Status(gpu="NVIDIA GeForce RTX 3080, 10240 MiB", hints=["from setup"])
    big = setup.Status(gpu="NVIDIA GeForce RTX 4090, 24564 MiB")
    assert app.status_hints(small)[0] == "from setup" and "at least 16 GB" in app.status_hints(small)[1]
    assert app.status_hints(big) == [] and app.status_hints(setup.Status()) == []
    assert app.status_hints(setup.Status(gpu="NVIDIA GeForce RTX 5060 Ti, 16311 MiB")) == []
    assert app.status_hints(small, setup=SimpleNamespace()) == ["from setup"]     # setups without the helper


def test_progress_bar_modes(tmp_path):
    root = _tk_root()
    try:
        a = app.App(root, setup=FakeSetup(), runner=FakeRunner(), executor=FakeWSL(), settings_file=tmp_path / "s.json")
        a.show_progress(runner.Progress("find"))                 # started, count unknown
        assert str(a.bar.cget("mode")) == "indeterminate"
        a.show_progress(runner.Progress("repair", 2, 5))
        assert str(a.bar.cget("mode")) == "determinate" and float(a.bar.cget("value")) == 2
        a.show_progress(runner.Progress("select"))
        a.show_progress(runner.Progress("failed", message="exit 1"))
        assert str(a.bar.cget("mode")) == "determinate" and a.stage.cget("text") == "Failed"
    finally:
        root.destroy()


class FakeNative(FakeWSL):
    layout = runner.native_layout("C:\\nvs_repair")


def test_native_window_hides_copy_and_preflights_with_its_layout(tmp_path):
    root = _tk_root()
    seen = []

    class Recording(FakeRunner):
        def commands(self, opts, layout=None):
            seen.append(layout)
            return []

    fr = Recording()
    try:
        a = app.App(root, setup=FakeSetup(), runner=fr, executor=FakeNative(), settings_file=tmp_path / "s.json")
        assert _pump(root, lambda: a.status is not None and not a.busy)
        assert not a.copy_box.winfo_manager()                     # native reads the scene in place
        a.v["scene"].set("C:\\data\\room")
        a.on_run()
        assert _pump(root, lambda: fr.runs and not a.busy)
        assert seen == [FakeNative.layout]
        a.on_stop()
        assert a.executor.stopped.wait(2)
    finally:
        root.destroy()


def test_stop_reports_stopped_not_failed(tmp_path):
    root = _tk_root()
    fw = FakeWSL()

    class UntilStopped(FakeRunner):
        def run(self, opts, wsl, on_line, on_progress):
            self.runs.append(opts)
            on_progress(runner.Progress("find"))
            fw.stopped.wait(5)
            on_progress(runner.Progress("failed", message="exit 1"))
            return 1

    fr = UntilStopped()
    try:
        a = app.App(root, setup=FakeSetup(), runner=fr, executor=fw, settings_file=tmp_path / "s.json")
        assert _pump(root, lambda: a.status is not None and not a.busy)
        a.v["scene"].set(r"C:\data\room")
        a.on_run()
        assert _pump(root, lambda: fr.runs)
        a.on_stop()
        assert _pump(root, lambda: not a.busy)
        log = a.log.get("1.0", "end")
        assert "[run] stopped" in log and "failed (exit" not in log
        assert a.stage.cget("text") == "Stopped"

        a.on_run()                                                  # a later failure without Stop is a failure
        assert _pump(root, lambda: len(fr.runs) == 2 and not a.busy)
        assert "[run] failed (exit 1)" in a.log.get("1.0", "end")
    finally:
        root.destroy()


def test_wsl_window_shows_copy(tmp_path):
    root = _tk_root()
    try:
        a = app.App(root, setup=FakeSetup(), runner=FakeRunner(), executor=FakeWSL(), settings_file=tmp_path / "s.json")
        assert a.copy_box.winfo_manager() == "grid"
    finally:
        root.destroy()


def test_run_preflight_shows_runner_errors(tmp_path, monkeypatch):
    root = _tk_root()
    shown = []
    monkeypatch.setattr("tkinter.messagebox.showerror", lambda title, msg: shown.append(msg))

    class Strict(FakeRunner):
        def commands(self, opts, layout=None):
            raise ValueError("scene must be a Windows path with a drive letter")

    fr = Strict()
    try:
        a = app.App(root, setup=FakeSetup(), runner=fr, executor=FakeWSL(), settings_file=tmp_path / "s.json")
        _pump(root, lambda: not a.busy)
        a.v["scene"].set("/not/a/windows/path")
        a.on_run()
        assert shown == ["scene must be a Windows path with a drive letter"] and fr.runs == []
    finally:
        root.destroy()
