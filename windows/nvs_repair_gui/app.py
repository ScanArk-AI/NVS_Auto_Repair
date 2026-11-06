"""The tkinter window.  Owner: B.

Everything that can be tested without a display -- settings, turning the form
into ``runner.Options``, where ``source/`` is -- lives in plain functions at the
top. ``App`` only lays out widgets and moves work to a background thread: every
WSL call runs off the UI thread and talks back through a queue polled with
``after()``, so the window never freezes during an install or a run.

``setup``, ``runner`` and the executor are passed in, which is how the tests
drive the window with fakes. ``make_backend`` picks the real pair: the offline
native bundle (``runtime\\python.exe`` beside the app) or WSL.
"""

from __future__ import annotations

import json
import ntpath
import os
import posixpath
import queue
import sys
import threading
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from . import __version__
from . import runner as runner_mod
from . import setup as setup_mod

PRESET_LABELS = {
    "default": "Indoor, robot height (0.4 m, level / up)",
    "object-centric": "Object-centric, outdoor",
    "object-centric-indoor": "Object-centric, indoor",
}
MODE_LABELS = {
    "check": "Check only (finds paths, passes renders through; no weights needed)",
    "repair": "Repair (1.3B)",
}
UNITS_MODES = ("auto", "metric", "number")


# --------------------------------------------------------------------------- #
# settings
# --------------------------------------------------------------------------- #
@dataclass
class Settings:
    scene: str = ""
    out: str = ""
    preset: str = "default"
    units_mode: str = "auto"         # auto | metric | number
    units_value: str = ""            # scene units per metre when units_mode == "number"
    caption: str = ""
    mode: str = "check"
    limit: str = ""                  # "" = all paths
    copy_to_wsl: bool = False
    gh_proxy: str = ""
    hf_endpoint: str = ""
    hf_disable_xet: bool = False


def settings_path() -> Path:
    """``%APPDATA%\\nvs_repair\\settings.json`` on Windows, ``~/.config/nvs_repair/`` elsewhere."""
    if sys.platform == "win32" and os.environ.get("APPDATA"):
        return Path(os.environ["APPDATA"]) / "nvs_repair" / "settings.json"
    return Path.home() / ".config" / "nvs_repair" / "settings.json"


def load_settings(path: Path | None = None) -> Settings:
    """Saved settings; defaults for anything missing, unknown or unreadable."""
    path = path or settings_path()
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return Settings()
    if not isinstance(raw, dict):
        return Settings()
    known = {f.name: f for f in fields(Settings)}
    s = Settings()
    for k, v in raw.items():
        if k in known and isinstance(v, type(getattr(s, k))):
            setattr(s, k, v)
    if s.preset not in runner_mod.PRESETS:
        s.preset = "default"
    if s.mode not in runner_mod.MODES:
        s.mode = "check"
    if s.units_mode not in UNITS_MODES:
        s.units_mode = "auto"
    return s


def save_settings(s: Settings, path: Path | None = None) -> Path:
    path = Path(path or settings_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(s), indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)
    return path


# --------------------------------------------------------------------------- #
# form -> runner.Options
# --------------------------------------------------------------------------- #
def _pathmod(p: str):
    """ntpath for Windows-looking paths (drive letter or backslash), posixpath otherwise."""
    return ntpath if ("\\" in p or (len(p) > 1 and p[1] == ":")) else posixpath


def default_out(scene: str) -> str:
    """``<parent>/<name>_nvs_repair`` next to the scene directory."""
    scene = scene.strip()
    if not scene:
        return ""
    m = _pathmod(scene)
    head, name = m.split(scene.rstrip("\\/"))
    return m.join(head, f"{name}_nvs_repair") if name else ""


def units_arg(s: Settings) -> str:
    if s.units_mode == "auto":
        return "auto"
    if s.units_mode == "metric":
        return "1"
    try:
        v = float(s.units_value)
    except ValueError:
        raise ValueError("Units per metre must be a number, e.g. 1.6") from None
    if not v > 0:
        raise ValueError("Units per metre must be greater than 0")
    return f"{v:g}"


def to_options(s: Settings) -> runner_mod.Options:
    """The run the form describes, or ValueError with a message for the user."""
    scene = s.scene.strip()
    if not scene:
        raise ValueError("Choose a scene directory first")
    out = s.out.strip() or default_out(scene)
    limit = None
    if s.limit.strip():
        try:
            limit = int(s.limit)
        except ValueError:
            raise ValueError("'Only the first N paths' must be a whole number") from None
        if limit < 1:
            raise ValueError("'Only the first N paths' must be at least 1")
    if s.preset not in runner_mod.PRESETS:
        raise ValueError(f"Unknown capture type {s.preset!r}")
    if s.mode not in runner_mod.MODES:
        raise ValueError(f"Unknown mode {s.mode!r}")
    return runner_mod.Options(scene=scene, out=out, preset=s.preset, units=units_arg(s),
                              caption=s.caption.strip(), mode=s.mode, limit=limit,
                              copy_to_wsl=bool(s.copy_to_wsl))


def mirrors(s: Settings) -> dict[str, str]:
    """The download mirrors to pass to ``setup.install``; only what the user filled in."""
    m = {}
    if s.gh_proxy.strip():
        m["GH_PROXY"] = s.gh_proxy.strip()
    if s.hf_endpoint.strip():
        m["HF_ENDPOINT"] = s.hf_endpoint.strip()
    if s.hf_disable_xet:
        m["HF_HUB_DISABLE_XET"] = "1"
    return m


def bundle_root(executable: str | None = None) -> Path | None:
    """The native bundle's top directory (the one holding ``runtime\\python.exe``), or None.

    The bundle's launcher starts ``runtime\\pythonw.exe``, so the interpreter's
    own directory and its parent are both checked.
    """
    exe = Path(executable or sys.executable).resolve()
    for d in (exe.parent, exe.parent.parent):
        if (d / "runtime" / "python.exe").is_file():
            return d
    return None


def source_dir() -> Path:
    """What ``setup.install`` installs from.

    native bundle: ``<root>\\app``; WSL exe: the ``source/`` folder next to it;
    from a checkout: the repo root.
    """
    root = bundle_root()
    if root is not None:
        return root / "app"
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "source"
    return Path(__file__).resolve().parents[2]


def make_backend(root: Path | None = None):
    """``(executor, setup module)``: native inside the offline bundle, WSL otherwise."""
    root = root if root is not None else bundle_root()
    if root is not None:
        from . import native, native_setup

        return native.Native(runner_mod.native_layout(str(root))), native_setup
    from .wsl import WSL

    return WSL(), setup_mod


def status_text(st) -> str:
    """One line for the Environment row (native bundle or WSL, by the status it got)."""
    mark = lambda ok: "OK" if ok else "missing"  # noqa: E731
    gpu = f"GPU: {st.gpu}" if st.gpu else "GPU: none found"
    if hasattr(st, "driver"):           # native_setup.NativeStatus
        driver = (f"driver {st.driver} {'OK' if st.driver_ok else 'too old'}" if st.driver
                  else "driver missing")
        parts = [driver, f"VC++ runtime {mark(st.vcredist)}", gpu,
                 f"runtime (torch+CUDA) {mark(st.runtime)}", f"GAVIS {mark(st.gavis)}"]
    else:                               # setup.Status (WSL)
        parts = [f"WSL2 {mark(st.wsl and st.wsl2)}", f"Ubuntu-24.04 {mark(st.distro and st.user)}", gpu,
                 f"installed {mark(st.installed)}"]
    return "   ".join(parts + [f"1.3B weights {mark(st.weights)}"])


# Repair (1.3B) peaks at ~13 GB with model.low_vram, which the run turns on for
# GPUs under 24 GB. A "16 GB" card reports ~15.9 GiB, hence 15 against "16 GB".
MIN_REPAIR_GIB = 15


def status_hints(st, setup=setup_mod) -> list[str]:
    """setup's own hints, plus a warning when the GPU is too small for Repair."""
    hints = list(st.hints)
    gib_of = getattr(setup, "gpu_memory_gib", None)
    gib = gib_of(st.gpu) if (gib_of and st.gpu) else None
    if gib is not None and gib < MIN_REPAIR_GIB:
        hints.append(f"Repair needs a GPU with at least 16 GB; this one has {gib:.0f} GB "
                     f"(Check the install still works).")
    return hints


STAGE_LABELS = {"prepare": "Preparing", "find": "Finding paths", "select": "Choosing reference photos",
                "repair": "Repairing", "done": "Done", "failed": "Failed"}


def stage_text(p) -> str:
    text = STAGE_LABELS.get(p.stage, p.stage)
    if p.stage == "prepare" and p.message:
        return f"{text}: {p.message}"
    return f"{text} {p.done}/{p.total}" if p.total else text


# --------------------------------------------------------------------------- #
# the window
# --------------------------------------------------------------------------- #
class App:
    """The single window. ``setup``/``runner``/``executor`` are injectable for tests;
    by default ``make_backend`` chooses native or WSL."""

    def __init__(self, root, setup=None, runner=runner_mod, executor=None, settings_file: Path | None = None):
        import tkinter as tk
        from tkinter import ttk

        if executor is None:
            executor, default_setup = make_backend()
            setup = setup or default_setup
        self.tk, self.ttk = tk, ttk
        self.root, self.setup, self.runner = root, setup or setup_mod, runner
        self.executor = executor
        self.layout = getattr(executor, "layout", None)
        self.settings_file = settings_file
        self.events: queue.Queue = queue.Queue()
        self.busy = False
        self.stopping = False
        self.status = None
        s = load_settings(settings_file)

        root.title(f"nvs_repair {__version__}")
        root.minsize(760, 560)
        self.v = {name: (tk.BooleanVar(value=getattr(s, name)) if isinstance(getattr(s, name), bool)
                         else tk.StringVar(value=getattr(s, name)))
                  for name in (f.name for f in fields(Settings))}
        self._out_auto = not s.out or s.out == default_out(s.scene)
        self.v["scene"].trace_add("write", self._scene_changed)

        pad = {"padx": 6, "pady": 3}
        frm = ttk.Frame(root, padding=8)
        frm.pack(fill="both", expand=True)
        frm.columnconfigure(1, weight=1)

        # --- environment ---------------------------------------------------
        env = ttk.LabelFrame(frm, text="Environment", padding=6)
        env.grid(row=0, column=0, columnspan=3, sticky="ew", **pad)
        env.columnconfigure(0, weight=1)
        self.status_label = ttk.Label(env, text="Checking the environment ...", justify="left")
        self.status_label.grid(row=0, column=0, sticky="w")
        self.hints_label = ttk.Label(env, text="", foreground="#b05000", justify="left")
        self.hints_label.grid(row=1, column=0, sticky="w")
        # Long GPU names and hints wrap with the window instead of being cut off.
        env.bind("<Configure>", lambda e: [w.configure(wraplength=max(200, e.width - 24))
                                           for w in (self.status_label, self.hints_label)])
        btns = ttk.Frame(env)
        btns.grid(row=3, column=0, sticky="e", pady=(4, 0))
        self.install_btn = ttk.Button(btns, text="Install / update", command=self.on_install)
        self.install_btn.pack(side="left", padx=2)
        ttk.Button(btns, text="Re-check", command=self.refresh_status).pack(side="left", padx=2)
        self.adv_btn = ttk.Button(btns, text="Advanced", command=self.toggle_advanced)
        self.adv_btn.pack(side="left", padx=2)

        self.adv = ttk.Frame(env)
        self.adv.columnconfigure(1, weight=1)
        for r, (key, label) in enumerate((("gh_proxy", "GH_PROXY (GitHub mirror prefix)"),
                                          ("hf_endpoint", "HF_ENDPOINT (Hugging Face mirror)"))):
            ttk.Label(self.adv, text=label).grid(row=r, column=0, sticky="w", **pad)
            ttk.Entry(self.adv, textvariable=self.v[key]).grid(row=r, column=1, sticky="ew", **pad)
        ttk.Checkbutton(self.adv, text="HF_HUB_DISABLE_XET=1 (if downloads fail with 401)",
                        variable=self.v["hf_disable_xet"]).grid(row=2, column=0, columnspan=2, sticky="w", **pad)
        self.copy_box = ttk.Checkbutton(self.adv, text="copy the scene into WSL first (faster than reading C: "
                                        "through /mnt/c)", variable=self.v["copy_to_wsl"])
        if self.layout is None or self.layout.scenes is not None:     # the native bundle reads in place
            self.copy_box.grid(row=3, column=0, columnspan=2, sticky="w", **pad)

        # --- scene and options ---------------------------------------------
        row = 1
        for key, label, pick in (("scene", "Scene directory", self.pick_scene),
                                 ("out", "Output directory", self.pick_out)):
            ttk.Label(frm, text=label).grid(row=row, column=0, sticky="w", **pad)
            ttk.Entry(frm, textvariable=self.v[key]).grid(row=row, column=1, sticky="ew", **pad)
            ttk.Button(frm, text="Browse ...", command=pick).grid(row=row, column=2, **pad)
            row += 1
        self.v["out"].trace_add("write", self._out_changed)

        ttk.Label(frm, text="Capture type").grid(row=row, column=0, sticky="nw", **pad)
        box = ttk.Frame(frm)
        box.grid(row=row, column=1, columnspan=2, sticky="w")
        for key in runner.PRESETS:
            ttk.Radiobutton(box, text=PRESET_LABELS.get(key, key), value=key,
                            variable=self.v["preset"]).pack(anchor="w")
        row += 1

        ttk.Label(frm, text="Scale").grid(row=row, column=0, sticky="w", **pad)
        box = ttk.Frame(frm)
        box.grid(row=row, column=1, columnspan=2, sticky="w")
        for key, text in (("auto", "Estimate"), ("metric", "Metric"), ("number", "Units per metre:")):
            ttk.Radiobutton(box, text=text, value=key, variable=self.v["units_mode"]).pack(side="left")
        ttk.Entry(box, width=8, textvariable=self.v["units_value"]).pack(side="left", padx=4)
        row += 1

        ttk.Label(frm, text="Scene description").grid(row=row, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.v["caption"]).grid(row=row, column=1, columnspan=2, sticky="ew", **pad)
        row += 1
        ttk.Label(frm, text="(optional, English; leave empty to repair without one)",
                  foreground="#666").grid(row=row, column=1, sticky="w", padx=6)
        row += 1

        ttk.Label(frm, text="Mode").grid(row=row, column=0, sticky="nw", **pad)
        box = ttk.Frame(frm)
        box.grid(row=row, column=1, columnspan=2, sticky="w")
        for key in runner.MODES:
            ttk.Radiobutton(box, text=MODE_LABELS.get(key, key), value=key,
                            variable=self.v["mode"]).pack(anchor="w")
        row += 1

        ttk.Label(frm, text="Only the first N paths").grid(row=row, column=0, sticky="w", **pad)
        box = ttk.Frame(frm)
        box.grid(row=row, column=1, columnspan=2, sticky="ew")
        ttk.Entry(box, width=6, textvariable=self.v["limit"]).pack(side="left")
        ttk.Label(box, text="(empty = all)", foreground="#666").pack(side="left", padx=6)
        self.stop_btn = ttk.Button(box, text="Stop", command=self.on_stop, state="disabled")
        self.stop_btn.pack(side="right", padx=2)
        self.run_btn = ttk.Button(box, text="Run", command=self.on_run)
        self.run_btn.pack(side="right", padx=2)
        row += 1

        # --- progress and log ----------------------------------------------
        self.stage = ttk.Label(frm, text="Idle")
        self.stage.grid(row=row, column=0, sticky="w", **pad)
        self.bar = ttk.Progressbar(frm, mode="determinate", maximum=1)
        self.bar.grid(row=row, column=1, columnspan=2, sticky="ew", **pad)
        row += 1
        logf = ttk.Frame(frm)
        logf.grid(row=row, column=0, columnspan=3, sticky="nsew", **pad)
        frm.rowconfigure(row, weight=1)
        self.log = tk.Text(logf, height=14, wrap="none", font=("Consolas", 9))
        sb = ttk.Scrollbar(logf, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        self.log.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        row += 1
        box = ttk.Frame(frm)
        box.grid(row=row, column=0, columnspan=3, sticky="w")
        ttk.Button(box, text="Open results", command=self.open_out).pack(side="left", padx=2)
        ttk.Button(box, text="Save log ...", command=self.save_log).pack(side="left", padx=2)

        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.after(100, self._poll)
        self.refresh_status()

    # --- form ---------------------------------------------------------------
    def settings(self) -> Settings:
        return Settings(**{k: var.get() for k, var in self.v.items()})

    def _scene_changed(self, *_):
        if self._out_auto:
            self._setting_out = True
            self.v["out"].set(default_out(self.v["scene"].get()))
            self._setting_out = False

    def _out_changed(self, *_):
        if not getattr(self, "_setting_out", False):
            self._out_auto = not self.v["out"].get().strip()

    def pick_scene(self):
        from tkinter import filedialog

        d = filedialog.askdirectory(title="Scene directory (3DGS ply + COLMAP + images)")
        if d:
            self.v["scene"].set(os.path.normpath(d))

    def pick_out(self):
        from tkinter import filedialog

        d = filedialog.askdirectory(title="Output directory")
        if d:
            self._out_auto = False
            self.v["out"].set(os.path.normpath(d))

    def toggle_advanced(self):
        if self.adv.winfo_manager():
            self.adv.grid_forget()
        else:
            self.adv.grid(row=2, column=0, sticky="ew", pady=(6, 0))

    # --- background work ----------------------------------------------------
    def _start(self, name: str, fn) -> bool:
        if self.busy:
            return False
        self.busy = True
        self.stopping = False
        self._set_buttons()

        def work():
            try:
                rc = fn()
            except Exception as exc:  # noqa: BLE001 - surface anything to the log
                self.events.put(("line", f"[{name}] error: {exc}"))
                rc = 1
            self.events.put(("finished", (name, rc)))

        threading.Thread(target=work, name=f"nvs_repair-{name}", daemon=True).start()
        return True

    def _poll(self):
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "line":
                    self.append(payload)
                elif kind == "progress":
                    self.show_progress(payload)
                elif kind == "status":
                    self.show_status(payload)
                elif kind == "finished":
                    name, rc = payload
                    self.busy = False
                    self._set_buttons()
                    if name == "run":
                        self._bar_mode("determinate")
                    if name != "status" and rc != 0 and self.stopping:
                        # The killed step's exit code is not a failure the user should chase.
                        self.append(f"[{name}] stopped")
                        self.stage.configure(text="Stopped")
                    elif name != "status":
                        self.append(f"[{name}] {'finished' if rc == 0 else f'failed (exit {rc})'}")
                        if name == "install":
                            self.refresh_status()
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _set_buttons(self):
        state = "disabled" if self.busy else "normal"
        for b in (self.run_btn, self.install_btn):
            b.configure(state=state)
        self.stop_btn.configure(state="normal" if self.busy else "disabled")

    def refresh_status(self):
        def fn():
            self.events.put(("status", self.setup.status(self.executor)))
            return 0

        if not self._start("status", fn):
            return
        self.status_label.configure(text="Checking the environment ...")

    def show_status(self, st):
        self.status = st
        self.status_label.configure(text=status_text(st))
        self.hints_label.configure(text="\n".join(status_hints(st, self.setup)))

    def on_install(self):
        s = self.settings()
        save_settings(s, self.settings_file)
        with_weights = s.mode == "repair"
        self.append(f"[install] {'with' if with_weights else 'without'} the 1.3B weights"
                    f"{' (set Mode to Repair (1.3B) to download them)' if not with_weights else ''}")
        self._start("install", lambda: self.setup.install(
            self.executor, str(source_dir()), lambda line: self.events.put(("line", line)),
            with_weights=with_weights, mirrors=mirrors(s)))

    def on_run(self):
        from tkinter import messagebox

        s = self.settings()
        try:
            opts = to_options(s)
            # runner validates what the form cannot (e.g. paths WSL cannot reach);
            # better to say so now than after the backend has started.
            if hasattr(self.runner, "commands"):
                self.runner.commands(opts, self.layout)
        except ValueError as exc:
            messagebox.showerror("nvs_repair", str(exc))
            return
        if self.status is not None and not self.status.ready:
            if not messagebox.askyesno("nvs_repair", "The environment is not ready (see the Environment "
                                                     "row). Run anyway?"):
                return
        if opts.mode == "repair" and self.status is not None and not self.status.weights:
            messagebox.showinfo("nvs_repair", "The 1.3B weights are not installed yet. With Mode set to "
                                              "Repair (1.3B), click Install / update first.")
            return
        save_settings(s, self.settings_file)
        if self.v["out"].get().strip() != opts.out:
            self.v["out"].set(opts.out)
        self.bar.configure(value=0, maximum=1)
        self.stage.configure(text="Starting")
        self.append(f"[run] {opts}")
        self._start("run", lambda: self.runner.run(
            opts, self.executor, lambda line: self.events.put(("line", line)),
            lambda p: self.events.put(("progress", p))))

    def on_stop(self):
        self.stopping = True
        self.append("[stop] stopping ...")
        threading.Thread(target=self.executor.stop, daemon=True).start()

    def show_progress(self, p):
        """Counts fill the bar; a stage that has started without a count yet
        (done = total = 0) runs it in indeterminate mode; done / failed stop it."""
        self.stage.configure(text=stage_text(p))
        if p.total:
            self._bar_mode("determinate")
            self.bar.configure(maximum=p.total, value=p.done)
        elif p.stage == "done":
            self._bar_mode("determinate")
            self.bar.configure(maximum=1, value=1)
        elif p.stage == "failed":
            self._bar_mode("determinate")
        else:
            self._bar_mode("indeterminate")

    def _bar_mode(self, mode: str):
        if str(self.bar.cget("mode")) == mode:
            return
        if mode == "indeterminate":
            self.bar.configure(mode="indeterminate", value=0)
            self.bar.start(15)
        else:
            self.bar.stop()
            self.bar.configure(mode="determinate")

    # --- log and output -------------------------------------------------------
    def append(self, line: str):
        self.log.insert("end", line.rstrip("\n") + "\n")
        if int(self.log.index("end-1c").split(".")[0]) > 20000:   # keep the widget responsive
            self.log.delete("1.0", "2000.0")
        self.log.see("end")

    def save_log(self):
        from tkinter import filedialog

        path = filedialog.asksaveasfilename(defaultextension=".txt", initialfile="nvs_repair_log.txt")
        if path:
            Path(path).write_text(self.log.get("1.0", "end"), encoding="utf-8")

    def open_out(self):
        from tkinter import messagebox

        out = self.v["out"].get().strip() or default_out(self.v["scene"].get())
        if not out or not os.path.isdir(out):
            messagebox.showinfo("nvs_repair", f"No output folder yet:\n{out or '(none)'}")
        elif sys.platform == "win32":
            os.startfile(out)  # noqa: S606 - opening a folder the user chose
        else:
            messagebox.showinfo("nvs_repair", out)

    def on_close(self):
        save_settings(self.settings(), self.settings_file)
        if self.busy:
            self.executor.stop()
        self.root.destroy()


def main() -> int:
    import tkinter as tk

    root = tk.Tk()
    App(root)
    root.mainloop()
    return 0
