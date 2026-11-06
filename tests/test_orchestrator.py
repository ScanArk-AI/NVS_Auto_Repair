"""Orchestrator: config layering, scene detection, make-config, stage wiring."""

from pathlib import Path

import os
import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_example_scene_config_loads():
    from nvs_repair.config import load

    cfg = load(str(ROOT / "configs/scenes/example.yaml"))
    assert cfg.out.endswith("outputs/example")
    assert Path(cfg.find.config).name == "example.find.yaml" and Path(cfg.find.config).is_absolute()
    assert cfg.workflow.reference_mode == "photos"


def _scene(tmp_path, *files):
    root = tmp_path / "scene"
    for rel in files:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).touch()
    return root


def test_detect_scene_picks_latest_iteration(tmp_path):
    from nvs_repair.config import detect_scene

    root = _scene(tmp_path, "sparse/0/cameras.bin", "images/x.jpg",
                  "point_cloud/iteration_7000/point_cloud.ply", "point_cloud/iteration_30000/point_cloud.ply")
    info = detect_scene(root)
    assert info["ply"] == Path("point_cloud/iteration_30000/point_cloud.ply")
    assert info["sparse"] == Path("sparse/0") and info["images"] == Path("images")


def test_detect_scene_root_ply_and_flat_sparse(tmp_path):
    from nvs_repair.config import detect_scene

    info = detect_scene(_scene(tmp_path, "sparse/cameras.txt", "point_cloud.ply"))
    assert info["ply"] == Path("point_cloud.ply") and info["sparse"] == Path("sparse")
    assert info["images"] is None


def test_make_config_writes_three_configs(tmp_path):
    import yaml
    from nvs_repair.make_config import generate

    root = _scene(tmp_path, "sparse/0/cameras.bin", "images/x.jpg", "splat.ply")
    wf = generate(root, tmp_path / "gen")
    w = yaml.safe_load(wf.read_text())
    rep = yaml.safe_load(Path(w["repair"]["config"]).read_text())
    assert rep["scene"]["ply"] == "splat.ply" and rep["scene"]["mask_invalid"] == "none"
    assert Path(w["find"]["config"]).is_file()


def test_make_config_explains_unknown_layout(tmp_path):
    from nvs_repair.make_config import generate

    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit, match="example.yaml"):
        generate(tmp_path / "empty", tmp_path / "gen")


def test_reference_mode_is_validated():
    from nvs_repair import stages
    from nvs_repair.config import load

    cfg = load(str(ROOT / "configs/scenes/example.yaml"), ["workflow.reference_mode=bogus"])
    with pytest.raises(ValueError, match="reference_mode"):
        stages.reference_mode(cfg)


def test_progressive_stage_wiring(monkeypatch, tmp_path):
    from nvs_repair import envs, stages
    from nvs_repair.config import load

    monkeypatch.setattr(envs, "resolve", lambda cfg, stage: "/usr/bin/python3")
    cfg = load(str(ROOT / "configs/scenes/example.yaml"),
               [f"out={tmp_path}", "workflow.reference_mode=progressive", "repair.set=[progressive.max_generated=2]"])
    st = stages.progressive_stage(cfg)
    sets = [st.argv[i + 1] for i, a in enumerate(st.argv) if a == "--set"]
    assert st.argv[:2] == ["-m", "repair_pipeline.progressive"]
    assert f"output.dir={tmp_path}/2_repair" in sets
    assert "progressive.enabled=true" in sets
    assert "progressive.max_generated=2" in sets                          # the user's repair.set
    assert not any(s.startswith("prompt.caption_h5=") for s in sets)      # no caption.h5 yet

    (tmp_path / "caption.h5").touch()
    arm = stages.progressive_stage(cfg, out_dir=tmp_path / "2_repair_x", extra=["progressive.enabled=false"])
    sets = [arm.argv[i + 1] for i, a in enumerate(arm.argv) if a == "--set"]
    assert f"prompt.caption_h5={tmp_path}/caption.h5" in sets
    assert sets[-1] == f"output.dir={tmp_path}/2_repair_x"
    assert sets.index("progressive.enabled=false") > sets.index("progressive.enabled=true")


def test_repair_backend_precedence(tmp_path):
    from nvs_repair import stages
    from nvs_repair.config import load

    cfg = load(str(ROOT / "configs/scenes/example.yaml"))
    assert stages._repair_backend(cfg) == "artifixer"                       # method.yaml default

    scene = tmp_path / "repair.yaml"
    scene.write_text("model:\n  backend: mypkg.mod:Model\n")
    cfg = load(str(ROOT / "configs/scenes/example.yaml"), [f"repair.config={scene}"])
    assert stages._repair_backend(cfg) == "mypkg.mod:Model"                 # scene config

    cfg = load(str(ROOT / "configs/scenes/example.yaml"),
               [f"repair.config={scene}", "repair.set=[model.backend=artifixer, model.backend=passthrough]"])
    assert stages._repair_backend(cfg) == "passthrough"                     # last --set wins


def test_wait_for_vram_only_gates_artifixer(monkeypatch):
    from nvs_repair import stages
    from nvs_repair.config import load

    def no_gpu_query(device):
        raise AssertionError("passthrough must not wait for VRAM")

    monkeypatch.setattr(stages, "_free_vram_gb", no_gpu_query)
    cfg = load(str(ROOT / "configs/scenes/example.yaml"), ["repair.set=[model.backend=passthrough]"])
    stages.wait_for_vram(cfg, "repair")                                     # returns without querying

    calls = []
    monkeypatch.setattr(stages, "_free_vram_gb", lambda device: calls.append(device) or 1000.0)
    cfg = load(str(ROOT / "configs/scenes/example.yaml"))
    stages.wait_for_vram(cfg, "repair")
    assert calls                                                            # artifixer is gated


@pytest.mark.parametrize("free, preset, expected", [
    (15.3, [], ["model.low_vram=true"]),                    # 16 GB card: switched on
    (23.0, [], ["model.low_vram=true"]),
    (40.0, [], []),                                         # room for the default config
    (15.3, ["model.low_vram=false"], ["model.low_vram=false"]),   # the user's choice stands
])
def test_wait_for_vram_turns_on_low_vram_below_24_gib(monkeypatch, free, preset, expected):
    from nvs_repair import envs, stages
    from nvs_repair.config import load

    monkeypatch.setattr(envs, "resolve", lambda cfg, stage: "/usr/bin/python3")
    monkeypatch.setattr(stages, "_free_vram_gb", lambda device: free)
    cfg = load(str(ROOT / "configs/scenes/example.yaml"), [f"repair.set=[{','.join(preset)}]"])
    stages.wait_for_vram(cfg, "repair")
    assert list(cfg.repair.set) == expected
    assert ("model.low_vram=true" in stages.pipeline_stage(cfg, "repair").argv) == (expected == ["model.low_vram=true"])


def test_wait_for_vram_waits_below_12_5_gib_for_the_13b(monkeypatch):
    from nvs_repair import stages
    from nvs_repair.config import load

    frees = iter([12.3, 12.7])
    monkeypatch.setattr(stages, "_free_vram_gb", lambda device: next(frees))
    monkeypatch.setattr(stages.time, "sleep", lambda s: None)
    cfg = load(str(ROOT / "configs/scenes/example.yaml"))
    stages.wait_for_vram(cfg, "repair")
    assert "model.low_vram=true" in list(cfg.repair.set)


@pytest.mark.parametrize("variant, expected", [("1.3b", "Wan2.1-T2V-1.3B-Diffusers"), ("14b", "Wan2.1-T2V-14B-Diffusers")])
def test_text_encoder_follows_variant(variant, expected):
    from nvs_repair.config import load

    cfg = load(str(ROOT / "configs/scenes/example.yaml"), [f"model.variant={variant}"])
    assert cfg.checkpoints.text_encoder.endswith(expected)
    cfg = load(str(ROOT / "configs/scenes/example.yaml"),
               [f"model.variant={variant}", "checkpoints.text_encoder=/elsewhere/umt5"])
    assert cfg.checkpoints.text_encoder == "/elsewhere/umt5"


def test_passthrough_run_skips_caption(monkeypatch, tmp_path):
    """A wiring check with model.backend=passthrough must not need a VLM or the text encoder."""
    import nvs_repair.caption_vlm as caption_vlm
    from nvs_repair import layout, stages
    from nvs_repair.config import load

    cfg = load(str(ROOT / "configs/scenes/example.yaml"),
               [f"out={tmp_path}", "repair.set=[model.backend=passthrough]"])
    traj = tmp_path / "1_find/repair/trajectories/traj_c0_00"
    traj.mkdir(parents=True)
    (traj / "poses.json").write_text("{}")
    (traj / "references").mkdir()

    def boom(*a, **k):
        raise AssertionError("captioning must be skipped for passthrough")

    ran = []
    monkeypatch.setattr(caption_vlm, "ensure_caption", boom)
    monkeypatch.setattr(stages, "find_stage", boom)
    monkeypatch.setattr(stages, "pipeline_stage",
                        lambda cfg, sub: type("S", (), {"run": lambda self: ran.append(sub) or 0})())
    assert stages.run_all(cfg) == 0
    assert ran == ["select", "repair"]


def test_make_config_units_per_metre(tmp_path):
    import yaml
    from nvs_repair.make_config import generate

    root = _scene(tmp_path, "sparse/0/cameras.bin", "splat.ply")
    find = lambda wf: yaml.safe_load(Path(yaml.safe_load(wf.read_text())["find"]["config"]).read_text())
    assert find(generate(root, tmp_path / "a"))["geometry"]["units_per_metre"] == "auto"
    assert find(generate(root, tmp_path / "b", units_per_metre="1"))["geometry"]["units_per_metre"] == 1.0
    with pytest.raises(SystemExit, match="positive"):
        generate(root, tmp_path / "c", units_per_metre="-2")


@pytest.mark.parametrize("paths", ["object-centric", "object-centric-indoor"])
def test_make_config_path_presets(tmp_path, paths):
    import yaml
    from nvs_repair.make_config import PATH_PRESETS, generate

    root = _scene(tmp_path, "sparse/0/cameras.bin", "splat.ply")
    find = lambda wf: yaml.safe_load(Path(yaml.safe_load(wf.read_text())["find"]["config"]).read_text())
    assert "trajectories" not in find(generate(root, tmp_path / "default"))
    preset = yaml.safe_load(PATH_PRESETS[paths].read_text())["trajectories"]
    got = find(generate(root, tmp_path / paths, paths=paths))["trajectories"]
    assert got == preset and min(got["pitches"]) < 0                          # copied verbatim; looks down
    with pytest.raises(SystemExit, match="unknown path preset"):
        generate(root, tmp_path / "x", paths="drone")


def test_relative_artifixer_repo_resolves_against_the_repo():
    from nvs_repair.config import load

    cfg = load(str(ROOT / "configs/scenes/example.yaml"), ["paths.artifixer_repo=../ArtiFixer"])
    assert cfg.paths.artifixer_repo == os.path.normpath(ROOT / "../ArtiFixer")
    cfg = load(str(ROOT / "configs/scenes/example.yaml"), ["paths.artifixer_repo=/abs/ArtiFixer"])
    assert cfg.paths.artifixer_repo == "/abs/ArtiFixer"


def test_caption_encoding_is_reused_for_the_same_description(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("h5py")
    import nvs_repair.caption_vlm as caption_vlm
    from nvs_repair.config import load

    caption_vlm._repair_path()
    from repair_pipeline import caption

    encoded = []

    def fake_encode(text, model_id):
        encoded.append(text)
        return torch.full((3, caption.TEXT_EMBED_DIM), float(len(encoded)), dtype=torch.bfloat16)

    monkeypatch.setattr(caption, "encode_caption", fake_encode)
    monkeypatch.setattr(caption_vlm, "CAPTION_CACHE", tmp_path / "cache")

    def run(out, text):
        cfg = load(str(ROOT / "configs/scenes/example.yaml"), [f"out={out}", f"workflow.caption_text={text}"])
        return caption_vlm.ensure_caption(cfg, tmp_path / out)

    first = run("a", "A white table.")
    second = run("b", "A white table.")                  # new output directory, same description
    assert encoded == ["A white table."]
    assert second.read_bytes() == first.read_bytes()
    assert (tmp_path / "b" / "caption.txt").read_text(encoding="utf-8") == "A white table.\n"
    run("c", "A grey chair.")
    assert encoded == ["A white table.", "A grey chair."]
    assert len(list((tmp_path / "cache").glob("*.h5"))) == 2
    assert not list((tmp_path / "cache").glob("*.partial"))


def test_doctor_checks_only_the_selected_variants_weights(tmp_path):
    """`bootstrap.sh --variant 1.3b` installs no 14B weights; doctor must not ask for them."""
    from nvs_repair import envs
    from nvs_repair.config import load

    ck = tmp_path / "checkpoints"
    cfg = load(str(ROOT / "configs/scenes/example.yaml"), [f"checkpoints_dir={ck}"])
    spec = cfg.checkpoints.artifixer["1.3b"]
    (ck / spec.checkpoint_pt).parent.mkdir(parents=True, exist_ok=True)
    (ck / spec.checkpoint_pt).touch()
    (ck / spec.model_id).mkdir(parents=True)
    assert envs._check_weights(cfg) == []
    cfg14 = load(str(ROOT / "configs/scenes/example.yaml"), [f"checkpoints_dir={ck}", "model.variant=14b"])
    assert [p.split(":")[0] for p in envs._check_weights(cfg14)] == ["14b.checkpoint_pt", "14b.model_id", "text_encoder"]
