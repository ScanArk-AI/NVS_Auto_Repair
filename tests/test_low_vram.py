"""model.low_vram: the KV cache is sized to the clip; references are VAE-encoded in batches."""

import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from repair_pipeline import model


def test_trim_kv_cache_caps_to_the_clip():
    calls = []
    pipe = SimpleNamespace(_initialize_kv_cache=lambda b, fsl, n: calls.append(n))
    model._trim_kv_cache(pipe)
    pipe._initialize_kv_cache(1, 2040, 21)          # no clip length yet: unchanged
    pipe._nvs_cache_frames = 7                       # a 25-frame clip
    pipe._initialize_kv_cache(1, 2040, 21)
    pipe._nvs_cache_frames = 30                      # longer than the window: the window wins
    pipe._initialize_kv_cache(1, 2040, 21)
    assert calls == [21, 7, 21]


def test_low_vram_defaults_off():
    from omegaconf import OmegaConf

    from repair_pipeline.config import METHOD_CONFIG

    m = OmegaConf.load(METHOD_CONFIG).model
    assert m.low_vram is False and m.trim_kv_cache is False and m.max_neighbors_per_encode is None
    assert model.LOW_VRAM_NEIGHBORS_PER_ENCODE == 2


@pytest.mark.parametrize("use_cuda", [True, False], ids=["cuda", "cpu"])
def test_caption_encoder_goes_to_the_gpu_when_there_is_one(monkeypatch, use_cuda):
    """Exercise the device choice without loading a model or using a GPU."""
    from repair_pipeline import caption

    bf16 = object()
    empty_cache = Mock()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        bfloat16=bf16, no_grad=nullcontext,
        cuda=SimpleNamespace(is_available=lambda: use_cuda, empty_cache=empty_cache),
    ))
    token_ids = Mock()
    mask = Mock()
    mask.gt.return_value.sum.return_value = [SimpleNamespace(item=lambda: 3)]
    tokenizer = Mock(return_value=SimpleNamespace(input_ids=token_ids, attention_mask=mask))
    tokenizer_loader = Mock(return_value=tokenizer)
    result = object()
    trimmed = Mock()
    trimmed.to.return_value.cpu.return_value = result
    rows = Mock()
    rows.__getitem__ = Mock(return_value=trimmed)
    embeddings = Mock()
    embeddings.__getitem__ = Mock(return_value=rows)
    encoder = Mock(return_value=SimpleNamespace(last_hidden_state=embeddings))
    encoder.device = "cuda:0" if use_cuda else "cpu"
    encoder_loader = Mock(return_value=encoder)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_loader)))
    monkeypatch.setattr(caption, "load_text_encoder", encoder_loader)

    model_id = "local/wan"
    assert caption.encode_caption("An indoor room.", model_id) is result
    tokenizer_loader.assert_called_once_with(model_id, subfolder="tokenizer")
    encoder_loader.assert_called_once_with(caption.Path(model_id) / "text_encoder", encoder.device, bf16)
    assert empty_cache.call_count == (1 if use_cuda else 0)
    token_ids.to.assert_called_once_with(encoder.device)
    mask.to.assert_called_once_with(encoder.device)
    tokenizer.assert_called_once_with(
        ["An indoor room."], padding="max_length", max_length=512,
        truncation=True, add_special_tokens=True, return_tensors="pt",
    )
    rows.__getitem__.assert_called_once_with(slice(None, 3))
    trimmed.to.assert_called_once_with(bf16)


@pytest.mark.parametrize("stored", ["float32", "bfloat16"])
def test_text_encoder_streams_the_same_weights_as_from_pretrained(monkeypatch, tmp_path, stored):
    """Sharded checkpoint -> bf16, in row blocks, with the tied embedding stored once."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from repair_pipeline import caption

    torch.manual_seed(0)
    config = transformers.UMT5Config(vocab_size=64, d_model=16, d_kv=4, d_ff=32, num_layers=2, num_heads=4,
                                     relative_attention_num_buckets=8)
    transformers.UMT5EncoderModel(config).to(getattr(torch, stored)).save_pretrained(tmp_path, max_shard_size="10KB")
    assert len(list(tmp_path.glob("*.safetensors"))) > 1                      # sharded, like Wan's

    monkeypatch.setattr(caption, "LOAD_CHUNK_BYTES", 16 * 4 * 5)              # 5 rows of d_model at a time
    streamed = caption.load_text_encoder(tmp_path, "cpu", torch.bfloat16)
    reference = transformers.UMT5EncoderModel.from_pretrained(tmp_path, dtype=torch.bfloat16)
    got, want = streamed.state_dict(), reference.state_dict()
    assert got.keys() == want.keys()
    for name in want:
        assert got[name].dtype == torch.bfloat16 and torch.equal(got[name], want[name]), name
    assert streamed.encoder.embed_tokens.weight is streamed.shared.weight

    from safetensors.torch import load_file, save_file   # an export that also stores the tied copy
    shard = sorted(tmp_path.glob("*.safetensors"))[0]
    tensors = load_file(shard)
    tensors["encoder.embed_tokens.weight"] = torch.zeros_like(reference.shared.weight, dtype=getattr(torch, stored))
    save_file(tensors, shard, metadata={"format": "pt"})
    assert torch.equal(caption.load_text_encoder(tmp_path, "cpu", torch.bfloat16).shared.weight, want["shared.weight"])
    assert not streamed.training
    ids = torch.tensor([[3, 7, 1, 0]])
    with torch.no_grad():
        assert torch.equal(streamed(ids).last_hidden_state, reference.eval()(ids).last_hidden_state)


def test_safetensors_reader_rejects_a_truncated_file(tmp_path):
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    from repair_pipeline.caption import _read_safetensors

    f = tmp_path / "x.safetensors"
    save_file({"a": torch.arange(12, dtype=torch.float32).view(4, 3), "b": torch.ones(5)}, f)
    blocks = [(n, b.clone(), r) for n, b, r in _read_safetensors(f, 2 * 3 * 4)]   # two rows of a
    assert [(n, r) for n, _, r in blocks] == [("a", slice(0, 2)), ("a", slice(2, 4)), ("b", slice(0, 5))]
    assert torch.equal(torch.cat([b for n, b, _ in blocks if n == "a"]), torch.arange(12.0).view(4, 3))
    f.write_bytes(f.read_bytes()[:-4])
    with pytest.raises(EOFError, match="truncated in b"):
        list(_read_safetensors(f, 1 << 20))
