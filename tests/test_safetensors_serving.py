"""The model is served from the pinned safetensors conversion; the pickle source is read only to convert it.

The conversion tests swap the ViT for a two-layer stand-in, so they run offline in CI. The last test runs
only where the real snapshot and its conversion are staged under weights/.
"""

from __future__ import annotations

import hashlib
import inspect
import json

import pytest

from deit_classification_pipeline import pipeline as pipeline_module
from deit_classification_pipeline.pipeline import (
    CONVERTED_SHA256,
    CONVERTED_WEIGHTS_FILE,
    DEFAULT_WEIGHTS_DIR,
    MANIFEST_NAME,
    MODEL_ID,
    WEIGHTS_FILE,
    DeiTPipeline,
)


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_converted_file_is_pinned_safetensors():
    assert CONVERTED_WEIGHTS_FILE.endswith(".safetensors")
    for digest in (CONVERTED_SHA256, pipeline_module.SOURCE_WEIGHTS_SHA256):
        assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)
    manifest = json.loads((DEFAULT_WEIGHTS_DIR / MANIFEST_NAME).read_text(encoding="utf-8"))
    entry = next(e for e in manifest["files"] if e["path"] == WEIGHTS_FILE)
    assert entry["sha256"] == pipeline_module.SOURCE_WEIGHTS_SHA256


def test_validator_declares_the_converted_digest():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "tools" / "validate_release_assets.py"
    spec = importlib.util.spec_from_file_location("validate_release_assets", path)
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    assert CONVERTED_SHA256 in validator.EXTERNAL_WEIGHT_DIGESTS


def test_serving_path_never_unpickles():
    for function in (DeiTPipeline.from_pretrained, pipeline_module._load_pretrained):
        source = inspect.getsource(function)
        assert "torch.load" not in source
    assert "strict=True" in inspect.getsource(pipeline_module._load_pretrained)


@pytest.fixture
def stand_in(tmp_path, pinned, monkeypatch):
    """A pinned two-layer 'pytorch_model.bin' plus the digest its conversion must have."""
    torch = pytest.importorskip("torch")

    def build(_root=None):
        return torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.LayerNorm(4))

    torch.manual_seed(0)
    state = build().state_dict()
    root = tmp_path / "weights"
    root.mkdir()
    torch.save(state, root / WEIGHTS_FILE)
    (root / "config.json").write_text("{}", encoding="utf-8")
    files = [
        {"path": name, "bytes": (root / name).stat().st_size, "sha256": _sha(root / name)}
        for name in ("config.json", WEIGHTS_FILE)
    ]
    manifest = {"modelId": MODEL_ID, "revision": pinned, "files": files}
    (root / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    reference = tmp_path / "reference.safetensors"
    pipeline_module._write_converted(state, reference)
    monkeypatch.setattr(pipeline_module, "SOURCE_WEIGHTS_SHA256", files[1]["sha256"])
    monkeypatch.setattr(pipeline_module, "CONVERTED_SHA256", _sha(reference))
    monkeypatch.setattr(pipeline_module, "_build_from_config", build)
    return torch, root, state


def test_conversion_is_byte_reproducible_and_strict(stand_in):
    _torch, root, _state = stand_in
    first = pipeline_module.convert_checkpoint(root)["converted"]["sha256"]
    (root / CONVERTED_WEIGHTS_FILE).unlink()
    second = pipeline_module.convert_checkpoint(root)["converted"]["sha256"]
    assert first == second == pipeline_module.CONVERTED_SHA256
    assert not list(root.glob("*.partial"))


def test_conversion_with_another_digest_is_refused_and_leaves_nothing(stand_in, monkeypatch):
    _torch, root, _state = stand_in
    monkeypatch.setattr(pipeline_module, "CONVERTED_SHA256", "f" * 64)
    with pytest.raises(ValueError, match="not the pinned"):
        pipeline_module.convert_checkpoint(root)
    assert not (root / CONVERTED_WEIGHTS_FILE).exists() and not list(root.glob("*.partial"))


def test_a_source_other_than_the_pinned_one_is_refused(stand_in, monkeypatch):
    _torch, root, _state = stand_in
    monkeypatch.setattr(pipeline_module, "SOURCE_WEIGHTS_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="!= pinned"):
        pipeline_module.convert_checkpoint(root)


def test_once_converted_the_snapshot_needs_no_pickle_and_never_unpickles(stand_in, monkeypatch):
    torch, root, _state = stand_in
    pipeline_module.convert_checkpoint(root)
    (root / WEIGHTS_FILE).unlink()

    def refuse(*_args, **_kwargs):
        raise AssertionError("torch.load called on the serving path")

    monkeypatch.setattr(torch, "load", refuse)
    assert pipeline_module.stage_missing_files(root, include_source=False) == []
    assert pipeline_module.verify_snapshot(root, include_source=False)["files"] == 2
    assert pipeline_module.verify_converted(root)["sha256"] == pipeline_module.CONVERTED_SHA256
    with pytest.raises(FileNotFoundError):
        pipeline_module.verify_snapshot(root)


@pytest.mark.skipif(
    not all((DEFAULT_WEIGHTS_DIR / name).is_file() for name in (WEIGHTS_FILE, CONVERTED_WEIGHTS_FILE)),
    reason="real snapshot and conversion not staged",
)
def test_real_conversion_equals_the_source_checkpoint():
    torch = pytest.importorskip("torch")
    from safetensors.torch import load_file

    pipeline_module.verify_snapshot(DEFAULT_WEIGHTS_DIR)
    pipeline_module.verify_converted(DEFAULT_WEIGHTS_DIR)
    source = torch.load(DEFAULT_WEIGHTS_DIR / WEIGHTS_FILE, map_location="cpu", weights_only=True)
    converted = load_file(str(DEFAULT_WEIGHTS_DIR / CONVERTED_WEIGHTS_FILE))
    assert set(source) == set(converted)
    assert all(torch.equal(source[k], converted[k]) and source[k].dtype == converted[k].dtype for k in source)
