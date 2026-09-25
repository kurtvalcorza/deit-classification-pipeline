import hashlib
import json
import re
from pathlib import Path

import pytest
from PIL import Image

from conftest import colour_image
from deit_classification_pipeline import (
    ARTIFACT_FORMAT,
    DEFAULT_WEIGHTS_DIR,
    IMAGENET_GROUPS,
    INPUT_SCHEMA,
    MAX_BATCH,
    MAX_IMAGE_SIDE,
    MIN_IMAGE_SIDE,
    MODEL_ID,
    MODEL_KEY,
    MODEL_REVISION,
    DeiTPipeline,
    classification_metrics,
    evaluation_report,
    is_pinned,
    majority_class,
    stage_missing_files,
    validate_inputs,
    verify_snapshot,
)
from deit_classification_pipeline import pipeline as pipeline_module

HEX40 = re.compile(r"^[0-9a-f]{40}$")
REPO = Path(__file__).resolve().parents[1]
SNAPSHOT = REPO / "weights" / MODEL_KEY


def test_identity_constants_agree_with_the_committed_manifest():
    manifest = json.loads((SNAPSHOT / "dimer-base-manifest.json").read_text(encoding="utf-8"))
    assert MODEL_ID == "facebook/deit-small-patch16-224" == manifest["modelId"]
    assert manifest["revision"] == pipeline_module.MODEL_REVISION
    assert is_pinned() == bool(HEX40.match(pipeline_module.MODEL_REVISION))
    assert DEFAULT_WEIGHTS_DIR == SNAPSHOT and ARTIFACT_FORMAT == "deit-adapter-v1"
    assert manifest["totalBytes"] == sum(entry["bytes"] for entry in manifest["files"])
    assert {entry["path"] for entry in manifest["files"]} == {
        "README.md", "config.json", "preprocessor_config.json", "pytorch_model.bin",
    }  # fmt: skip
    assert pipeline_module.WEIGHTS_FILE == "pytorch_model.bin"
    [reference] = manifest["referenceFiles"]
    assert reference["path"] == "tf_model.h5" and "not loaded" in reference["note"]
    if not is_pinned():
        assert all(entry["sha256"] is None for entry in manifest["files"] + manifest["referenceFiles"])


def test_pipeline_uses_deit_eval_transform_not_the_snapshot_preprocessor_config():
    raw = (SNAPSHOT / "preprocessor_config.json").read_bytes()
    manifest = json.loads((SNAPSHOT / "dimer-base-manifest.json").read_text(encoding="utf-8"))
    assert len(raw) == next(e["bytes"] for e in manifest["files"] if e["path"] == "preprocessor_config.json")
    hub = json.loads(raw)
    assert (hub["size"], hub["image_mean"], hub["image_std"]) == (
        224,
        [0.5] * 3,
        [0.5] * 3,
    ) and "crop_size" not in hub
    pytest.importorskip("torchvision")
    steps = pipeline_module.eval_transform().transforms
    assert [type(s).__name__ for s in steps] == ["Resize", "CenterCrop", "ToTensor", "Normalize"]
    assert steps[0].size == 256 and steps[0].interpolation.value == "bicubic" and steps[1].size == (224, 224)
    assert tuple(steps[3].mean) == (0.485, 0.456, 0.406) and tuple(steps[3].std) == (0.229, 0.224, 0.225)


def _labels_with_groups() -> list[str]:
    labels = [f"class {i}" for i in range(1000)]
    for index, name in pipeline_module.IMAGENET_GROUP_LABELS.items():
        labels[index] = f"{name}, extra synonym"
    return labels


def test_imagenet_group_labels_cover_the_groups_and_are_checked():
    assert set(pipeline_module.IMAGENET_GROUP_LABELS) == {i for ids in IMAGENET_GROUPS.values() for i in ids}
    assert 717 not in IMAGENET_GROUPS["truck"]  # CIFAR-10 excludes pickup trucks
    pipeline_module.check_imagenet_groups(_labels_with_groups())
    shifted = _labels_with_groups()
    shifted[30], shifted[31] = shifted[31], shifted[30]
    with pytest.raises(ValueError, match="ImageNet-1k order"):
        pipeline_module.check_imagenet_groups(shifted)
    with pytest.raises(ValueError, match="1000 ImageNet labels"):
        pipeline_module.check_imagenet_groups(["a", "b"])


def test_loader_strict_loads_a_full_size_checkpoint_and_refuses_drift(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    model = pipeline_module.build_model()
    assert sum(p.numel() for p in model.parameters()) == 22_050_664  # the Hub card's "22M"
    model.config.architectures = ["ViTForImageClassification"]
    model.config.id2label = dict(enumerate(_labels_with_groups()))
    model.config.label2id = {v: k for k, v in model.config.id2label.items()}
    model.config.save_pretrained(tmp_path)
    torch.save(model.state_dict(), tmp_path / "pytorch_model.bin")
    loaded = pipeline_module._load_pretrained(tmp_path)
    assert all(torch.equal(v, loaded.state_dict()[k]) for k, v in model.state_dict().items())
    state = model.state_dict()
    state.pop("classifier.bias")
    torch.save(state, tmp_path / "pytorch_model.bin")
    with pytest.raises(RuntimeError, match="classifier.bias"):
        pipeline_module._load_pretrained(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    config["num_hidden_layers"] = 6
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="does not describe DeiT-Small"):
        pipeline_module._load_pretrained(tmp_path)


def test_unpinned_package_refuses_every_weight_operation(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_module, "MODEL_REVISION", "unpinned")
    for call in (
        lambda: verify_snapshot(tmp_path),
        lambda: stage_missing_files(tmp_path, allow_download=True, downloader=lambda *_: None),
        lambda: DeiTPipeline.from_pretrained(weights_dir=tmp_path),
    ):
        with pytest.raises(RuntimeError, match="pin_snapshot.py"):
            call()


def _write_snapshot(
    root: Path, revision: str, content: bytes, sha: str | None = None, size: int | None = None
) -> None:
    (root / "config.json").write_bytes(content)
    manifest = {
        "modelId": MODEL_ID,
        "revision": revision,
        "files": [
            {
                "path": "config.json",
                "bytes": len(content) if size is None else size,
                "sha256": hashlib.sha256(content).hexdigest() if sha is None else sha,
            }
        ],
        "totalBytes": len(content),
    }
    (root / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_verify_snapshot_accepts_matching_and_rejects_every_mismatch(tmp_path, pinned):
    _write_snapshot(tmp_path, pinned, b"{}")
    assert verify_snapshot(tmp_path)["files"] == 1
    _write_snapshot(tmp_path, pinned, b"{}", sha="0" * 64)
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(tmp_path)
    _write_snapshot(tmp_path, pinned, b"{}", size=99)
    with pytest.raises(ValueError, match="size"):
        verify_snapshot(tmp_path)
    _write_snapshot(tmp_path, "f" * 40, b"{}")
    with pytest.raises(ValueError, match="revision"):
        verify_snapshot(tmp_path)
    _write_snapshot(tmp_path, pinned, b"{}")
    manifest = json.loads((tmp_path / "dimer-base-manifest.json").read_text())
    manifest["files"][0]["sha256"] = None
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="no sha256"):
        verify_snapshot(tmp_path)
    _write_snapshot(tmp_path, pinned, b"{}")
    (tmp_path / "config.json").unlink()
    with pytest.raises(FileNotFoundError):
        verify_snapshot(tmp_path)


def test_stage_missing_files_fetches_only_absent_entries(tmp_path, pinned):
    payload = b"weights"
    _write_snapshot(tmp_path, pinned, b"{}")
    manifest = json.loads((tmp_path / "dimer-base-manifest.json").read_text())
    manifest["files"].append(
        {
            "path": pipeline_module.WEIGHTS_FILE,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    )
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(tmp_path)
    fetched = []

    def fake(relative, root):
        fetched.append(relative)
        (root / relative).write_bytes(payload)

    assert (
        stage_missing_files(tmp_path, allow_download=True, downloader=fake)
        == [pipeline_module.WEIGHTS_FILE]
        == fetched
    )
    assert verify_snapshot(tmp_path)["weights_sha256"] == hashlib.sha256(payload).hexdigest()
    manifest["modelId"] = "someone/else"
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_files(tmp_path, allow_download=True, downloader=fake)


def test_classification_metrics_and_majority_baseline():
    metrics = classification_metrics(
        ["frog", "frog", "truck", "frog"],
        ["frog", "truck", "truck", "truck"],
        ["frog", "truck"],
        majority_class="truck",
    )
    assert metrics["accuracy"] == 0.5 and metrics["per_class_recall"] == {"frog": 1.0, "truck": 1 / 3}
    assert metrics["balanced_accuracy"] == pytest.approx((1.0 + 1 / 3) / 2)
    assert metrics["confusion_matrix"]["rows_true_cols_predicted"] == [[1, 0], [2, 1]]
    assert metrics["majority_baseline_accuracy"] == 0.75
    assert classification_metrics(["a"], ["a"], ["a", "b"])["per_class_recall"]["b"] is None
    with pytest.raises(ValueError, match="majority_class"):
        classification_metrics(["a"], ["a"], ["a", "b"], majority_class="c")
    with pytest.raises(ValueError, match="predictions"):
        classification_metrics(["a"], [], ["a", "b"])
    assert majority_class([{"label": "b"}, {"label": "a"}, {"label": "b"}]) == "b"
    assert majority_class([{"label": "b"}, {"label": "a"}]) == "a"


def test_validate_inputs_manifest_and_rejections():
    manifest = validate_inputs(
        [colour_image("frog", 0), Image.new("L", (40, 20))], top_k=3, names=["a.png", "b.png"]
    )
    assert manifest["schema"] == INPUT_SCHEMA and manifest["verdict"] == "accepted" and manifest["top_k"] == 3
    assert manifest["inputs"][1] == {"id": "b.png", "mode": "L", "size": [40, 20]}
    assert (manifest["model_id"], manifest["model_revision"]) == (MODEL_ID, MODEL_REVISION)
    for images, top_k, error, message in (
        ("x.png", 5, TypeError, "PIL.Image.Image"),
        ([Image.new("RGB", (MIN_IMAGE_SIDE - 1, 20))], 5, ValueError, "outside"),
        ([Image.new("RGB", (MAX_IMAGE_SIDE + 1, 20))], 5, ValueError, "outside"),
        ([colour_image("frog", 0)] * (MAX_BATCH + 1), 5, ValueError, "MAX_BATCH"),
        ([colour_image("frog", 0)], 0, ValueError, "top_k"),
        ([colour_image("frog", 0)], True, TypeError, "top_k"),
    ):
        with pytest.raises(error, match=message):
            validate_inputs(images, top_k)
    with pytest.raises(ValueError, match="one entry per image"):
        validate_inputs([colour_image("frog", 0)], names=["a", "b"])


def _result(indices):
    return {
        "predictions": [
            {"top_k": [{"index": i, "label": str(i), "score": 0.5} for i in row]} for row in indices
        ],
        "top_k": 2,
    }


def test_evaluation_report_verdicts():
    report = evaluation_report(_result([[30, 1]]))
    assert (
        report["verdict"] == "not-measurable"
        and report["metrics"] == []
        and "majority-class" in report["needs"]
    )
    report = evaluation_report(
        _result([[30, 1], [1, 867], [5, 6]]), ["frog", "truck", "truck"], groups=IMAGENET_GROUPS
    )
    assert report["verdict"] == "sample-sanity"
    assert [(m["k"], m["value"]) for m in report["metrics"]] == [
        (1, pytest.approx(1 / 3)),
        (2, pytest.approx(2 / 3)),
    ]
    with pytest.raises(ValueError, match="labels"):
        evaluation_report(_result([[30, 1]]), ["frog", "truck"], groups=IMAGENET_GROUPS)
