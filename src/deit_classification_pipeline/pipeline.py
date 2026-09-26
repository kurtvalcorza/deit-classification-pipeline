"""ImageNet-1k classification and bounded fine-tuning with the ``facebook/deit-small-patch16-224`` checkpoint.

The class loads weights only from a digest-verified local snapshot (``weights/<key>/``). The architecture is
``ViTForImageClassification`` from the pinned ``transformers`` release, built from the snapshot's
``config.json`` with no network access and no remote code. The Hub repository hosts no SafeTensors file, so
the verified ``pytorch_model.bin`` is read with ``torch.load(..., weights_only=True)``, which refuses
arbitrary pickled objects, and loaded with ``strict=True``.

Preprocessing is DeiT's own evaluation transform (bicubic resize of the shorter side to 256 px, centre crop
224, ImageNet mean and standard deviation), not the snapshot's ``preprocessor_config.json``, which specifies
a plain 224 px resize with mean and standard deviation 0.5.

Until ``tools/pin_snapshot.py`` has recorded an immutable revision and every file's SHA-256, the package
refuses to stage, verify or load weights: an unpinned snapshot is never trusted.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .data import validate_dataset

MODEL_ID = "facebook/deit-small-patch16-224"
MODEL_REVISION = "164deee347853469b97442b3817f22eece80c7e3"
MODEL_LICENSE = "apache-2.0"
MODEL_KEY = "deit-small-patch16-224"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"
ARTIFACT_FORMAT = "deit-adapter-v1"
UNPINNED = "unpinned"
PIN_COMMAND = "python tools/pin_snapshot.py"
WEIGHTS_FILE = "pytorch_model.bin"
CONFIG_FILE = "config.json"

NUM_IMAGENET_CLASSES = 1000
MIN_IMAGE_SIDE = 8
MAX_IMAGE_SIDE = 4096
MAX_BATCH = 64  # images per predict() call and per forward pass
MAX_CLASSES = 1000
DEFAULT_TOP_K = 5
DECISION_RULE = (
    "argmax"  # the reported label is the softmax argmax; there is no threshold and no reject option
)

# ImageNet-1k classes that fall inside each tutorial class, for the zero-shot baseline. CIFAR-10 defines
# "truck" as big trucks only and excludes pickup trucks, so pickup (717), minivan (656) and police van
# (734) are left out; "frog" takes all three ImageNet frog classes.
IMAGENET_GROUPS: dict[str, tuple[int, ...]] = {
    "frog": (30, 31, 32),  # bullfrog, tree frog, tailed frog
    "truck": (555, 569, 675, 864, 867),  # fire engine, garbage truck, moving van, tow truck, trailer truck
}

# Bounded tutorial fine-tuning defaults: AdamW on cross-entropy, float32, no augmentation.
DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_SEED = 20260925


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_pinned() -> bool:
    """True once MODEL_REVISION names an immutable 40-hex commit."""
    revision = MODEL_REVISION
    return len(revision) == 40 and all(c in "0123456789abcdef" for c in revision)


def _require_pinned(action: str) -> None:
    if not is_pinned():
        raise RuntimeError(
            f"refusing to {action}: {MODEL_ID} has no pinned revision yet (MODEL_REVISION = "
            f"{MODEL_REVISION!r}); run `{PIN_COMMAND}` to record the commit and every file's SHA-256"
        )


def _read_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        return json.load(fh)


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check a local snapshot against its DIMER manifest; raise naming the first mismatch."""
    _require_pinned("verify the snapshot")
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest = _read_manifest(root)
    if manifest.get("modelId") != MODEL_ID:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {MODEL_ID!r}")
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {MODEL_REVISION!r}")
    for entry in manifest["files"]:
        if not entry.get("sha256"):
            raise ValueError(f"{entry['path']}: manifest records no sha256; run `{PIN_COMMAND}`")
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = _sha256(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
    return {
        "path": str(root),
        "model_id": manifest["modelId"],
        "revision": manifest["revision"],
        "files": len(manifest["files"]),
        "total_bytes": manifest.get("totalBytes"),
        "weights_sha256": next((e["sha256"] for e in manifest["files"] if e["path"] == WEIGHTS_FILE), None),
    }


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at MODEL_REVISION straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent locally (a fresh clone commits the manifest but
    git-ignores the weights). Returns the relative paths fetched; `verify_snapshot` still runs after."""
    _require_pinned("stage weights")
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest = _read_manifest(root)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; pass allow_download=True to fetch them at "
            f"{MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


# --------------------------------------------------------------------------- metrics


def classification_metrics(
    predicted: Sequence[str],
    truth: Sequence[str],
    class_names: Sequence[str],
    *,
    majority_class: str | None = None,
) -> dict[str, Any]:
    """Accuracy, balanced accuracy, per-class recall and the confusion matrix, plus the accuracy of always
    predicting ``majority_class`` (pass the training majority, so the baseline is not tuned on the
    evaluated set)."""
    if len(predicted) != len(truth):
        raise ValueError(f"{len(predicted)} predictions but {len(truth)} labels")
    if not truth:
        raise ValueError("at least one labelled item is needed")
    names = list(class_names)
    index = {name: i for i, name in enumerate(names)}
    matrix = np.zeros((len(names), len(names)), dtype=int)
    for p, t in zip(predicted, truth, strict=True):
        matrix[index[t], index[p]] += 1
    support = matrix.sum(axis=1)
    recall = {
        name: (float(matrix[i, i] / support[i]) if support[i] else None) for i, name in enumerate(names)
    }
    present = [value for value in recall.values() if value is not None]
    result = {
        "accuracy": float(np.trace(matrix) / matrix.sum()),
        "balanced_accuracy": float(np.mean(present)),
        "per_class_recall": recall,
        "support": {name: int(n) for name, n in zip(names, support, strict=True)},
        "confusion_matrix": {"labels": names, "rows_true_cols_predicted": matrix.tolist()},
        "n": int(matrix.sum()),
    }
    if majority_class is not None:
        if majority_class not in index:
            raise ValueError(f"majority_class {majority_class!r} is not one of {names}")
        result["majority_class"] = majority_class
        result["majority_baseline_accuracy"] = float(
            sum(1 for t in truth if t == majority_class) / len(truth)
        )
    return result


def majority_class(records: Sequence[Mapping[str, Any]]) -> str:
    """The most frequent label, ties broken by name, for the majority-class baseline."""
    counts: dict[str, int] = {}
    for record in records:
        counts[record["label"]] = counts.get(record["label"], 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


# --------------------------------------------------------------------------- input validation


def _check_images(images: Any) -> list[Image.Image]:
    if isinstance(images, Image.Image):
        images = [images]
    if not isinstance(images, Sequence) or isinstance(images, str | bytes):
        raise TypeError("images must be a PIL.Image.Image or a sequence of them")
    if not 1 <= len(images) <= MAX_BATCH:
        raise ValueError(f"batch size must be between 1 and MAX_BATCH={MAX_BATCH}, got {len(images)}")
    for image in images:
        if not isinstance(image, Image.Image):
            raise TypeError(f"each image must be a PIL.Image.Image, got {type(image).__name__}")
        if min(image.size) < MIN_IMAGE_SIDE or max(image.size) > MAX_IMAGE_SIDE:
            raise ValueError(
                f"image size {image.size} outside MIN_IMAGE_SIDE..MAX_IMAGE_SIDE = "
                f"{MIN_IMAGE_SIDE}..{MAX_IMAGE_SIDE} px"
            )
    return [image.convert("RGB") for image in images]


def _check_top_k(top_k: Any, n_classes: int) -> int:
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("top_k must be an int")
    if not 1 <= top_k <= n_classes:
        raise ValueError(f"top_k must be between 1 and {n_classes}, got {top_k}")
    return top_k


def _check_class_names(class_names: Sequence[str]) -> tuple[str, ...]:
    names = tuple(class_names)
    if not 2 <= len(names) <= MAX_CLASSES:
        raise ValueError(f"class_names must hold 2..{MAX_CLASSES} names, got {len(names)}")
    for name in names:
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"class names must be non-empty strings, got {name!r}")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"class_names contains duplicates: {duplicates}")
    return names


INPUT_SCHEMA: dict[str, Any] = {
    "input": "PIL.Image.Image or a sequence of them; any mode, converted to RGB",
    "image_side_px": [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE],
    "batch": [1, MAX_BATCH],
    "top_k": [1, NUM_IMAGENET_CLASSES],
    "decision_rule": DECISION_RULE,
    "preprocessing": (
        "RGB; bicubic resize of the shorter side to 256 px, centre crop 224x224, ImageNet mean/std "
        "normalisation (DeiT's evaluation transform; the snapshot's preprocessor_config.json differs)"
    ),
}


def validate_inputs(
    images: Image.Image | Sequence[Image.Image],
    top_k: int = DEFAULT_TOP_K,
    *,
    names: Sequence[str] | None = None,
    n_classes: int = NUM_IMAGENET_CLASSES,
) -> dict[str, Any]:
    """Validation stage: return the input manifest (schema, per-input observations, request, verdict)."""
    checked = _check_images(images)
    _check_top_k(top_k, n_classes)
    if names is not None and len(names) != len(checked):
        raise ValueError("names must have one entry per image")
    originals = [images] if isinstance(images, Image.Image) else list(images)
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": [
            {"id": names[i] if names else f"image-{i}", "mode": image.mode, "size": list(image.size)}
            for i, image in enumerate(originals)
        ],
        "top_k": top_k,
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def evaluation_report(
    result: Mapping[str, Any],
    truth: Sequence[str] | None = None,
    *,
    groups: Mapping[str, Sequence[int]] | None = None,
    sample_kind: str = "sample",
) -> dict[str, Any]:
    """Single-batch evaluation stage for ImageNet-head predictions.

    With ``truth`` (one tutorial class name per image) and ``groups`` (the ImageNet indices each class
    covers), a prediction counts as correct when its top-1 or any of its top-k indices falls inside the
    image's group. Without them the verdict is ``not-measurable`` and the report names the missing data.
    """
    predictions = list(result["predictions"])
    base = {
        "task": f"{NUM_IMAGENET_CLASSES}-class ImageNet-1k single-label classification",
        "decision_rule": result.get("decision_rule", DECISION_RULE),
        "sample_kind": sample_kind,
        "n_predictions": len(predictions),
        "baselines": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
    if truth is None or groups is None:
        return {
            **base,
            "metrics": [],
            "verdict": "not-measurable",
            "reason": "no ground-truth class was supplied for the evaluated images",
            "needs": (
                "labelled images from your own domain, scored with accuracy and balanced accuracy against "
                "the "
                "majority-class baseline of the training split"
            ),
        }
    if len(truth) != len(predictions):
        raise ValueError(f"{len(predictions)} predictions but {len(truth)} labels")
    top_k = int(result.get("top_k", DEFAULT_TOP_K))
    metrics = []
    for k in sorted({1, top_k}):
        hits = [
            any(item["index"] in groups[label] for item in pred["top_k"][:k])
            for pred, label in zip(predictions, truth, strict=True)
        ]
        metrics.append(
            {
                "id": "group_hit_rate",
                "k": k,
                "value": float(np.mean(hits)),
                "estimation": "one pass, no dispersion estimate",
            }
        )
    return {
        **base,
        "metrics": metrics,
        "verdict": "sample-sanity",
        "reason": f"{len(predictions)} labelled image(s) scored against hand-chosen ImageNet class groups; "
        "not a benchmark",
        "needs": "a labelled evaluation set from the deployment domain for any generalisable accuracy claim",
    }


# --------------------------------------------------------------------------- backend (transformers ViT)
# Everything model-specific lives in this section: how the architecture is built, how the pinned state
# dict is loaded, which transform it expects, and which module is the classification head.

ARCHITECTURE = "ViTForImageClassification"
HEAD_MODULE = "classifier"
IMAGE_SIZE = 224
# DeiT's evaluation transform (facebookresearch/deit datasets.py build_transform): resize the shorter side to
# int(256 / 224 * 224) = 256 px with bicubic interpolation, centre-crop 224, normalise with timm's
# IMAGENET_DEFAULT_MEAN / IMAGENET_DEFAULT_STD.
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
EXPECTED_CONFIG = {
    "model_type": "vit",
    "hidden_size": 384,
    "num_hidden_layers": 12,
    "num_attention_heads": 6,
    "patch_size": 16,
    "image_size": 224,
}
# First name of each ImageNet class the zero-shot groups use; the loader refuses a config whose id2label
# puts other classes at these indices.
IMAGENET_GROUP_LABELS = {
    30: "bullfrog",
    31: "tree frog",
    32: "tailed frog",
    555: "fire engine",
    569: "garbage truck",
    675: "moving van",
    864: "tow truck",
    867: "trailer truck",
}


def build_model(num_classes: int = NUM_IMAGENET_CLASSES, **config_overrides: Any) -> Any:
    """``ViTForImageClassification`` with random weights at the DeiT-Small shape (overridable for tests)."""
    from transformers import ViTConfig, ViTForImageClassification

    fields = {key: value for key, value in EXPECTED_CONFIG.items() if key != "model_type"}
    fields.update(intermediate_size=4 * fields["hidden_size"], num_labels=num_classes)
    fields.update(config_overrides)
    return ViTForImageClassification(ViTConfig(**fields))


def eval_transform(model: Any = None, image_size: int | None = None) -> Callable[[Image.Image], Any]:
    """DeiT's evaluation transform at ``image_size`` (default: the model's configured size, else 224)."""
    from torchvision import transforms

    size = image_size or (model.config.image_size if model is not None else IMAGE_SIZE)
    return transforms.Compose(
        [
            transforms.Resize(int(256 / 224 * size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ]
    )


def forward_logits(model: Any, pixel_values: Any) -> Any:
    return model(pixel_values=pixel_values).logits


def imagenet_labels(model: Any) -> tuple[str, ...]:
    """The class names in index order, from the loaded config's ``id2label``."""
    id2label = model.config.id2label
    return tuple(id2label[i] for i in range(len(id2label)))


def check_imagenet_groups(labels: Sequence[str]) -> None:
    """Refuse a label list whose zero-shot group indices do not name the expected ImageNet classes."""
    if len(labels) != NUM_IMAGENET_CLASSES:
        raise ValueError(f"expected {NUM_IMAGENET_CLASSES} ImageNet labels, got {len(labels)}")
    wrong = {
        i: labels[i] for i, name in IMAGENET_GROUP_LABELS.items() if labels[i].split(",")[0].strip() != name
    }
    if wrong:
        raise ValueError(f"id2label does not follow the ImageNet-1k order at the group indices: {wrong}")


def _load_pretrained(root: Path) -> Any:
    import torch
    from transformers import ViTConfig, ViTForImageClassification

    config = ViTConfig.from_pretrained(str(root), local_files_only=True)
    drift = {
        key: getattr(config, key) for key, value in EXPECTED_CONFIG.items() if getattr(config, key) != value
    }
    if drift or config.num_labels != NUM_IMAGENET_CLASSES or ARCHITECTURE not in (config.architectures or []):
        raise ValueError(f"snapshot config.json does not describe DeiT-Small with an ImageNet head: {drift}")
    model = ViTForImageClassification(config)
    state = torch.load(root / WEIGHTS_FILE, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    check_imagenet_groups(imagenet_labels(model))
    return model


def _rehead(model: Any, num_classes: int) -> tuple[str, ...]:
    import torch

    model.classifier = torch.nn.Linear(model.config.hidden_size, num_classes)
    model.config.num_labels = num_classes
    return (f"{HEAD_MODULE}.",)


def _backbone_prefixes(model: Any) -> tuple[str, ...]:
    return tuple(f"{name}." for name, _ in model.named_children() if name != HEAD_MODULE)


# --------------------------------------------------------------------------- pipeline


def _hold_batchnorm(modules: Sequence[Any]) -> None:
    """Keep BatchNorm layers in eval mode, so frozen layers keep their running statistics."""
    import torch

    for module in modules:
        for sub in module.modules():
            if isinstance(sub, torch.nn.modules.batchnorm._BatchNorm):
                sub.eval()


@dataclass
class DeiTPipeline:
    """ImageNet-1k classification, and transfer fine-tuning onto a caller's classes, over DeiT-Small."""

    model: Any
    transform: Callable[[Image.Image], Any]
    device: str
    class_names: tuple[str, ...]
    source: str = "snapshot"
    base_state_digest: str | None = None
    adapted: bool = False
    reinitialised: tuple[str, ...] = field(default_factory=tuple)
    frozen_prefixes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def imagenet_head(self) -> bool:
        return len(self.class_names) == NUM_IMAGENET_CLASSES and not self.reinitialised and not self.adapted

    @classmethod
    def from_components(
        cls,
        model: Any,
        *,
        class_names: Sequence[str],
        device: str = "cpu",
        transform: Callable[[Image.Image], Any] | None = None,
        source: str = "components",
    ) -> DeiTPipeline:
        """Wrap an already-built model; the transform defaults to the model's own evaluation transform."""
        return cls(
            model=model.to(device).eval(),
            transform=transform or eval_transform(model),
            device=device,
            class_names=tuple(class_names),
            source=source,
        )

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
        class_names: Sequence[str] | None = None,
        seed: int = DEFAULT_SEED,
    ) -> DeiTPipeline:
        """Stage (only with ``allow_download``), verify and load the pinned checkpoint. With ``class_names``,
        replace the 1000-class head with a new one for those classes, initialised under ``seed``."""
        _require_pinned("load the model")
        root = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
        if not (root / MANIFEST_NAME).is_file():
            raise FileNotFoundError(
                f"no snapshot manifest at {root}; stage {MODEL_ID} under weights/{MODEL_KEY} "
                "(allow_download=True fetches the manifest-listed files)"
            )
        stage_missing_files(root, allow_download=allow_download)
        snapshot = verify_snapshot(root)
        names = _check_class_names(class_names) if class_names is not None else None

        import torch

        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        model = _load_pretrained(root)
        reinitialised: tuple[str, ...] = ()
        if names is not None:
            torch.manual_seed(seed)
            reinitialised = _rehead(model, len(names))
        pipe = cls.from_components(
            model,
            class_names=names or imagenet_labels(model),
            device=resolved_device,
            transform=eval_transform(model),
            source=str(root),
        )
        pipe.base_state_digest = snapshot["weights_sha256"]
        pipe.reinitialised = reinitialised
        return pipe

    def _probabilities(self, images: Sequence[Image.Image]) -> Any:
        import torch

        was_training = self.model.training
        self.model.eval()
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(images), MAX_BATCH):
                batch = torch.stack(
                    [self.transform(image) for image in images[start : start + MAX_BATCH]]
                ).to(self.device)
                outputs.append(torch.softmax(forward_logits(self.model, batch).float(), dim=-1).cpu())
        if was_training:
            self.model.train()
        probabilities = torch.cat(outputs)
        if probabilities.shape != (len(images), len(self.class_names)):
            raise RuntimeError(
                f"model returned {tuple(probabilities.shape)}, expected ({len(images)}, "
                f"{len(self.class_names)})"
            )
        return probabilities

    def predict(
        self, images: Image.Image | Sequence[Image.Image], top_k: int | None = None
    ) -> dict[str, Any]:
        """Classify images; ``score`` is a softmax over this pipeline's classes, not a calibrated value."""
        import torch

        batch = _check_images(images)
        resolved = (
            min(DEFAULT_TOP_K, len(self.class_names))
            if top_k is None
            else _check_top_k(top_k, len(self.class_names))
        )
        values, indices = torch.topk(self._probabilities(batch), k=resolved, dim=-1)
        predictions = []
        for row_values, row_indices in zip(values.tolist(), indices.tolist(), strict=True):
            ranked = [
                {"label": self.class_names[i], "index": int(i), "score": float(s)}
                for s, i in zip(row_values, row_indices, strict=True)
            ]
            predictions.append(
                {
                    "predicted_label": ranked[0]["label"],
                    "predicted_index": ranked[0]["index"],
                    "top_k": ranked,
                }
            )
        return {
            "predictions": predictions,
            "top_k": resolved,
            "decision_rule": DECISION_RULE,
            "class_names_count": len(self.class_names),
            "adapted": self.adapted,
            "device": self.device,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }

    def zero_shot_evaluate(
        self,
        records: Sequence[Mapping[str, Any]],
        groups: Mapping[str, Sequence[int]] = IMAGENET_GROUPS,
        *,
        majority: str | None = None,
    ) -> dict[str, Any]:
        """Score the ImageNet head on tutorial classes without training: each image goes to the class whose
        ImageNet group holds the most softmax mass. Needs the unmodified 1000-class head."""
        if not self.imagenet_head:
            raise RuntimeError("zero_shot_evaluate needs the pretrained 1000-class ImageNet head")
        names = list(groups)
        missing = sorted({r["label"] for r in records} - set(names))
        if missing:
            raise ValueError(f"records carry labels with no ImageNet group: {missing}")
        probabilities = self._probabilities([r["image"].convert("RGB") for r in records])
        mass = np.stack([probabilities[:, list(groups[name])].sum(dim=1).numpy() for name in names], axis=1)
        predicted = [names[int(i)] for i in mass.argmax(axis=1)]
        truth = [r["label"] for r in records]
        metrics = classification_metrics(predicted, truth, names, majority_class=majority)
        return {
            **metrics,
            "rule": "argmax of summed ImageNet softmax mass per class group",
            "groups": {name: list(ids) for name, ids in groups.items()},
            "mean_group_mass": float(mass.sum(axis=1).mean()),
            "adapted": False,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }

    def finetune(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
        seed: int = DEFAULT_SEED,
        freeze_backbone: bool = False,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Bounded cross-entropy fine-tuning on ``records`` (``{"image", "label"}``), mutating this pipeline.

        With ``freeze_backbone`` only the head trains and the backbone keeps its weights (ViT has no
        BatchNorm; any BatchNorm outside the head would be held in eval mode); otherwise the whole network
        trains.
        The inputs use the evaluation transform: there is no data augmentation.
        """
        import torch
        import torch.nn.functional as F

        validate_dataset(records, self.class_names, epochs=epochs)
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError(f"batch_size must be a positive int, got {batch_size!r}")
        if (
            isinstance(learning_rate, bool)
            or not isinstance(learning_rate, int | float)
            or not 0.0 < float(learning_rate) <= 1.0
        ):
            raise ValueError(f"learning_rate must be a number in (0, 1], got {learning_rate!r}")

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        rng = np.random.default_rng(seed)
        backbone = _backbone_prefixes(self.model)
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad = not (freeze_backbone and name.startswith(backbone))
        self.frozen_prefixes = backbone if freeze_backbone else ()
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=float(learning_rate), weight_decay=float(weight_decay))
        class_to_id = {name: i for i, name in enumerate(self.class_names)}
        frozen_modules = [
            module for name, module in self.model.named_children() if f"{name}." in self.frozen_prefixes
        ]

        epoch_losses: list[float] = []
        for epoch in range(epochs):
            self.model.train()
            _hold_batchnorm(frozen_modules)
            order = rng.permutation(len(records))
            running, seen = 0.0, 0
            for start in range(0, len(records), batch_size):
                batch = [records[int(i)] for i in order[start : start + batch_size]]
                inputs = torch.stack([self.transform(r["image"].convert("RGB")) for r in batch]).to(
                    self.device
                )
                targets = torch.tensor(
                    [class_to_id[r["label"]] for r in batch], dtype=torch.long, device=self.device
                )
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(forward_logits(self.model, inputs), targets)
                loss.backward()
                optimizer.step()
                running += float(loss.detach().cpu()) * len(batch)
                seen += len(batch)
            epoch_losses.append(running / max(1, seen))
            if progress is not None:
                progress({"epoch": epoch + 1, "epochs": epochs, "loss": epoch_losses[-1]})
        self.model.eval()
        self.adapted = True
        return {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "optimizer": "AdamW",
            "loss": "cross-entropy",
            "augmentation": "none (evaluation transform)",
            "seed": seed,
            "precision": "float32",
            "freeze_backbone": freeze_backbone,
            "frozen_prefixes": list(self.frozen_prefixes),
            "trainable_parameters": sum(p.numel() for p in trainable),
            "total_parameters": sum(p.numel() for p in self.model.parameters()),
            "epoch_losses": epoch_losses,
            "final_loss": epoch_losses[-1] if epoch_losses else None,
            "device": self.device,
            "class_names": list(self.class_names),
            "reinitialised_prefixes": list(self.reinitialised),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }

    def evaluate(
        self, records: Sequence[Mapping[str, Any]], *, majority: str | None = None
    ) -> dict[str, Any]:
        """Score labelled records with this pipeline's head: accuracy, balanced accuracy, per-class recall,
        confusion matrix, and the accuracy of always answering ``majority`` (pass the training majority)."""
        unknown = sorted({r["label"] for r in records} - set(self.class_names))
        if unknown:
            raise ValueError(f"records carry labels outside this pipeline's classes: {unknown}")
        images = [r["image"].convert("RGB") for r in records]
        probabilities = self._probabilities(images)
        predicted = [self.class_names[int(i)] for i in probabilities.argmax(dim=-1).tolist()]
        metrics = classification_metrics(
            predicted, [r["label"] for r in records], self.class_names, majority_class=majority
        )
        return {
            **metrics,
            "mean_top1_score": float(probabilities.max(dim=-1).values.mean()),
            "adapted": self.adapted,
            "estimation": f"one pass over {len(records)} held-out images; no resampling, no dispersion "
            "estimate",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }

    def save_artifact(self, path: str | Path, *, notes: str | None = None) -> dict[str, Any]:
        """Write the adapted tensors as one SafeTensors file with the provenance in its metadata.

        Tensors under ``frozen_prefixes`` are left out: they equal the verified base checkpoint, which
        ``load_artifact`` loads first. The artifact is therefore an adapter bound to the base checkpoint.
        """
        from safetensors.torch import save_file

        if not self.adapted:
            raise RuntimeError("nothing to export: the pipeline has not been fine-tuned")
        artifact_path = Path(path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        tensors = {
            name: value.detach().cpu().contiguous().clone()
            for name, value in self.model.state_dict().items()
            if not any(name.startswith(prefix) for prefix in self.frozen_prefixes)
        }
        metadata = {
            "format": ARTIFACT_FORMAT,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_key": MODEL_KEY,
            "class_names": json.dumps(list(self.class_names)),
            "frozen_prefixes": json.dumps(list(self.frozen_prefixes)),
            "base_state_digest": self.base_state_digest or "",
            "notes": notes or "",
        }
        save_file(tensors, str(artifact_path), metadata=metadata)
        return {
            "path": str(artifact_path),
            "bytes": artifact_path.stat().st_size,
            "sha256": _sha256(artifact_path),
            "format": ARTIFACT_FORMAT,
            "class_names": list(self.class_names),
            "tensors": len(tensors),
            "frozen_prefixes": list(self.frozen_prefixes),
            "base_state_digest": self.base_state_digest,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }

    @staticmethod
    def read_artifact_metadata(path: str | Path) -> dict[str, Any]:
        """Read and check the artifact's provenance header without loading any tensor."""
        from safetensors import safe_open

        with safe_open(str(path), framework="pt") as handle:
            metadata = dict(handle.metadata() or {})
        if metadata.get("format") != ARTIFACT_FORMAT:
            raise ValueError(f"artifact format {metadata.get('format')!r} != {ARTIFACT_FORMAT!r}")
        if (metadata.get("model_id"), metadata.get("model_revision")) != (MODEL_ID, MODEL_REVISION):
            raise ValueError(
                f"artifact was built on {metadata.get('model_id')}@{metadata.get('model_revision')}, "
                f"package pins {MODEL_ID}@{MODEL_REVISION}"
            )
        if metadata.get("model_key") != MODEL_KEY:
            raise ValueError(
                f"artifact was built on {metadata.get('model_key')!r}, package pins {MODEL_KEY!r}"
            )
        return {
            **metadata,
            "class_names": json.loads(metadata["class_names"]),
            "frozen_prefixes": json.loads(metadata["frozen_prefixes"]),
        }

    def apply_artifact(self, path: str | Path) -> None:
        """Load adapter tensors onto this base, re-headed pipeline; refuse a tensor set that does not fit."""
        from safetensors.torch import load_file

        metadata = self.read_artifact_metadata(path)
        if tuple(metadata["class_names"]) != tuple(self.class_names):
            raise ValueError("artifact class_names differ from this pipeline's class_names")
        expected_base = metadata.get("base_state_digest") or None
        if expected_base and self.base_state_digest and expected_base != self.base_state_digest:
            raise ValueError("artifact was exported against a different base checkpoint digest")
        tensors = load_file(str(path), device="cpu")
        prefixes = tuple(metadata["frozen_prefixes"])
        result = self.model.load_state_dict(tensors, strict=False)
        if result.unexpected_keys:
            raise ValueError(
                f"artifact carries tensors the model does not have: {result.unexpected_keys[:5]}"
            )
        stray = [key for key in result.missing_keys if not any(key.startswith(p) for p in prefixes)]
        if stray:
            raise ValueError(f"artifact is missing trainable tensors: {stray[:5]}")
        self.model.eval()
        self.adapted = True
        self.frozen_prefixes = prefixes
        self.source = f"artifact:{Path(path).name}"

    @classmethod
    def load_artifact(
        cls, path: str | Path, *, weights_dir: str | Path | None = None, device: str | None = None
    ) -> DeiTPipeline:
        """Rebuild an adapted pipeline: verified base checkpoint first, then the adapter tensors."""
        metadata = cls.read_artifact_metadata(path)
        pipe = cls.from_pretrained(
            device=device, weights_dir=weights_dir, class_names=metadata["class_names"]
        )
        pipe.apply_artifact(path)
        return pipe
