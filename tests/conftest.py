import builtins
import io
import zipfile

import numpy as np
import pytest
from PIL import Image

from deit_classification_pipeline import pipeline as pipeline_module

TEST_REVISION = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture
def forbid_model_imports(monkeypatch):
    """Rejected requests must stop before importing or initializing model libraries."""
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.partition(".")[0] in {"torch", "transformers", "torchvision", "safetensors"}:
            raise AssertionError(f"model dependency imported before rejection: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


@pytest.fixture
def pinned(monkeypatch):
    """Pretend the package is pinned to TEST_REVISION, so snapshot checks can run on synthetic manifests."""
    monkeypatch.setattr(pipeline_module, "MODEL_REVISION", TEST_REVISION)
    return TEST_REVISION


def colour_image(label: str, seed: int, side: int = 32) -> Image.Image:
    """A small image whose dominant channel encodes the label: green for frog, red for truck."""
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 60, (side, side, 3), dtype=np.uint8)
    pixels[..., 1 if label == "frog" else 0] += 150
    return Image.fromarray(pixels)


def colour_records(per_class: int = 10, seed: int = 0) -> list[dict]:
    return [
        {
            "id": f"{label}/{i:03d}.png",
            "image": colour_image(label, seed * 1000 + i + 100 * k),
            "label": label,
        }
        for k, label in enumerate(("frog", "truck"))
        for i in range(per_class)
    ]


def class_zip(
    per_class: int = 4, *, root: str = "CIFAR-10-subset", extra: dict[str, bytes] | None = None
) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for record in colour_records(per_class):
            png = io.BytesIO()
            record["image"].save(png, format="PNG")
            archive.writestr(f"{root}/{record['id']}", png.getvalue())
        for name, data in (extra or {}).items():
            archive.writestr(name, data)
    return buffer.getvalue()
