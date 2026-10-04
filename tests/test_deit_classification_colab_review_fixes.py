# ruff: noqa: E501  -- assertion lines name the exported fields in full
"""Regression tests for the 2026-10-02 review of deit_classification_colab (DEIT-M1..M4, DEIT-m1..m3).

The data-level tests need NumPy and Pillow only. The notebook tests execute the committed .ipynb's own learner
cells (Sections 4-13) in a namespace built from the package, with a small random-weight DeiT standing in
for the pinned checkpoint (the same class and code paths as `tests/test_small_model.py`) and a synthetic archive
laid out like the pinned CIFAR-10 subset (original and darkened copies). The stand-in says nothing about
classification quality; these tests check control flow, the duplicate-aware split, the exported fields and the
BYOD file handling. The real-weights checks are recorded in docs/release-verification.md.
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import sys
import types
import zipfile
from pathlib import Path

import pytest

with contextlib.suppress(ImportError):  # torch before any NumPy linear algebra (Windows DLL load order)
    import torch

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from conftest import colour_image, colour_records  # noqa: E402
from deit_classification_pipeline import data as data_module  # noqa: E402
from deit_classification_pipeline.data import (  # noqa: E402
    NEAR_DUPLICATE_CORRELATION,
    assign_duplicate_groups,
    cross_split_duplicates,
    split_dataset,
)

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "deit_classification_colab.ipynb"
needs_torch = pytest.mark.skipif("torch" not in sys.modules, reason="torch is not installed")


def _photo(label: str, seed: int) -> Image.Image:
    """A smooth 32 px image with photograph-like structure (bicubic upsampling of random 4×4 colours)."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, (4, 4, 3), dtype=np.uint8)
    channel = 1 if label == "frog" else 0
    small[..., channel] = small[..., channel] // 2 + 120
    return Image.fromarray(small).resize((32, 32), Image.BICUBIC)


def _darkened(image: Image.Image, factor: float = 0.3) -> Image.Image:
    return Image.fromarray((np.asarray(image, dtype=np.float64) * factor).astype(np.uint8))


def _archive_like_records(n: int = 8) -> list[dict]:
    """Like the pinned sample: every photograph twice, half pixel-identical, half darkened."""
    records = []
    for k, label in enumerate(("frog", "truck")):
        for i in range(n):
            image = _photo(label, 500 + 100 * k + i)
            copy = image.copy() if i % 2 == 0 else _darkened(image)
            records.append({"id": f"original_images/{label}/image_{i}.png", "image": image, "label": label})
            records.append({"id": f"darkened_images/{label}/image_{i}.png", "image": copy, "label": label})
    return records


# ---------------------------------------------------------------- DEIT-M2: duplicate-aware split (data level)


def test_copies_of_one_photograph_share_a_group():
    grouped, summary = assign_duplicate_groups(_archive_like_records())
    assert summary["records"] == 32 and summary["groups"] == 16 and summary["groups_with_duplicates"] == 16
    assert summary["exact_copies"] == 8 and summary["near_duplicate_links"] == 8 and summary["mixed_label_groups"] == 0
    by_id = {r["id"]: r["group"] for r in grouped}
    for record in grouped:
        twin = record["id"].replace("original_images", "darkened_images") if "original" in record["id"] else record["id"].replace("darkened_images", "original_images")
        assert by_id[twin] == record["group"]
    assert all("group" not in r for r in _archive_like_records(1))  # the input records are not modified


def test_different_photographs_stay_apart():
    _grouped, summary = assign_duplicate_groups(colour_records(10))
    assert summary["groups"] == 20 and summary["groups_with_duplicates"] == 0


def test_group_split_keeps_every_copy_on_one_side_and_the_pixels_confirm_it():
    grouped, _summary = assign_duplicate_groups(_archive_like_records())
    train, rest = split_dataset(grouped, train_fraction=0.7, seed=0, group_key="group")
    held, unseen = split_dataset(rest, train_fraction=0.5, seed=0, group_key="group")
    groups = [{r["group"] for r in part} for part in (train, held, unseen)]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert sorted(len(p) for p in (train, held, unseen)) == [4, 4, 24]
    report = cross_split_duplicates(train, {"held_out": held, "unseen": unseen})
    assert all(row["pixel_copy_in_reference"] == 0 and row["near_duplicate_in_reference"] == 0 for row in report.values())


def test_a_per_record_split_of_the_same_data_leaks_and_the_check_sees_it():
    train, held = split_dataset(_archive_like_records(), train_fraction=0.5, seed=0)
    report = cross_split_duplicates(train, {"held_out": held})["held_out"]
    assert report["near_duplicate_in_reference"] > report["pixel_copy_in_reference"] > 0


def test_split_without_group_key_is_unchanged():
    """The per-record path is the pre-review algorithm, record for record."""
    records = colour_records(9)

    def old_split(items, train_fraction, seed):
        by_class: dict = {}
        for record in items:
            by_class.setdefault(record["label"], []).append(dict(record))
        rng = np.random.default_rng(seed)
        train, held = [], []
        for label in sorted(by_class):
            part = by_class[label]
            order = rng.permutation(len(part))
            n_train = min(len(part) - 1, max(1, round(len(part) * train_fraction)))
            train.extend(part[int(i)] for i in order[:n_train])
            held.extend(part[int(i)] for i in order[n_train:])
        return train, held

    for seed in (0, 3):
        new = split_dataset(records, train_fraction=0.7, seed=seed)
        old = old_split(records, 0.7, seed)
        assert [[r["id"] for r in part] for part in new] == [[r["id"] for r in part] for part in old]


def test_group_split_refuses_missing_keys_and_classes_of_one_group():
    with pytest.raises(ValueError, match="run assign_duplicate_groups first"):
        split_dataset(colour_records(4), group_key="group")
    image = colour_image("frog", 1)
    records = [
        {"id": "a", "image": image, "label": "frog"},
        {"id": "b", "image": image.copy(), "label": "frog"},
        *colour_records(3)[3:],
    ]
    grouped, _summary = assign_duplicate_groups(records)
    with pytest.raises(ValueError, match="class 'frog' has 1 independent group"):
        split_dataset(grouped, group_key="group")


def test_near_duplicate_threshold_is_the_documented_value():
    assert NEAR_DUPLICATE_CORRELATION == 0.95 and data_module.NEAR_DUPLICATE_SIDE == 16


# ---------------------------------------------------------------- notebook harness


def _notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _source(cell: dict) -> str:
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


def _section_code(nb: dict, number: int) -> str:
    cells = nb["cells"]
    for index, cell in enumerate(cells):
        if cell["cell_type"] == "markdown" and f"## {number}. " in _source(cell):
            return next(_source(c) for c in cells[index + 1 :] if c["cell_type"] == "code")
    raise KeyError(number)


def _set(source: str, name: str, value: str) -> str:
    new, n = re.subn(rf"^{name} = .*?(  # @param.*)$", lambda m: f"{name} = {value}{m.group(1)}", source, flags=re.M)
    assert n == 1, name
    return new


def _sample_zip(path: Path, n: int = 8) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for record in _archive_like_records(n):
            png = io.BytesIO()
            record["image"].save(png, format="PNG")
            archive.writestr(f"CIFAR-10-subset/{record['id']}", png.getvalue())


@pytest.fixture
def notebook_run(tmp_path, monkeypatch):
    """Run Sections 4-13 of the committed notebook on a small stand-in model; returns (namespace, run)."""
    if "torch" not in sys.modules:
        pytest.skip("torch is not installed")
    transformers = pytest.importorskip("transformers")
    torchvision = pytest.importorskip("torchvision")

    from deit_classification_pipeline import pipeline as pipeline_module

    small = {"hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 2, "image_size": 32, "patch_size": 8}

    class TinyPipeline(pipeline_module.DeiTPipeline):
        """Stand-in for the pinned checkpoint: a fixed small base, re-headed under `seed`; `load_artifact` is inherited."""

        @classmethod
        def from_pretrained(cls, device=None, weights_dir=None, allow_download=False, class_names=None, seed=0):
            with torch.random.fork_rng():
                torch.manual_seed(0)  # the "snapshot": every tensor except the head is the same for every build
                model = pipeline_module.build_model(**small)
                reinitialised = ()
                if class_names is not None:
                    torch.manual_seed(seed)
                    reinitialised = pipeline_module._rehead(model, len(class_names))
            names = tuple(class_names) if class_names is not None else pipeline_module.imagenet_labels(model)
            pipe = cls.from_components(model, class_names=names, transform=pipeline_module.eval_transform(image_size=32), source="tiny")
            pipe.reinitialised = reinitialised
            return pipe

    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    _sample_zip(tmp_path / "data" / data_module.SAMPLE_DATASET_FILE)
    nb = _notebook()
    ns: dict = {"__name__": "__main__"}
    ns.update({k: v for k, v in vars(data_module).items() if not k.startswith("__")})
    ns.update({k: v for k, v in vars(pipeline_module).items() if not k.startswith("__")})
    ns.update(
        DeiTPipeline=TinyPipeline,
        WEIGHTS_DIR=tmp_path / "weights",
        NOTEBOOK_SOURCE={"repository_revision": "test"},
        Image=Image,
        torch=torch,
        torchvision=torchvision,
        transformers=transformers,
        platform=__import__("platform"),
        fetch_sample_archive=lambda directory, allow_download=False: {
            "dataset_id": "synthetic", "revision": "test", "license": "test", "bytes": 0, "sha256": "test", "fetched": False,
        },
    )
    ns["pipe"] = TinyPipeline.from_pretrained()

    def run(number: int, **fields: str) -> str:
        source = _section_code(nb, number)
        for name, value in fields.items():
            source = _set(source, name, value)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            exec(compile(source, f"section{number}", "exec"), ns)
        return buffer.getvalue()

    for number in range(4, 14):
        run(number)
    return ns, run


# ---------------------------------------------------------------- DEIT-M2 / DEIT-m2: default path


def test_default_run_splits_by_group_and_states_the_verdict(notebook_run):
    ns, _run = notebook_run
    assert ns["duplicate_summary"]["groups"] == 16 and ns["duplicate_summary"]["records"] == 32
    assert all(not row["pixel_copy_in_reference"] and not row["near_duplicate_in_reference"] for row in ns["leakage"].values())
    result = json.loads(Path("outputs/deit_classification_result.json").read_text(encoding="utf-8"))
    assert result["split"]["group_key"] == "group" and result["split"]["duplicates_across_splits"] == ns["leakage"]
    assert result["dataset"]["duplicate_groups"]["groups"] == 16
    assert result["comparison_verdict"] == ns["comparison_verdict"] and len(result["run_history"]) == 1
    assert result["reload_check"]["equivalent"] is True and "train_seconds" in result["finetune"]
    header = Path("outputs/deit_classification_predictions.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header == "split,id,group,truth,predicted,score"


def test_section_5_refuses_records_that_were_not_grouped(notebook_run):
    ns, run = notebook_run
    ns["records"] = _archive_like_records()  # ungrouped records: split_dataset refuses before anything leaks
    with pytest.raises(ValueError, match="run assign_duplicate_groups first"):
        run(5)


@pytest.mark.parametrize(
    ("zero_shot_accuracy", "expected"),
    [(1.0, "this run measures no fine-tuning gain"), (0.0, "is above zero-shot")],
)
def test_comparison_verdict_names_the_ceiling(notebook_run, zero_shot_accuracy, expected):
    ns, run = notebook_run
    ns["adapted"] = None
    ns["zero_shot"] = {"accuracy": zero_shot_accuracy, "balanced_accuracy": zero_shot_accuracy}
    real_evaluate = ns["adapter"].evaluate
    ns["adapter"].evaluate = lambda records, majority=None: {**real_evaluate(records, majority=majority), "accuracy": 1.0}
    out = run(9)
    assert expected in out and expected in ns["comparison_verdict"]


# ---------------------------------------------------------------- DEIT-M3: the fine-tune re-run and the export


def test_rerunning_only_the_finetune_cell_trains_from_the_reheaded_model(notebook_run):
    ns, run = notebook_run
    first = list(ns["run"]["epoch_losses"])
    out = run(8)  # same settings, re-run as a learner would
    assert "rebuilt_from_verified_snapshot" in out and ns["run"]["epoch_losses"] == first
    frozen = run(8, FREEZE_BACKBONE="True")  # the review's P4: edit the field, re-run this cell only
    assert "rebuilt_from_verified_snapshot" in frozen
    assert ns["run"]["freeze_backbone"] is True and ns["run"]["trainable_parameters"] < ns["run"]["total_parameters"]
    run(11)  # export straight after: the head-only adapter reproduces the model in memory (was a bare AssertionError)
    assert ns["reload_check"]["equivalent"] is True and ns["descriptor"]["frozen_prefixes"] == ["vit."]
    run(9)
    assert [row["freeze_backbone"] for row in ns["RUN_HISTORY"]] == [False, True]


def test_an_export_that_does_not_reproduce_the_model_names_the_next_step(notebook_run):
    ns, run = notebook_run
    run(8, FREEZE_BACKBONE="True")
    ns["adapter"].finetune(ns["train_records"], epochs=1, batch_size=8, learning_rate=1e-2, seed=0, freeze_backbone=False)
    ns["adapter"].frozen_prefixes = ("vit.",)  # the pre-review state: a frozen export of a changed backbone
    with pytest.raises(RuntimeError, match="the export does not reproduce the evaluated model.*Run after"):
        run(11)


# ---------------------------------------------------------------- DEIT-m1: BYOD dataset branch


def _class_folder(root: Path, per_class: int = 6, loose: int = 0) -> Path:
    for k, label in enumerate(("cat", "dog")):
        (root / label).mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            colour_image("frog" if k == 0 else "truck", 900 + 10 * k + i).save(root / label / f"{i}.png")
    for i in range(loose):
        colour_image("frog", 990 + i).save(root / f"loose_{i}.png")
    return root


def test_byod_dataset_writes_results_and_compares_the_reload(notebook_run, tmp_path):
    ns, run = notebook_run
    folder = _class_folder(tmp_path / "mine", loose=1)
    ns["EPOCHS"] = 1
    out = run(13, USE_BYOD_DATASET="True", BYOD_DATASET_PATH=repr(str(folder)))
    assert "1 image(s) at the top level were ignored" in out
    result = json.loads(Path("outputs/byod/byod_deit_classification_result.json").read_text(encoding="utf-8"))
    assert result["class_names"] == ["cat", "dog"] and result["reload_check"]["equivalent"] is True
    assert result["split"]["group_key"] == "group" and result["ignored_top_level_images"] == 1
    assert result["held_out"]["n"] >= 2 and "unseen" in result["split"]
    assert all(not row["pixel_copy_in_reference"] and not row["near_duplicate_in_reference"] for row in result["split"]["duplicates_across_splits"].values())
    assert Path("outputs/byod/byod_deit_classification_predictions.csv").is_file()


def test_byod_zip_with_a_train_val_layout_is_merged_and_says_so(notebook_run, tmp_path):
    ns, run = notebook_run
    archive = tmp_path / "split.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for part in ("train", "val"):
            for k, label in enumerate(("cat", "dog")):
                for i in range(3):
                    png = io.BytesIO()
                    colour_image("frog" if k == 0 else "truck", 700 + 50 * (part == "val") + 10 * k + i).save(png, format="PNG")
                    zf.writestr(f"{part}/{label}/{i}.png", png.getvalue())
    ns["EPOCHS"] = 1
    out = run(13, USE_BYOD_DATASET="True", BYOD_DATASET_PATH=repr(str(archive)))
    assert "look like a pre-made split" in out and "merged by class name and re-split" in out
    assert json.loads(Path("outputs/byod/byod_deit_classification_result.json").read_text(encoding="utf-8"))["manifest"]["n_records"] == 12


def test_byod_refusals_name_the_rule_and_the_field(notebook_run, tmp_path, monkeypatch):
    ns, run = notebook_run
    tar = tmp_path / "mine.tar"
    tar.write_bytes(b"not a zip")
    with pytest.raises(ValueError, match=r"is not a \.zip: the dataset must be a directory or a \.zip"):
        run(13, USE_BYOD_DATASET="True", BYOD_DATASET_PATH=repr(str(tar)))
    one_class = tmp_path / "one"
    (one_class / "cat").mkdir(parents=True)
    for i in range(3):
        colour_image("frog", 800 + i).save(one_class / "cat" / f"{i}.png")
        colour_image("truck", 850 + i).save(one_class / f"loose_{i}.png")
    with pytest.raises(ValueError, match=r"at least 2 class folders.*3 loose top-level image\(s\); move them into class folders"):
        run(13, USE_BYOD_DATASET="True", BYOD_DATASET_PATH=repr(str(one_class)))
    monkeypatch.setitem(sys.modules, "google.colab", None)  # no Colab: the import fails as on Kaggle or local Jupyter
    with pytest.raises(FileNotFoundError, match="BYOD_DATASET_PATH is empty and this runtime has no Colab upload dialog"):
        run(13, USE_BYOD_DATASET="True")
    with pytest.raises(FileNotFoundError, match="BYOD_IMAGE_PATH is empty"):
        run(13, USE_BYOD_IMAGE="True")


# ---------------------------------------------------------------- DEIT-M1 / DEIT-M4 / DEIT-m3: static checks


def test_only_the_uv_install_and_router_cells_run_in_the_kernel():
    nb = _notebook()
    code = [_source(c) for c in nb["cells"] if c["cell_type"] == "code"]
    kernel = [c for c in code if "# dimer: kernel cell" in c]
    assert len(kernel) == 2 and "--require-hashes" in kernel[0] and "--managed-python" in kernel[0]
    assert nb["metadata"]["dimer"]["notebook_spec"] == "2.2"


def test_review_prose_is_gone_and_the_guided_layer_is_present():
    markdown = "\n".join(_source(c) for c in _notebook()["cells"] if c["cell_type"] == "markdown")
    for stale in ("independent thumbnails", "Expect roughly chance", "Runtimes are not measured", "@@", "falls back toward the zero-shot baseline"):
        assert stale not in markdown, stale
    for needed in ("**Who this is for.**", "**How to use this notebook.**", "**Roadmap:**", "## Glossary", "## Troubleshooting", "## Conclusion (your notes)", "## 14. Your turn"):
        assert needed in markdown, needed
    assert markdown.count("**Predict before running:**") >= 7 and markdown.count("<details><summary>Check your reasoning</summary>") >= 9


# ---------------------------------------------------------------- FIX_PACKET addendum: google.colab stubs carry a spec


@pytest.mark.parametrize("real_google", [False, True])
def test_worker_colab_stubs_have_specs(monkeypatch, real_google):
    """Colab only: accelerate calls importlib.util.find_spec("google.colab"), which raised on the spec-less stub."""
    import ast
    import importlib.util

    nb = _notebook()
    router = [_source(c) for c in nb["cells"] if c["cell_type"] == "code"][1]
    worker = next(
        node.value.value
        for node in ast.parse(router).body
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "_WORKER_SOURCE"
    )
    start = worker.index('if os.environ.get("DIMER_KERNEL_IS_COLAB") == "1":')
    shim = worker[start : worker.index('_main = types.ModuleType("__main__")', start)]
    fake_google = types.ModuleType("google")
    fake_google.__path__ = []
    monkeypatch.setitem(sys.modules, "google", fake_google if real_google else None)
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    monkeypatch.delitem(sys.modules, "google.colab.files", raising=False)
    monkeypatch.setenv("DIMER_KERNEL_IS_COLAB", "1")
    try:
        exec(compile(shim, "worker-colab-shim", "exec"), {"os": __import__("os"), "sys": sys, "types": types, "_send": None, "_recv": None})
        for name in ("google.colab", "google.colab.files"):
            spec = importlib.util.find_spec(name)  # raised ValueError before the fix
            assert spec is not None and spec.name == name
        assert sys.modules["google.colab"].__path__ == [] and callable(sys.modules["google.colab.files"].upload)
        if not real_google:
            assert importlib.util.find_spec("google") is not None
    finally:
        for name in ("google", "google.colab", "google.colab.files"):
            sys.modules.pop(name, None)  # monkeypatch then restores whatever was there before
