"""The data module: pinned sample archive, archive and folder readers, validation and splitting. No torch."""

import hashlib
import zipfile

import pytest
from PIL import Image

from conftest import class_zip, colour_image, colour_records
from deit_classification_pipeline import data as data_module
from deit_classification_pipeline.data import (
    SAMPLE_CLASSES,
    SAMPLE_DATASET_BYTES,
    SAMPLE_DATASET_FILE,
    SAMPLE_DATASET_ID,
    SAMPLE_DATASET_REVISION,
    SAMPLE_DATASET_SHA256,
    blank_image,
    fetch_sample_archive,
    noise_image,
    read_class_archive,
    read_class_folder,
    split_dataset,
    validate_dataset,
    verify_archive,
)


def test_sample_identity_is_an_immutable_pin():
    assert SAMPLE_DATASET_ID == "Cleanlab/cifar-10-subset" and SAMPLE_DATASET_FILE == "CIFAR-10-subset.zip"
    assert len(SAMPLE_DATASET_REVISION) == 40 and len(SAMPLE_DATASET_SHA256) == 64
    assert SAMPLE_DATASET_BYTES == 986707 and SAMPLE_CLASSES == ("frog", "truck")


def test_read_class_archive_labels_by_parent_folder(tmp_path):
    path = tmp_path / "set.zip"
    path.write_bytes(
        class_zip(
            3, extra={"__MACOSX/CIFAR-10-subset/frog/._000.png": b"x", "CIFAR-10-subset/README.txt": b"hi"}
        )
    )
    records = read_class_archive(path)
    assert sorted({r["label"] for r in records}) == ["frog", "truck"] and len(records) == 6
    assert all(r["image"].mode == "RGB" and r["image"].size == (32, 32) for r in records)


def test_read_class_archive_filters_samples_and_is_deterministic(tmp_path):
    path = tmp_path / "set.zip"
    path.write_bytes(class_zip(6))
    first = read_class_archive(path, classes=["truck"], max_per_class=2, seed=3)
    again = read_class_archive(path, classes=["truck"], max_per_class=2, seed=3)
    assert [r["id"] for r in first] == [r["id"] for r in again] and {r["label"] for r in first} == {"truck"}
    assert len(first) == 2
    with pytest.raises(ValueError, match=r"classes \['cat'\] have no image"):
        read_class_archive(path, classes=["frog", "cat"])
    with pytest.raises(ValueError, match="max_per_class"):
        read_class_archive(path, max_per_class=0)


@pytest.mark.parametrize("name", ["../evil/frog/a.png", "/abs/frog/a.png", "C:/x/frog/a.png"])
def test_read_class_archive_refuses_escaping_members(tmp_path, name):
    path = tmp_path / "set.zip"
    path.write_bytes(class_zip(2, extra={name: b"x"}))
    with pytest.raises(ValueError, match="leaves the archive root"):
        read_class_archive(path)


def test_read_class_archive_bounds_members_size_and_decoding(tmp_path, monkeypatch):
    path = tmp_path / "set.zip"
    path.write_bytes(class_zip(2))
    monkeypatch.setattr(data_module, "MAX_ARCHIVE_MEMBERS", 3)
    with pytest.raises(ValueError, match="MAX_ARCHIVE_MEMBERS"):
        read_class_archive(path)
    monkeypatch.setattr(data_module, "MAX_ARCHIVE_MEMBERS", 100)
    monkeypatch.setattr(data_module, "MAX_ARCHIVE_BYTES", 10)
    with pytest.raises(ValueError, match="MAX_ARCHIVE_BYTES"):
        read_class_archive(path)
    monkeypatch.setattr(data_module, "MAX_ARCHIVE_BYTES", 1 << 30)
    broken = tmp_path / "broken.zip"
    broken.write_bytes(class_zip(2, extra={"CIFAR-10-subset/frog/bad.png": b"not a png"}))
    with pytest.raises(ValueError, match="not a decodable image"):
        read_class_archive(broken)
    empty = tmp_path / "empty.zip"
    with zipfile.ZipFile(empty, "w") as archive:
        archive.writestr("flat.png", b"x")
    with pytest.raises(ValueError, match="no image inside a class folder"):
        read_class_archive(empty)


def test_read_class_folder_reads_and_refuses_escaping_links(tmp_path):
    for label in ("frog", "truck"):
        (tmp_path / "set" / label).mkdir(parents=True)
        for i in range(2):
            colour_image(label, i).save(tmp_path / "set" / label / f"{i}.png")
    records = read_class_folder(tmp_path / "set")
    assert len(records) == 4 and records[0]["id"] == "frog/0.png"
    colour_image("frog", 9).save(tmp_path / "outside.png")
    (tmp_path / "set" / "frog" / "link.png").symlink_to(tmp_path / "outside.png")
    with pytest.raises(ValueError, match="resolves outside"):
        read_class_folder(tmp_path / "set")
    with pytest.raises(FileNotFoundError):
        read_class_folder(tmp_path / "missing")


def test_fetch_sample_archive_verifies_downloaded_and_existing_bytes(tmp_path, monkeypatch):
    payload = class_zip(2)
    monkeypatch.setattr(data_module, "SAMPLE_DATASET_BYTES", len(payload))
    monkeypatch.setattr(data_module, "SAMPLE_DATASET_SHA256", hashlib.sha256(payload).hexdigest())
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        fetch_sample_archive(tmp_path)
    calls = []

    def fake(root):
        calls.append(root)
        (root / SAMPLE_DATASET_FILE).write_bytes(payload)

    info = fetch_sample_archive(tmp_path, allow_download=True, downloader=fake)
    assert info["fetched"] and info["bytes"] == len(payload) and calls == [tmp_path]
    assert fetch_sample_archive(tmp_path, allow_download=True, downloader=fake)["fetched"] is False
    (tmp_path / SAMPLE_DATASET_FILE).write_bytes(bytes([payload[0] ^ 0xFF]) + payload[1:])
    with pytest.raises(ValueError, match="sha256"):
        fetch_sample_archive(tmp_path)


def test_verify_archive_checks_size_then_digest(tmp_path):
    path = tmp_path / "a.zip"
    path.write_bytes(b"abc")
    assert verify_archive(path, size=3, sha256=hashlib.sha256(b"abc").hexdigest())["bytes"] == 3
    with pytest.raises(ValueError, match="size"):
        verify_archive(path, size=4, sha256="0" * 64)
    with pytest.raises(ValueError, match="sha256"):
        verify_archive(path, size=3, sha256="0" * 64)


def test_validate_dataset_accepts_and_reports_findings():
    records = colour_records(4)
    manifest = validate_dataset(records, SAMPLE_CLASSES, epochs=2)
    assert manifest["verdict"] == "accepted" and manifest["images_per_class"] == {"frog": 4, "truck": 4}
    assert manifest["findings"] == [] and manifest["image_side_px"] == [32, 32]
    skewed = records + colour_records(10, seed=5)[:10]
    assert any("imbalance" in f for f in validate_dataset(skewed, SAMPLE_CLASSES)["findings"])
    duplicated = records + [dict(records[0], id="copy")]
    assert validate_dataset(duplicated, SAMPLE_CLASSES)["duplicate_groups"] == 1


@pytest.mark.parametrize(
    ("records", "message"),
    [
        ([], "at least one record"),
        ([{"image": "x", "label": "frog"}], "PIL.Image.Image"),
        ([{"image": Image.new("RGB", (4, 4)), "label": "frog"}], "outside"),
        ([{"image": blank_image(32, 32), "label": "cat"}], "unknown class"),
        ([{"image": blank_image(32, 32)}], "must be a mapping"),
        (
            [
                {"image": blank_image(32, 32), "label": "frog"},
                {"image": noise_image(1, 32, 32), "label": "truck"},
            ],
            "at least 2 images",
        ),
    ],
)
def test_validate_dataset_names_the_failed_rule(records, message):
    with pytest.raises((TypeError, ValueError), match=message):
        validate_dataset(records, SAMPLE_CLASSES, epochs=1)


def test_validate_dataset_rejects_bad_vocabulary_and_epochs():
    records = colour_records(2)
    for names, message in ((["frog"], "2.."), (["frog", "frog"], "duplicates"), (["frog", ""], "non-empty")):
        with pytest.raises(ValueError, match=message):
            validate_dataset(records, names)
    with pytest.raises(ValueError, match="epochs"):
        validate_dataset(records, SAMPLE_CLASSES, epochs=0)


def test_split_dataset_is_stratified_deterministic_and_disjoint():
    records = colour_records(10)
    train, held = split_dataset(records, train_fraction=0.7, seed=4)
    again, _ = split_dataset(records, train_fraction=0.7, seed=4)
    assert [r["id"] for r in train] == [r["id"] for r in again]
    assert len(train) == 14 and len(held) == 6 and not {r["id"] for r in train} & {r["id"] for r in held}
    assert {r["label"] for r in held} == {"frog", "truck"}
    tiny_train, tiny_held = split_dataset(colour_records(2), train_fraction=0.99)
    assert len(tiny_held) == 2 and len(tiny_train) == 2
    with pytest.raises(ValueError, match="train_fraction"):
        split_dataset(records, train_fraction=1.0)
    with pytest.raises(ValueError, match="at least 2"):
        split_dataset(records[:1], train_fraction=0.5)


def test_probe_images_are_deterministic():
    assert blank_image().getextrema() == ((255, 255), (255, 255), (255, 255))
    assert noise_image(3).tobytes() == noise_image(3).tobytes() != noise_image(4).tobytes()
