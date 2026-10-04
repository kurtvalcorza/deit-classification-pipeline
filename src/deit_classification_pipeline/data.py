"""Labelled image-classification data: the pinned tutorial sample, class-folder readers, validation, splits.

Nothing here needs torch — Pillow and numpy only, plus ``huggingface_hub`` when the sample archive is
downloaded. Two sources produce the same record shape, ``{"id": str, "image": PIL.Image, "label": str}``:

* ``fetch_sample_archive`` + ``read_class_archive``: the pinned ``Cleanlab/cifar-10-subset`` archive
  (MIT licence), checked against a recorded byte size and SHA-256 before any member is opened;
* ``read_class_folder``: a caller's own directory of ``<class>/<image>`` files (BYOD).

Both refuse absolute member paths, ``..`` segments and oversized archives before decoding anything, and
``validate_dataset`` checks the result before any model runs. ``assign_duplicate_groups`` joins
pixel-identical and near-duplicate images (e.g. a darkened copy of a photograph) into groups,
``split_dataset(..., group_key="group")`` keeps every group on one side of a split, and
``cross_split_duplicates`` checks from the pixels that no held-out image has a copy in the training split.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image

# The tutorial sample: a CIFAR-10 subset published as one zip archive of class folders. The revision,
# size and digest are those of the archive at that Hub commit; `fetch_sample_archive` refuses any other bytes.
SAMPLE_DATASET_ID = "Cleanlab/cifar-10-subset"
SAMPLE_DATASET_REVISION = "bb5a7aabf1d14d2d1e3e49d0d8f917bda3622f75"
SAMPLE_DATASET_FILE = "CIFAR-10-subset.zip"
SAMPLE_DATASET_BYTES = 986707
SAMPLE_DATASET_SHA256 = "66f90a4f87d865e8eb653b62f10e754684075a32314177de76832349d4b1fb19"
SAMPLE_DATASET_LICENSE = "mit"
SAMPLE_CLASSES: tuple[str, ...] = ("frog", "truck")

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
MAX_ARCHIVE_MEMBERS = 20000
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024  # total uncompressed size, checked from the zip directory
MAX_RECORDS = 5000
MAX_CLASSES = 1000
MIN_PER_CLASS = 2
MIN_IMAGE_SIDE = 8
MAX_IMAGE_SIDE = 4096


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(
    path: str | Path, *, size: int = SAMPLE_DATASET_BYTES, sha256: str = SAMPLE_DATASET_SHA256
) -> dict[str, Any]:
    """Check an archive's byte size and SHA-256; raise naming the first mismatch."""
    archive = Path(path)
    if not archive.is_file():
        raise FileNotFoundError(f"archive not found: {archive}")
    actual_size = archive.stat().st_size
    if actual_size != size:
        raise ValueError(f"{archive.name}: size {actual_size} != recorded {size}")
    digest = _sha256_file(archive)
    if digest != sha256:
        raise ValueError(f"{archive.name}: sha256 {digest} != recorded {sha256}")
    return {"path": str(archive), "bytes": actual_size, "sha256": digest}


def _hub_dataset_download(root: Path) -> None:
    from huggingface_hub import hf_hub_download

    hf_hub_download(
        SAMPLE_DATASET_ID,
        SAMPLE_DATASET_FILE,
        repo_type="dataset",
        revision=SAMPLE_DATASET_REVISION,
        local_dir=str(root),
    )


def fetch_sample_archive(
    directory: str | Path,
    *,
    allow_download: bool = False,
    downloader: Callable[[Path], None] | None = None,
) -> dict[str, Any]:
    """Return the verified tutorial archive under ``directory``, downloading it at the pinned revision if
    it is absent and ``allow_download`` is set. The archive is verified whether or not it was downloaded."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    path = root / SAMPLE_DATASET_FILE
    fetched = False
    if not path.is_file():
        if not allow_download:
            raise FileNotFoundError(
                f"{path} is absent; pass allow_download=True to fetch {SAMPLE_DATASET_ID}"
            )
        (downloader or _hub_dataset_download)(root)
        fetched = True
    info = verify_archive(path, size=SAMPLE_DATASET_BYTES, sha256=SAMPLE_DATASET_SHA256)
    return {
        **info,
        "fetched": fetched,
        "dataset_id": SAMPLE_DATASET_ID,
        "revision": SAMPLE_DATASET_REVISION,
        "license": SAMPLE_DATASET_LICENSE,
    }


def _safe_member(name: str) -> PurePosixPath:
    member = PurePosixPath(name)
    if (
        name.startswith(("/", "\\"))
        or member.is_absolute()
        or ".." in member.parts
        or ":" in (member.parts[0] if member.parts else "")
    ):
        raise ValueError(f"archive member {name!r} is absolute or leaves the archive root")
    return member


def _open_image(data: bytes, where: str) -> Image.Image:
    try:
        with Image.open(io.BytesIO(data)) as handle:
            width, height = handle.size
            if max(width, height) > MAX_IMAGE_SIDE:
                raise ValueError(
                    f"{where}: image side {max(width, height)} px > MAX_IMAGE_SIDE {MAX_IMAGE_SIDE}"
                )
            return handle.convert("RGB")
    except (OSError, SyntaxError) as exc:
        raise ValueError(f"{where}: not a decodable image ({exc})") from exc


def _select(
    by_class: Mapping[str, list[Any]], classes: Sequence[str] | None, max_per_class: int | None, seed: int
) -> dict[str, list[Any]]:
    if classes is not None:
        missing = [name for name in classes if name not in by_class]
        if missing:
            raise ValueError(f"classes {missing} have no image; found {sorted(by_class)}")
        by_class = {name: by_class[name] for name in classes}
    if max_per_class is not None:
        if isinstance(max_per_class, bool) or not isinstance(max_per_class, int) or max_per_class < 1:
            raise ValueError(f"max_per_class must be a positive int, got {max_per_class!r}")
        rng = np.random.default_rng(seed)
        by_class = {
            name: [items[int(i)] for i in sorted(rng.permutation(len(items))[:max_per_class])]
            for name, items in sorted(by_class.items())
        }
    return dict(by_class)


def read_class_archive(
    path: str | Path,
    *,
    classes: Sequence[str] | None = None,
    max_per_class: int | None = None,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Read ``{"id", "image", "label"}`` records from a zip of class folders (``.../<class>/<image>``).

    The label is the image's parent folder name. Member names are checked (no absolute paths, no ``..``)
    and the member count and total uncompressed size are bounded before any member is decompressed.
    ``classes`` keeps only those folders and fails if one is absent; ``max_per_class`` keeps a seeded
    sample of that many images per class.
    """
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if len(infos) > MAX_ARCHIVE_MEMBERS:
            raise ValueError(f"archive has {len(infos)} members > MAX_ARCHIVE_MEMBERS {MAX_ARCHIVE_MEMBERS}")
        total = sum(info.file_size for info in infos)
        if total > MAX_ARCHIVE_BYTES:
            raise ValueError(f"archive expands to {total} bytes > MAX_ARCHIVE_BYTES {MAX_ARCHIVE_BYTES}")
        by_class: dict[str, list[str]] = {}
        for info in infos:
            member = _safe_member(info.filename)
            if info.is_dir() or "__MACOSX" in member.parts or member.name.startswith("."):
                continue
            if member.suffix.lower() not in IMAGE_SUFFIXES or len(member.parts) < 2:
                continue
            by_class.setdefault(member.parts[-2], []).append(info.filename)
        if not by_class:
            raise ValueError("archive holds no image inside a class folder")
        selected = _select({k: sorted(v) for k, v in by_class.items()}, classes, max_per_class, seed)
        return [
            {"id": name, "image": _open_image(archive.read(name), name), "label": label}
            for label, names in selected.items()
            for name in names
        ]


def read_class_folder(
    directory: str | Path,
    *,
    classes: Sequence[str] | None = None,
    max_per_class: int | None = None,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Read records from ``<directory>/<class>/<image>``; links that leave ``directory`` are refused."""
    root = Path(directory).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"{root} is not a directory")
    by_class: dict[str, list[Path]] = {}
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        for file in sorted(class_dir.iterdir()):
            if file.suffix.lower() not in IMAGE_SUFFIXES or file.name.startswith("."):
                continue
            resolved = file.resolve()
            if root not in resolved.parents:
                raise ValueError(f"{file} resolves outside {root}")
            by_class.setdefault(class_dir.name, []).append(resolved)
    if not by_class:
        raise ValueError(f"{root} holds no image inside a class folder")
    selected = _select(by_class, classes, max_per_class, seed)
    return [
        {
            "id": str(file.relative_to(root)),
            "image": _open_image(file.read_bytes(), str(file.relative_to(root))),
            "label": label,
        }
        for label, files in selected.items()
        for file in files
    ]


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


def _pixel_digest(image: Image.Image) -> str:
    return _sha256_bytes(image.convert("RGB").tobytes() + repr(image.size).encode())


def validate_dataset(
    records: Sequence[Mapping[str, Any]], class_names: Sequence[str], *, epochs: int = 5
) -> dict[str, Any]:
    """Validation stage: raise on the first broken record, else return the dataset manifest.

    A class with fewer than ``MIN_PER_CLASS`` images is an error, because it cannot be split. Findings,
    which do not stop the run, report class imbalance above 2:1 and exact pixel duplicates; a duplicate
    inflates the held-out score when one copy lands on each side of a split.
    """
    names = _check_class_names(class_names)
    if not records:
        raise ValueError("dataset must hold at least one record")
    if len(records) > MAX_RECORDS:
        raise ValueError(f"dataset holds {len(records)} records > MAX_RECORDS {MAX_RECORDS}")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 100:
        raise ValueError(f"epochs must be an int in 1..100, got {epochs!r}")
    counts = dict.fromkeys(names, 0)
    sides: list[int] = []
    digests: dict[str, list[str]] = {}
    for idx, record in enumerate(records):
        if not isinstance(record, Mapping) or not {"image", "label"} <= set(record):
            raise ValueError(f"record {idx} must be a mapping with 'image' and 'label'")
        image = record["image"]
        if not isinstance(image, Image.Image):
            raise TypeError(f"record {idx}: image must be a PIL.Image.Image, got {type(image).__name__}")
        if min(image.size) < MIN_IMAGE_SIDE or max(image.size) > MAX_IMAGE_SIDE:
            raise ValueError(
                f"record {idx}: image size {image.size} outside {MIN_IMAGE_SIDE}..{MAX_IMAGE_SIDE} px"
            )
        if record["label"] not in counts:
            raise ValueError(
                f"record {idx} has unknown class {record['label']!r}; expected one of {list(names)}"
            )
        counts[record["label"]] += 1
        sides.extend(image.size)
        digests.setdefault(_pixel_digest(image), []).append(str(record.get("id", idx)))
    short = {name: n for name, n in counts.items() if n < MIN_PER_CLASS}
    if short:
        raise ValueError(
            f"every class needs at least {MIN_PER_CLASS} images to split; short classes: {short}"
        )
    findings = []
    if max(counts.values()) > 2 * min(counts.values()):
        findings.append(f"class imbalance above 2:1: {counts}")
    duplicates = [ids for ids in digests.values() if len(ids) > 1]
    if duplicates:
        findings.append(f"{len(duplicates)} group(s) of pixel-identical images, e.g. {duplicates[0][:3]}")
    return {
        "n_records": len(records),
        "class_names": list(names),
        "images_per_class": counts,
        "image_side_px": [min(sides), max(sides)],
        "duplicate_groups": len(duplicates),
        "epochs": epochs,
        "findings": findings,
        "verdict": "accepted",
    }


# Near-duplicate test (review 2026-10-02, CN2-M2): two images are one group when their 16×16 grayscale
# thumbnails, mean-centred and scaled to unit length, have a correlation of at least this value. The measure
# ignores global brightness and contrast, so a darkened copy of a photograph stays with its original. On the
# pinned CIFAR-10 sample every original/darkened pair correlates at >= 0.987 and no two different photographs
# above 0.872, so 0.95 separates them with a margin on both sides.
NEAR_DUPLICATE_SIDE = 16
NEAR_DUPLICATE_CORRELATION = 0.95
GROUP_KEY = "group"


def _near_duplicate_vectors(records: Sequence[Mapping[str, Any]]) -> np.ndarray:
    rows = []
    for record in records:
        side = (NEAR_DUPLICATE_SIDE, NEAR_DUPLICATE_SIDE)
        thumb = record["image"].convert("L").resize(side, Image.BILINEAR)
        vector = np.asarray(thumb, dtype=np.float64).ravel()
        vector = vector - vector.mean()
        norm = float(np.linalg.norm(vector))
        # A flat image has no shape to correlate; it can only match by pixel digest.
        rows.append(vector / norm if norm > 1e-9 else np.zeros_like(vector))
    return np.asarray(rows, dtype=np.float32).reshape(len(rows), NEAR_DUPLICATE_SIDE * NEAR_DUPLICATE_SIDE)


def _near_duplicate_pairs(
    left: np.ndarray, right: np.ndarray, threshold: float, *, same: bool
) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for start in range(0, len(left), 512):  # blocks bound memory at MAX_RECORDS
        block = left[start : start + 512] @ right.T
        for i, j in zip(*np.nonzero(block >= threshold), strict=True):
            a = start + int(i)
            if not same or a < int(j):
                pairs.append((a, int(j)))
    return pairs


def assign_duplicate_groups(
    records: Sequence[Mapping[str, Any]], *, threshold: float = NEAR_DUPLICATE_CORRELATION
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return copies of ``records`` with a ``"group"`` key, and a summary of the grouping.

    Records that are pixel-identical (same decoded RGB pixels and size) or near duplicates (see
    ``NEAR_DUPLICATE_CORRELATION``) are joined, transitively, into one group; every other record is its own
    group. The group key is the smallest record id in the group. Pass the result to
    ``split_dataset(..., group_key="group")`` so that no group straddles two splits.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")
    items = [dict(record) for record in records]
    ids = [str(item.get("id", index)) for index, item in enumerate(items)]
    parent = list(range(len(items)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        root_i, root_j = find(i), find(j)
        if root_i != root_j:
            parent[max(root_i, root_j)] = min(root_i, root_j)

    first_by_digest: dict[str, int] = {}
    exact_links = 0
    for index, item in enumerate(items):
        digest = _pixel_digest(item["image"])
        if digest in first_by_digest:
            union(first_by_digest[digest], index)
            exact_links += 1
        else:
            first_by_digest[digest] = index
    vectors = _near_duplicate_vectors(items)
    near_links = 0
    for i, j in _near_duplicate_pairs(vectors, vectors, threshold, same=True):
        if find(i) != find(j):
            near_links += 1
        union(i, j)
    members: dict[int, list[int]] = {}
    for index in range(len(items)):
        members.setdefault(find(index), []).append(index)
    for indices in members.values():
        key = min(ids[i] for i in indices)
        for i in indices:
            items[i][GROUP_KEY] = key
    multi = [indices for indices in members.values() if len(indices) > 1]
    mixed = [
        sorted(ids[i] for i in indices)[:3]
        for indices in multi
        if len({items[i]["label"] for i in indices}) > 1
    ]
    summary = {
        "records": len(items),
        "groups": len(members),
        "groups_with_duplicates": len(multi),
        "records_in_duplicate_groups": sum(len(indices) for indices in multi),
        "exact_copies": exact_links,
        "near_duplicate_links": near_links,
        "mixed_label_groups": len(mixed),
        "mixed_label_examples": mixed[:3],
        "method": (
            f"pixel digest, or {NEAR_DUPLICATE_SIDE}x{NEAR_DUPLICATE_SIDE} grayscale "
            f"correlation >= {threshold}"
        ),
    }
    return items, summary


def cross_split_duplicates(
    reference: Sequence[Mapping[str, Any]],
    others: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    threshold: float = NEAR_DUPLICATE_CORRELATION,
) -> dict[str, dict[str, int]]:
    """For each split in ``others``, count its records that have a pixel-identical copy, or a near duplicate,
    in ``reference`` (normally the training split). Computed from the pixels, independently of any group key;
    both counts must be 0 for a held-out score that is not partly a score on training images."""
    digests = {_pixel_digest(record["image"]) for record in reference}
    reference_vectors = _near_duplicate_vectors(reference)
    report: dict[str, dict[str, int]] = {}
    for name, part in others.items():
        exact = sum(_pixel_digest(record["image"]) in digests for record in part)
        pairs = _near_duplicate_pairs(_near_duplicate_vectors(part), reference_vectors, threshold, same=False)
        near = len({i for i, _j in pairs})
        report[name] = {
            "records": len(part),
            "pixel_copy_in_reference": exact,
            "near_duplicate_in_reference": near,
        }
    return report


def split_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    train_fraction: float = 0.7,
    seed: int = 0,
    group_key: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Stratified, seeded split: each class is shuffled and cut at ``train_fraction``, keeping at least one
    image of every class on each side. A random split assumes independent records; images that share a
    source photograph, scene or session must be split by that group instead, or the held-out score leaks.

    With ``group_key`` (e.g. ``"group"`` from ``assign_duplicate_groups``) whole groups are shuffled and
    assigned, so every record of a group lands on the same side; a group belongs to the class most of its
    records carry, and the cut is the number of groups whose record count comes closest to
    ``train_fraction``. Without it, the split is per record, exactly as before."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be between 0 and 1, got {train_fraction}")
    if group_key is not None:
        return _split_by_group(records, train_fraction, seed, group_key)
    by_class: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_class.setdefault(record["label"], []).append(dict(record))
    rng = np.random.default_rng(seed)
    train: list[dict[str, Any]] = []
    held_out: list[dict[str, Any]] = []
    for label in sorted(by_class):
        items = by_class[label]
        if len(items) < 2:
            raise ValueError(f"class {label!r} has {len(items)} record(s); at least 2 are needed to split")
        order = rng.permutation(len(items))
        n_train = min(len(items) - 1, max(1, round(len(items) * train_fraction)))
        train.extend(items[int(i)] for i in order[:n_train])
        held_out.extend(items[int(i)] for i in order[n_train:])
    return train, held_out


def _split_by_group(
    records: Sequence[Mapping[str, Any]], train_fraction: float, seed: int, group_key: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for index, record in enumerate(records):
        if group_key not in record:
            raise ValueError(
                f"record {index} has no {group_key!r} key; "
                "run assign_duplicate_groups first or pass group_key=None"
            )
        groups.setdefault(str(record[group_key]), []).append(dict(record))
    by_class: dict[str, list[str]] = {}
    for key in sorted(groups):
        labels = [item["label"] for item in groups[key]]
        by_class.setdefault(max(sorted(set(labels)), key=labels.count), []).append(key)
    rng = np.random.default_rng(seed)
    train: list[dict[str, Any]] = []
    held_out: list[dict[str, Any]] = []
    for label in sorted(by_class):
        keys = by_class[label]
        if len(keys) < 2:
            raise ValueError(
                f"class {label!r} has {len(keys)} independent group(s) after duplicate grouping; "
                "at least 2 are needed to split (add images that are not copies of each other)"
            )
        order = [keys[int(i)] for i in rng.permutation(len(keys))]
        sizes = np.cumsum([len(groups[key]) for key in order])
        target = sizes[-1] * train_fraction
        n_train = min(len(keys) - 1, max(1, int(np.argmin(np.abs(sizes[:-1] - target))) + 1))
        for position, key in enumerate(order):
            (train if position < n_train else held_out).extend(groups[key])
    return train, held_out


def blank_image(width: int = 224, height: int = 224) -> Image.Image:
    """A featureless white image: a closed-set classifier still assigns it one of its classes."""
    return Image.new("RGB", (width, height), (255, 255, 255))


def noise_image(seed: int = 0, width: int = 224, height: int = 224) -> Image.Image:
    """Uniform RGB noise: structure-free input, for the same reason as ``blank_image``."""
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95 % with the default ``z``), e.g. an accuracy
    of ``successes`` correct answers out of ``n`` held-out images. It stays inside [0, 1] and is not
    degenerate at 0 or n correct, which matters on the few dozen images a tutorial split holds."""
    if n < 1 or not 0 <= successes <= n:
        raise ValueError(f"need 0 <= successes <= n and n >= 1, got {successes}/{n}")
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denominator
    low = 0.0 if successes == 0 else max(0.0, centre - half)
    high = 1.0 if successes == n else min(1.0, centre + half)
    return low, high
