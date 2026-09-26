# DeiT-Small image classification pipeline

DIMER pipeline for **DeiT-Small** (`facebook/deit-small-patch16-224`), a data-efficient Vision Transformer trained on ImageNet-1k. The pipeline loads the checkpoint only from a digest-verified local snapshot, returns top-k softmax scores over the ImageNet-1k classes, and adds a bounded fine-tuning workflow that replaces the head for a new set of classes, compares it with majority-class and zero-shot baselines, and exports a SafeTensors adapter.

> **The upstream snapshot is pinned** to Hub commit `164deee347853469b97442b3817f22eece80c7e3` (pinned 2026-09-25). The manifest records every file's byte size and SHA-256, and each LFS digest matched the Hub's record. Default-path execution recorded on 2026-09-26 (Kaggle T4); REL12 BYOD exercise pending before promotion (see [Release status](#release-status)).

## Upstream alignment

- Model: `facebook/deit-small-patch16-224`
- Revision: `164deee347853469b97442b3817f22eece80c7e3`
- Upstream weight license: Apache-2.0
- Upstream task: single-label classification over the 1000 ImageNet-1k classes
- Repository adaptation: bounded gradient fine-tuning of a new head, with the whole network trained by default or the transformer frozen
- Executed artifact: `pytorch_model.bin`, read with `torch.load(..., weights_only=True)` (the Hub repository has no SafeTensors file); `tf_model.h5` is recorded for provenance, never staged
- Preprocessing: DeiT's own evaluation transform (bicubic resize to 256 px, centre crop 224, ImageNet mean/std), not the snapshot's `preprocessor_config.json` (224 px resize, mean/std 0.5)
- Tutorial sample: `Cleanlab/cifar-10-subset` at commit `bb5a7aabf1d14d2d1e3e49d0d8f917bda3622f75` (MIT), `frog` and `truck` folders

## Quick start

```python
from PIL import Image
from deit_classification_pipeline import (
    SAMPLE_CLASSES, DeiTPipeline, fetch_sample_archive, majority_class, read_class_archive, split_dataset,
)

pipe = DeiTPipeline.from_pretrained(allow_download=True)   # stages + verifies weights/deit-small-patch16-224
print(pipe.predict(Image.open("photo.jpg"), top_k=5)["predictions"][0]["top_k"])

info = fetch_sample_archive("data", allow_download=True)             # pinned archive, SHA-256 checked
records = read_class_archive(info["path"], classes=SAMPLE_CLASSES)
train, held_out = split_dataset(records, train_fraction=0.7)
adapter = DeiTPipeline.from_pretrained(class_names=SAMPLE_CLASSES)
adapter.finetune(train)                                               # 5 epochs, whole network
print(adapter.evaluate(held_out, majority=majority_class(train)))     # accuracy, balanced accuracy, baseline
adapter.save_artifact("outputs/deit_adapter.safetensors")
```

Install into a Python 3.12 environment that already holds the pinned dependencies with `pip install -e . --no-deps`, and run `pytest` for the offline test suite. No weights or dataset are needed: `tests/test_small_model.py` builds a two-layer `ViTForImageClassification` with random weights on 32 px inputs, and the data tests build their own small archives.

## Pinning the snapshot

The snapshot is pinned (see [Upstream alignment](#upstream-alignment)). To move to a newer upstream commit, from the repository root with network access to huggingface.co:

1. Run `python tools/pin_snapshot.py` (or `--revision <commit>`). It resolves `main` to a commit, downloads the four manifest files at that commit into `weights/deit-small-patch16-224/`, checks each LFS file against the Hub's SHA-256, records the LFS SHA-256 of `tf_model.h5` without downloading it, and writes the commit and digests into the manifest and `MODEL_REVISION`.
2. Commit, then run `python tools/build_notebook.py` and commit the regenerated notebook.
3. Update the commit and digests cited in `README.md`, `MODEL_CARD.md`, `STATUS.md`, `docs/WEIGHTS.md`, `tutorials/README.md` and `docs/release-verification.md`.
4. Run `python tools/validate_release_assets.py` and `pytest`. A new pin invalidates any recorded execution, so the status returns to Candidate until the new commit is run.

## Weights layout

```
weights/deit-small-patch16-224/
  dimer-base-manifest.json   # modelId, revision, per-file bytes + SHA-256 (4 files + 1 reference file)
  preprocessor_config.json   # committed; 224 px resize, mean/std 0.5 (recorded, not used)
  config.json                # staged; ViTForImageClassification, 384 wide, 12 layers, 6 heads, patch 16
  pytorch_model.bin          # git-ignored, 88,283,631 bytes; the executed artifact
  README.md                  # staged with the weights
```

## Input ceilings and decision rule

`MIN_IMAGE_SIDE = 8`, `MAX_IMAGE_SIDE = 4096`, `MAX_BATCH = 64`, `NUM_IMAGENET_CLASSES = 1000`, `DEFAULT_TOP_K = 5`. The reported label is the softmax argmax; there is no threshold and no reject option. Adaptation datasets hold up to 5,000 records over 2 to 1,000 classes, with at least 2 images per class. See `MODEL_CARD.md` for what the score means and who owns an abstention rule.

## Tutorials

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/deit-classification-pipeline/blob/main/tutorials/deit_classification_colab.ipynb)

`tutorials/deit_classification_colab.ipynb` is declared `E2E` / `GUIDED` under DIMER Notebook Specification 2.1 and is **standalone** (§4): `tools/build_notebook.py` generates it, and it carries the package modules, the model identity, the manifest and the runtime pins, so it runs without this repository. Its default `Run all` path downloads and verifies the pinned CIFAR-10 subset, measures the majority-class and zero-shot ImageNet baselines, probes a blank and a noise image, fine-tunes, evaluates the held-out split, classifies an unseen split, and exports and reloads the adapter. BYOD image and dataset branches are off by default. See `tutorials/README.md` and `docs/release-verification.md`.

## Release status

**Candidate.** The snapshot is pinned (`164deee`). Default-path execution recorded on 2026-09-26 (Kaggle T4): the exact notebook blob `1a0317f0db42` (commit `4386e6d`) ran top-to-bottom with both BYOD branches off. On one seeded split of 60 held-out CIFAR-10 thumbnails, accuracy was 1.000 for the fine-tuned head and 0.983 for the zero-shot ImageNet mapping (majority baseline 0.500, untrained head 0.483), a one-image difference; 46 of the 60 held-out images have a darkened/original counterpart in the training split, so the score is not evidence of generalisation; one runtime. REL12 BYOD exercise pending before promotion: release step 7 has not been run. Static checks, unit tests and the small-model test do not constitute notebook execution evidence; `docs/release-verification.md` defines the release gate.

## Documentation

- `MODEL_CARD.md`: MODEL_CARD_SPEC 1.2 card, provenance, input/output contract.
- `docs/WEIGHTS.md`: weight and sample-data provenance, pinning and hosting notes.
- `STATUS.md`: release status.

## Licensing

This repository's code is Apache-2.0 (see `LICENSE`). The upstream weights are Apache-2.0, and the tutorial's sample archive is MIT; see `docs/WEIGHTS.md` and `MODEL_CARD.md`.

## AI Assistance Disclosure

This repository’s code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
