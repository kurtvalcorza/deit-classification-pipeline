"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.2 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded package, and the
model pin/stage/verify cells are produced by the generator from repository sources so they cannot
drift from the package.

This is an `E2E` template, so it must state `run_all` itself, and its default path really adapts:
NOTEBOOK_SPEC 2.2 RUN7/FT2 make a bounded fine-tune mandatory rather than optional for this profile.
Every value a reader can change is a `# @param` form field, and each file-reading BYOD branch has a
location field that bypasses the upload dialog when set (EXE1, EXE2).

Review 2026-10-02 (DEIT-M1..M4, DEIT-m1..m3): the notebook runs in the fleet's uv isolated environment
(generator /2.1); the split keeps pixel-identical and near-duplicate images (the archive's darkened copies) on
one side (`assign_duplicate_groups` + `split_dataset(..., group_key="group")`, checked from the pixels by
`cross_split_duplicates`); the comparison says when it is at ceiling instead of claiming a fine-tune gain; the
fine-tune rebuilds the re-headed model before every run; the guided layer (audience, task contract, how to use,
roadmap, predictions, worked answers, a change-one-thing activity, troubleshooting, glossary, conclusion) is
present; and the BYOD dataset branch writes a result JSON with a reload check.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "deit_classification_pipeline",
    "repo_name": "deit-classification-pipeline",
    "stem": "deit_classification",
    "notebook_name": "deit_classification_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    # GDL11 (NOTEBOOK_SPEC 2.2 §3.5): Sections 1-3 labelled Infrastructure and the carried module source collapsed.
    "infrastructure_labels": True,
    "isolated_runtime": True,
    # The fleet's uv isolated-environment mechanism (bart-mnli-zero-shot-classification-pipeline ee128d2, generator /2.1):
    # managed CPython, a size- and SHA-256-verified uv wheel, and a lock compiled from the pyproject pins with
    # `uv pip compile pyproject.toml --python-version 3.12 --python-platform x86_64-manylinux_2_28 --generate-hashes
    # --only-binary :all: -o tutorials/requirements-colab.lock.txt`, transitive versions constrained to the
    # depth-anything-depth-estimation-pipeline lock (same direct pins plus torchaudio), already run on a Colab T4.
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "lock": "tutorials/requirements-colab.lock.txt",
    "pipeline_class": "DeiTPipeline",
    "weights_key": "deit-small-patch16-224",
    "modules": [
        "data.py",
        "pipeline.py",
    ],
    "entry_module": "pipeline.py",
    "runtime_imports": ["torch", "torchvision", "transformers", "numpy", "PIL"],
    "title": "DeiT-Small (patch 16, 224 px, ImageNet-1k) — DIMER image classification and bounded fine-tuning (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/deit-classification-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/deit-classification-pipeline/blob/main/tutorials/deit_classification_colab.ipynb",
        ),
        (
            "Hugging Face",
            "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-facebook%2Fdeit--small--patch16--224-ffcc4d?style=flat",
            "https://huggingface.co/facebook/deit-small-patch16-224",
        ),
        (
            "Upstream",
            "https://img.shields.io/badge/Upstream-facebookresearch%2Fdeit-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/facebookresearch/deit",
        ),
        (
            "arXiv",
            "https://img.shields.io/badge/arXiv-2012.12877-b31b1b.svg",
            "https://arxiv.org/abs/2012.12877",
        ),
        (
            "License",
            "https://img.shields.io/badge/License-Apache--2.0-green.svg",
            "https://github.com/kurtvalcorza/deit-classification-pipeline/blob/main/LICENSE",
        ),
    ],
    "capability": "ImageNet-1k image classification with DeiT-Small, a data-efficient Vision Transformer, and a bounded fine-tune that replaces the 1000-class head with one for your own classes, compares it on a held-out split with the majority-class and zero-shot ImageNet baselines, and exports a reloadable SafeTensors adapter",
    "intro": (
        "DeiT-Small is a Vision Transformer (ViT) of 22M parameters trained on ImageNet-1k alone with the data-efficient recipe of Touvron et al. "
        "The image is cut into 16×16-pixel patches; each patch is linearly embedded, a `[CLS]` token and position embeddings are added, and "
        "12 transformer layers of width 384 with 6 attention heads process the 197-token sequence. A linear head on the final `[CLS]` token "
        "outputs a softmax over the 1000 ImageNet classes. The `facebook/deit-small-patch16-224` checkpoint is the Hugging Face conversion of "
        "that model; this notebook loads it as `ViTForImageClassification`.\n\n"
        "**Preprocessing note.** The checkpoint's `preprocessor_config.json` specifies a plain 224 px resize with mean and standard deviation "
        "0.5, but the DeiT training code and the upstream model card describe a bicubic resize to 256 px, a 224 px centre crop and the ImageNet "
        "mean and standard deviation. The pipeline applies DeiT's own evaluation transform.\n\n"
        "**The default path really adapts the model:** it downloads a pinned 400-image CIFAR-10 subset (`frog` and `truck`), groups the "
        "archive's duplicate and darkened copies so that no photograph lands on both sides of a split, measures two baselines on a held-out "
        "split — always answering the training majority, and mapping the ImageNet head's frog and truck classes to the two labels without any "
        "training — then replaces the head with a two-class layer, fine-tunes, scores the result on the same held-out split, classifies a "
        "further unseen split, exports the changed tensors as a SafeTensors adapter, and reloads that adapter onto a fresh copy of the verified "
        "base model to check that it reproduces the same predictions. Every number you see is measured in this notebook runtime. "
        "One result is worth knowing in advance: **frog versus truck is easy for this model.** The zero-shot ImageNet mapping already "
        "classifies 59 of the 60 held-out images correctly in the recorded run, so the fine-tune can gain at most one image, a difference "
        "the 95% intervals cannot separate; Section 9 says so in the reading it prints from the counts, and the notebook does not claim a "
        "fine-tuning gain it cannot measure. The lesson is the workflow, not a gain.\n\n"
        "**Who this is for.** A learner who knows basic Python and PIL and wants to see how a pretrained image classifier is re-headed and "
        "fine-tuned for new classes, and how the result is measured without fooling themselves: baselines first, a split that keeps copies "
        "of one photograph together, and an honest reading of a tie. No prior experience with DeiT, vision transformers, PyTorch training loops or fine-tuning "
        "is assumed; each term is explained where it is first used and again in the **Glossary** at the end.\n\n"
        "**Input → Model → Output.**\n\n"
        "| | Inference (ImageNet head) | Adaptation (two-class head) |\n"
        "|---|---|---|\n"
        "| Input | one RGB image, sides 8..4,096 px, and `TOP_K` | 280 labelled CIFAR-10 thumbnails (`frog`, `truck`, 32×32 px, upsampled to 224 px) |\n"
        "| Model | DeiT-Small with its 1000-class ImageNet head | the same network with a fresh 384-to-2 head; every weight trained by default (`FREEZE_BACKBONE = False`) |\n"
        "| Output | the `TOP_K` ImageNet labels with softmax scores, and a `sample-sanity` report | held-out accuracy and balanced accuracy next to three baselines, unseen-split predictions, and a SafeTensors adapter that reloads to the same predictions |\n\n"
        "**How to use this notebook.** Choose a runtime (a GPU runtime is much faster; CPU works), then **Runtime → Run all**. Sections 1–3 are "
        "**infrastructure** — the isolated environment, the carried code (collapsed) and the model verification — and can be run without study. "
        "The learning path starts in Section 4. Form fields (`# @param`) are the only values meant to be edited; the defaults reproduce the "
        "recorded path. Before Sections 4, 5, 6, 7, 8, 9 and 10 run you are asked to **predict** what they will print; the following section "
        "opens with **What to notice** and a collapsible **Check your reasoning** block with a worked answer from a recorded run. Section 8 "
        "always starts training from the freshly re-headed model, so re-running it after a change compares the new setting against the same "
        "starting point. Section 14 is a **change-one-thing activity**; it and each optional experiment name the field to change and the cell "
        "to re-run from (**Runtime → Run after**). **Troubleshooting**, a **Glossary** and a **Conclusion** template are at the end. Your notes "
        "are optional and are not required submissions.\n\n"
        "**Roadmap:** 4 download, verify and validate the sample, and group its copies → 5 split by group, ImageNet top-5 and the zero-shot "
        "baseline → 6 blank and noise probes → 7 re-head and measure the untrained head → 8 bounded fine-tune → 9 held-out comparison, and "
        "what a tie means → 10 unseen split → 11 export and reload the adapter → 12 outputs → 13 optional BYOD → 14 **change one thing: freeze "
        "the backbone** → conclude."
    ),
    "learning_objectives": (
        "by the end you should be able to (1) describe the path *image → resize and crop to 224 px → 196 patches of 16×16 px, a `[CLS]` token and position embeddings → 12 transformer layers → the 384-wide `[CLS]` token "
        "→ a softmax over the head's classes* and read a top-5 list (Section 5); (2) explain why copies of one photograph must stay on one side "
        "of a split, and check it from the pixels (Sections 4–5); (3) build a zero-shot baseline by mapping ImageNet classes onto task labels, "
        "and say why it must be measured before any fine-tune (Section 5); (4) explain why a closed-set classifier labels a blank or noise "
        "image confidently (Sections 6 and 10); (5) say what an untrained head's score does and does not tell you (Section 7); (6) read a "
        "training loss as optimisation evidence and held-out accuracy as task evidence, and state what a tie at the ceiling shows (Sections "
        "8–9); (7) check that an exported adapter reproduces the evaluated model (Section 11); and (8) predict and then measure what freezing "
        "the backbone changes when both runs start from the same re-headed model (Section 14). Along the way the notebook builds a locked "
        "runtime, stages and digest-verifies the immutable upstream model revision and the pinned dataset, and writes machine-readable outputs "
        "with provenance."
    ),
    "exclusions": (
        "a fine-tuning gain over the zero-shot baseline (frog versus truck is at or near ceiling for the pretrained model in the recorded runs, so "
        "the default comparison cannot show a measurable gain); ImageNet benchmark results (nothing here re-scores the ImageNet validation set); CIFAR-10 "
        "benchmark results (the tutorial uses two classes and a few hundred images); real-world frog or vehicle recognition (CIFAR-10 images are "
        "32×32 thumbnails, upsampled here to 224 px); open-set recognition or a reject option (every image receives one of the known classes); "
        "calibrated probabilities; object detection, segmentation or multi-label tagging."
    ),
    "prerequisites": [
        "- **Learner:** basic Python and PIL, and Colab or Jupyter familiarity; no prior experience with DeiT, vision transformers or fine-tuning. The notebook explains softmax, the zero-shot mapping, re-heading, accuracy, balanced accuracy and the adapter where they are first used; the Glossary repeats them.",
        "- **Runtime:** a fresh supported runtime (Google Colab, Kaggle or Linux Jupyter — **Linux x86_64 only**; the notebook builds its own isolated Python 3.12.12 environment, so a Windows or macOS kernel is not supported). A CUDA GPU such as a Colab or Kaggle T4 is recommended for the fine-tune and is used automatically when present; the notebook also runs on CPU, more slowly. Measured, each figure with its environment: a Kaggle Tesla T4 run of the earlier in-kernel-install revision (notebook blob `1a0317f0`, 2026-09-26) took 348.7 s in two passes — 264.6 s until the install cell stopped for a restart, then 84.1 s for every cell after the restart, about 15 s of it the fine-tune; a local CPU run of the same blob (24-thread Windows workstation, 2026-10-03, install skipped, files pre-staged) took 87.1 s of cell time, 72.3 s of it the fine-tune; a local CPU run of this revision (same workstation, 2026-10-04, install skipped, files pre-staged, host shared with other jobs) took 257 s of cell time, 216 s of it the fine-tune. This revision's isolated-environment build has not yet been timed on a hosted runtime; expect it to add several minutes (an estimate). The locked install (PyTorch 2.14.0 with its CUDA libraries) is the largest download; the checkpoint is about 88 MB.",
        "- **Knowledge:** basic Python and PIL; what a softmax over classes is; the idea of an image transformer operating on patches; accuracy, balanced accuracy and a confusion matrix; why a baseline is needed before a score means anything.",
        "- **Data:** the default path downloads one pinned archive, `CIFAR-10-subset.zip` from the Hugging Face dataset `Cleanlab/cifar-10-subset` (MIT licence, 986,707 bytes, verified by SHA-256 before it is opened), and keeps its `frog` and `truck` folders. BYOD is optional and off by default. Expected BYOD input: one image, or a directory or `.zip` of class folders, `<class>/<image>`, with at least 2 classes and at least 2 images per class that are not copies of each other (more is better: 4 or more independent images per class also gives an unseen split). A `.tar` or other archive is not read; unpack it or re-zip it.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so; uploaded inputs stay in this runtime and are not sent to any inference API. The default path uploads nothing.",
    ],
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime builds an isolated environment from the hash-locked pins (the kernel's own "
        "packages are left alone, so no restart is needed), stages and digest-verifies the pinned checkpoint, downloads and digest-verifies "
        "the pinned CIFAR-10 subset, validates it, groups its duplicate images and splits it by group, measures the majority-class and "
        "zero-shot ImageNet baselines, probes the model with a blank and a noise image, **runs the bounded fine-tune**, evaluates the held-out "
        "split, classifies an unseen split, exports the adapter, reloads it onto a fresh base model to verify the predictions, and writes "
        "machine-readable outputs with provenance. Nothing is skipped behind a default-off flag, and no clone, DIMER worker, credential, upload "
        "dialog or configuration edit is required (NOTEBOOK_SPEC 2.2 §5, RUN7, FT2). CUDA is used when present; a GPU runtime is recommended. "
        "Measured times are listed under Prerequisites."
    ),
    "byod": (
        "Two optional BYOD branches are included, and both are off by default (`USE_BYOD_IMAGE = False`, `USE_BYOD_DATASET = False`). "
        "`USE_BYOD_IMAGE` runs your own image through the same validation, ImageNet prediction and evaluation-report stages as the sample. "
        "`USE_BYOD_DATASET` takes your own class folders through the full adaptation workflow — validate, group duplicates, split by group, "
        "baselines, fine-tune, evaluate, export, reload and compare — under NOTEBOOK_SPEC 2.2 DAT14, and writes its own result JSON and "
        "predictions CSV. Set `BYOD_IMAGE_PATH` or `BYOD_DATASET_PATH` (a directory or a `.zip`) to read from a location without an upload "
        "dialog (EXE2)."
    ),
    "cells": [
        # ---------------------------------------------------------------- 4. Dataset
        {
            "md": (
                "## 4. Download, verify and validate the labelled sample, and group its copies\n\n"
                "`fetch_sample_archive` downloads `CIFAR-10-subset.zip` from the `Cleanlab/cifar-10-subset` dataset **at a pinned commit** and "
                "checks its byte size and SHA-256 before any member is opened. There is no fallback: a mismatch stops the notebook. "
                "`read_class_archive` then reads the `frog` and `truck` class folders. It refuses absolute member paths and `..` segments, and it "
                "bounds the member count and the uncompressed size before decompressing anything.\n\n"
                "`validate_dataset` checks every record before any model runs: record keys, image type and size, and labels. It reports classes "
                "with too few images as errors, and imbalance and pixel-identical duplicates as findings.\n\n"
                "**The archive is not 400 independent photographs.** It holds each of 200 CIFAR-10 photographs twice: once under "
                "`original_images/<class>/image_N.png` and once under `darkened_images/<class>/image_N.png`. For 100 of them the two files are "
                "pixel-identical; for the other 100 the second copy is the same photograph darkened to about 30% brightness. A copy on each side "
                "of a split would let the held-out score partly re-score training images. `assign_duplicate_groups` therefore joins every "
                "pixel-identical pair, and every near-duplicate pair — 16×16 grayscale thumbnails, mean-centred and scaled to unit length, "
                "with a correlation of at least `NEAR_DUPLICATE_CORRELATION` (0.95), a measure that ignores overall brightness — into one "
                "**group**, and adds its key to each record as `group`. Section 5 splits by that key.\n\n"
                "**What to look for:** CIFAR-10 images are 32×32 pixels. The model's preprocessing upsamples them to 224×224, so every image the "
                "network sees here is a blurred thumbnail, far from the photographs ImageNet was collected from.\n\n"
                "**Predict before running:** of the 400 records, how many groups will remain once each photograph's copies are joined, and how "
                "many of the joins will be exact pixel copies rather than darkened ones?"
            ),
            "code": (
                "import hashlib\n"
                "import json\n"
                "import os\n"
                "import time\n"
                "from pathlib import Path\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "OUTPUTS = Path('outputs')\n"
                "DATA_DIR = Path('data')\n\n"
                'PER_CLASS = 200  # @param {{type:"integer"}}\n'
                'DATASET_SEED = 0  # @param {{type:"integer"}}\n'
                'EPOCHS = 5  # @param {{type:"integer"}}\n\n'
                "archive_info = fetch_sample_archive(DATA_DIR, allow_download=True)\n"
                "print({{k: archive_info[k] for k in ('dataset_id', 'revision', 'license', 'bytes', 'sha256', 'fetched')}})\n"
                "records = read_class_archive(DATA_DIR / SAMPLE_DATASET_FILE, classes=SAMPLE_CLASSES, max_per_class=PER_CLASS, seed=DATASET_SEED)\n"
                "dataset_manifest = validate_dataset(records, SAMPLE_CLASSES, epochs=EPOCHS)\n"
                "print(json.dumps(dataset_manifest, indent=2))\n\n"
                "# DEIT-M2: copies of one photograph (pixel-identical or darkened) share a group key; Section 5 splits by it.\n"
                "records, duplicate_summary = assign_duplicate_groups(records)\n"
                "print(json.dumps(duplicate_summary, indent=2))\n\n"
                "preview = Image.new('RGB', (8 * 64, 2 * 64))\n"
                "for row, label in enumerate(SAMPLE_CLASSES):\n"
                "    for col, record in enumerate([r for r in records if r['label'] == label][:8]):\n"
                "        preview.paste(record['image'].resize((64, 64), Image.NEAREST), (64 * col, 64 * row))\n"
                "preview"
            ),
        },
        # ---------------------------------------------------------------- 5. Split & zero-shot
        {
            "md": (
                "**What to notice (Section 4):** `duplicate_groups` in the manifest, and `groups`, `exact_copies` and `near_duplicate_links` "
                "in the grouping summary.\n\n"
                "<details><summary>Check your reasoning</summary>In the recorded run (local CPU run of this revision, 2026-10-04, real weights, every field at its default) `validate_dataset` reported `duplicate_groups: 100` — the 100 pixel-identical pairs — and the grouping summary read 400 records → **200 groups**, every one of them a pair: 100 `exact_copies` and 100 `near_duplicate_links` (the darkened copies), with no group mixing two labels. The sample therefore holds 200 photographs, not 400, and a split by single images would put copies of many of them on both sides.</details>\n\n"
                "## 5. Split by group, and measure two baselines before any training\n\n"
                "The records are split per class with a fixed seed, **by group**: whole groups are shuffled and cut so that about 70% of the "
                "images train, and the rest is halved, again by group, into a **held-out** split, which scores every method below, and an "
                "**unseen** split, used only in Section 10. No hyperparameter is chosen on either. The cell then checks the result from the "
                "pixels, independently of the group keys: `cross_split_duplicates` counts held-out and unseen images that have a pixel-identical "
                "copy or a near duplicate in the training split, and the cell stops if any count is above 0. A split by single images would "
                "not be valid here, because the archive's copies are not independent; images that share a photograph, a scene or a camera "
                "session must always be split by that group.\n\n"
                "Two baselines give the fine-tune something to beat:\n\n"
                "- **Majority class:** always answer the most frequent training label. On a balanced split this is 50%, and any useful model must "
                "clear it.\n"
                "- **Zero-shot ImageNet mapping:** the pretrained 1000-class head already has frog and truck classes. `zero_shot_evaluate` sums the "
                "softmax mass over ImageNet's three frog classes and over its big-truck classes (fire engine, garbage truck, moving van, tow truck, "
                "trailer truck; CIFAR-10's `truck` excludes pickups), and answers whichever group is larger. No weight changes.\n\n"
                "The cell also prints ImageNet top-5 predictions for four images and a `sample-sanity` evaluation report: does the true group appear "
                "in the top-1 or the top-5?\n\n"
                "**Predict before running:** will the ImageNet head put a frog class at top-1 for the frog thumbnails? And how far above the "
                "50% majority baseline will the zero-shot mapping land on the held-out split?"
            ),
            "code": (
                'SEED = 0  # @param {{type:"integer"}}\n'
                'TOP_K = 5  # @param {{type:"integer"}}\n\n'
                "train_records, rest = split_dataset(records, train_fraction=0.7, seed=SEED, group_key='group')\n"
                "held_out, unseen = split_dataset(rest, train_fraction=0.5, seed=SEED, group_key='group')\n"
                "ids = [{{r['id'] for r in part}} for part in (train_records, held_out, unseen)]\n"
                "assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]), 'split leaked records'\n"
                "leakage = cross_split_duplicates(train_records, {{'held_out': held_out, 'unseen': unseen}})\n"
                "leakage['unseen_vs_held_out'] = cross_split_duplicates(held_out, {{'unseen': unseen}})['unseen']\n"
                "if any(row['pixel_copy_in_reference'] or row['near_duplicate_in_reference'] for row in leakage.values()):\n"
                "    raise RuntimeError(f'copies of one photograph straddle the split: {{leakage}}; split with group_key=\"group\" after assign_duplicate_groups')\n"
                "TRAIN_MAJORITY = majority_class(train_records)\n"
                "print(json.dumps({{'train': len(train_records), 'held_out': len(held_out), 'unseen': len(unseen), 'train_majority': TRAIN_MAJORITY,\n"
                "                  'duplicates_across_splits': leakage}}, indent=2))\n\n"
                "sample = held_out[:2] + held_out[-2:]\n"
                "input_manifest = validate_inputs([r['image'] for r in sample], top_k=TOP_K, names=[r['id'] for r in sample])\n"
                "try:\n"
                "    validate_inputs([r['image'] for r in sample], top_k=0)\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'probe': 'top_k=0', 'rejected': str(exc)}})\n"
                "imagenet_result = pipe.predict([r['image'] for r in sample], top_k=TOP_K)\n"
                "for record, pred in zip(sample, imagenet_result['predictions'], strict=True):\n"
                "    print(record['label'], '->', [(item['label'].split(',')[0], round(item['score'], 3)) for item in pred['top_k']])\n"
                "imagenet_report = evaluation_report(imagenet_result, [r['label'] for r in sample], groups=IMAGENET_GROUPS, sample_kind='sample')\n"
                "print({{'verdict': imagenet_report['verdict'], 'metrics': [(m['k'], m['value']) for m in imagenet_report['metrics']]}})\n\n"
                "zero_shot = pipe.zero_shot_evaluate(held_out, IMAGENET_GROUPS, majority=TRAIN_MAJORITY)\n"
                "print(json.dumps({{k: zero_shot[k] for k in ('accuracy', 'balanced_accuracy', 'majority_baseline_accuracy', 'mean_group_mass', 'per_class_recall')}}, indent=2))"
            ),
        },
        # ---------------------------------------------------------------- 6. Degenerate inputs
        {
            "md": (
                "**What to notice (Section 5):** the split sizes, the four zeros under `duplicates_across_splits`, the top-1 label of each "
                "sample image, and the zero-shot `accuracy`.\n\n"
                "<details><summary>Check your reasoning</summary>The recorded run split 280 / 60 / 60 (train majority `frog`: the classes tie, and the tie goes to the first name) and all four cross-split counts were 0 (also for `SEED` 0–4). Before this fix a split by single images with the same seed left 25 of the 60 held-out and 19 of the 60 unseen images with a pixel-identical copy in training, and 46 and 36 with a copy of either kind. Both frog thumbnails got `fox squirrel` at top-1 (0.387 and 0.571; `tailed frog` second in both) and both truck thumbnails `revolver` 0.306, with `moving van` second — identical lists, because those two are the two copies of one photograph, kept together in the held-out split; `sample-sanity` read top-1 0.0, top-5 1.0. The ImageNet top-1 labels look wrong on 32 px thumbnails, yet the zero-shot mapping, which adds up the scores of all frog classes and of all truck classes, scored 0.983 (59 of 60; recall frog 0.967, truck 1.000), far above the 0.500 majority baseline: the pretrained head already separates these two classes.</details>\n\n"
                "## 6. Degenerate input probes: blank canvas and noise\n\n"
                'A closed-set classifier has no "none of these" answer: every image gets one of its classes, with a score that sums to 1 across '
                "them. The cell shows what the ImageNet head answers for a white image and for uniform noise, and how high its top score is.\n\n"
                "**What to look for:** a confident label on structure-free input is a property of softmax classification, not evidence about the "
                "image. The same probe is repeated on the adapted two-class model in Section 10.\n\n"
                "**Predict before running:** with 1000 classes to spread over, will the top score for a blank image be above or below 0.1?"
            ),
            "code": (
                "degenerate = {{}}\n"
                "for name, image in (('blank', blank_image()), ('noise', noise_image(0))):\n"
                "    pred = pipe.predict(image, top_k=3)['predictions'][0]\n"
                "    degenerate[name] = [(item['label'].split(',')[0], round(item['score'], 3)) for item in pred['top_k']]\n"
                "print(json.dumps(degenerate, indent=2))"
            ),
        },
        # ---------------------------------------------------------------- 7. Re-head & baseline
        {
            "md": (
                "**What to notice (Section 6):** the top label and score for each probe.\n\n"
                "<details><summary>Check your reasoning</summary>Blank: `nematode` 0.002, `switch` 0.002, `hook` 0.002 — far below 0.1, spread thinly over 1000 classes. Noise: `jellyfish` 0.016, `nematode` 0.015, `kite` 0.013 (the same values as the Kaggle T4 record of the earlier revision). The labels are meaningless; the head must answer something, and here it answers with almost no confidence.</details>\n\n"
                "## 7. Replace the head and measure the untrained two-class model\n\n"
                "`from_pretrained(class_names=SAMPLE_CLASSES)` loads the verified checkpoint again and replaces only the classification layer "
                "(`classifier`, 384 features to 1000 classes) with a new 384-to-2 layer initialised under `SEED`. Every other weight keeps its "
                "ImageNet value.\n\n"
                "**An untrained head is not a chance baseline.** Its weights are random, so it knows nothing about which label is which — but it "
                "reads the network's strong pretrained features, so a random direction through them can separate frogs from trucks by accident, "
                "in either direction. Its accuracy can land well below or well above 50%, and it changes with `SEED`. This row says what the "
                "starting point of the fine-tune looks like; it is not evidence that the fine-tune, rather than the pretrained features, solved "
                "the task. The zero-shot row of Section 5 is the baseline that answers that question.\n\n"
                "**Predict before running:** will the untrained head score exactly 50%, below it, or above it?"
            ),
            "code": (
                "adapter = DeiTPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, class_names=SAMPLE_CLASSES, seed=SEED)\n"
                "print({{'class_names': list(adapter.class_names), 'reinitialised': list(adapter.reinitialised), 'device': adapter.device}})\n"
                "untrained = adapter.evaluate(held_out, majority=TRAIN_MAJORITY)\n"
                "print({{k: untrained[k] for k in ('accuracy', 'balanced_accuracy', 'per_class_recall')}})"
            ),
        },
        # ---------------------------------------------------------------- 8. Fine-tune
        {
            "md": (
                "**What to notice (Section 7):** the untrained head's `accuracy` and its per-class recall.\n\n"
                "<details><summary>Check your reasoning</summary>The recorded run (local CPU run of this revision, 2026-10-04, real weights, every field at its default) measured 0.600 (36 of 60; recall frog 0.667, truck 0.533) at `SEED = 0` — above the 0.500 majority baseline, with no training at all. Across `SEED` 0–4 on the same split the untrained head scored 0.600, 0.200, 0.933, 0.400 and 0.517: a random head carries no information about the task and can land well below (or above) 50%, depending only on the random direction it starts from.</details>\n\n"
                "## 8. Bounded fine-tuning\n\n"
                "This cell runs the real adaptation step in this runtime: gradient fine-tuning with cross-entropy over the two classes.\n\n"
                "- **What trains:** with `FREEZE_BACKBONE = False` (the default) every weight trains. With `True`, only the new head trains and "
                "the transformer (`vit.`) keeps its weights; ViT normalises with LayerNorm, so there are no running statistics to hold. The cell "
                "prints the trainable and total parameter counts.\n"
                "- **Schedule:** `EPOCHS` epochs of AdamW at learning rate `1e-4` with weight decay `0.01`, batch size 16, float32, seed `SEED`, "
                "and no data augmentation: training images go through the same evaluation transform as test images.\n"
                "- **Adaptation always starts from the re-headed model.** `finetune` changes the model in place. If you re-run this cell after the "
                "model has been fine-tuned — for example after changing `FREEZE_BACKBONE` — it first rebuilds `adapter` from the verified snapshot "
                "with the same `SEED` (the same fresh head that Section 7 measured), so a changed setting is compared from the same starting "
                "point instead of training an already adapted model further.\n\n"
                "**Read the loss as optimisation evidence only.** A falling training loss says the optimizer is fitting the training images; the "
                "held-out comparison in the next section is the task evidence.\n\n"
                "**Predict before running:** will the first epoch's loss be above or below 0.5, and how close to zero will the fifth be?"
            ),
            "code": (
                'LEARNING_RATE = 1e-4  # @param {{type:"number"}}\n'
                'BATCH_SIZE = 16  # @param {{type:"integer"}}\n'
                'FREEZE_BACKBONE = False  # @param {{type:"boolean"}}\n\n'
                "if adapter.adapted:\n"
                "    # A re-run: start again from the re-headed base (same SEED, same head as the Section 7 baseline).\n"
                "    adapter = DeiTPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, class_names=SAMPLE_CLASSES, seed=SEED)\n"
                "    print({{'rebuilt_from_verified_snapshot': True, 'adapted': adapter.adapted}})\n"
                "train_started = time.perf_counter()\n"
                "run = adapter.finetune(\n"
                "    train_records,\n"
                "    epochs=EPOCHS,\n"
                "    batch_size=BATCH_SIZE,\n"
                "    learning_rate=LEARNING_RATE,\n"
                "    seed=SEED,\n"
                "    freeze_backbone=FREEZE_BACKBONE,\n"
                "    progress=lambda row: print(f\"epoch {{row['epoch']}}/{{row['epochs']}}  loss {{row['loss']:.4f}}\"),\n"
                ")\n"
                "train_seconds = round(time.perf_counter() - train_started, 1)\n"
                "print(json.dumps({{**{{key: run[key] for key in ('freeze_backbone', 'trainable_parameters', 'total_parameters', 'epochs',\n"
                "                                             'batch_size', 'learning_rate', 'optimizer', 'augmentation', 'precision', 'device')}},\n"
                "                  'train_seconds': train_seconds}}, indent=2))"
            ),
        },
        # ---------------------------------------------------------------- 9. Evaluate held-out
        {
            "md": (
                "**What to notice (Section 8):** `trainable_parameters` against `total_parameters`, the five epoch losses, and `train_seconds`.\n\n"
                "<details><summary>Check your reasoning</summary>The recorded run (local CPU run of this revision, 2026-10-04, real weights, every field at its default) trained all 21,666,434 of 21,666,434 parameters; the losses were 0.1378, 0.0684, 0.0062, 0.0003 and 0.0003, and `train_seconds` was 215.7 on a shared 24-thread CPU (the Kaggle T4 record of the earlier revision took about 15 s). The loss falls to almost zero: the network fits its 280 training images nearly perfectly, which is optimisation evidence only; whether the model now classifies new images well is Section 9's question.</details>\n\n"
                "## 9. Evaluate on the held-out split, and read the comparison honestly\n\n"
                "All four rows are scored on the same held-out images. Each row shows how many images it answered correctly out of how many, "
                "the `accuracy` that count gives, its **95% Wilson interval**, and `balanced_accuracy`, the mean of the per-class recalls, which "
                "cannot be inflated by favouring the larger class. The Wilson interval is the range of true accuracies that are compatible with "
                "the count at the 95% level; with a few dozen images one image moves the accuracy by a few points, so read rows whose intervals "
                "overlap as indistinguishable on this split. The majority-class row is the baseline every method must clear. The confusion "
                "matrix shows which way the errors go. These are tutorial metrics from one seeded split and one training run.\n\n"
                "**The comparison that matters is fine-tuned against zero-shot**, not fine-tuned against the untrained head: the untrained head "
                "is the fine-tune's starting point, while the zero-shot mapping is what the pretrained model already does without any training. "
                "The cell computes a reading from the counts, not a fixed conclusion: when the zero-shot mapping already gets every held-out "
                "image right (or all but one), the comparison is **at ceiling** — the fine-tune can only tie, and the run measures no "
                "fine-tuning gain, whatever it prints for the fine-tuned row; otherwise it says whether the two Wilson intervals overlap.\n\n"
                "Each run of this cell also adds a row to `RUN_HISTORY`, so the Section 14 activity can put two settings side by side.\n\n"
                "**Predict before running:** will the fine-tuned row beat the zero-shot row, tie it, or fall below it?"
            ),
            "code": (
                "adapted = adapter.evaluate(held_out, majority=TRAIN_MAJORITY)\n"
                "N_HELD = adapted['n']\n"
                "comparison = {{\n"
                "    'majority class': {{'accuracy': adapted['majority_baseline_accuracy'], 'balanced_accuracy': 1 / len(SAMPLE_CLASSES)}},\n"
                "    'zero-shot ImageNet mapping': zero_shot,\n"
                "    'untrained two-class head': untrained,\n"
                "    'fine-tuned': adapted,\n"
                "}}\n"
                "held_out_intervals = {{}}\n"
                "print(f\"{{'method':<28s}} {{'correct':>9s}} {{'accuracy':>9s}} {{'95% Wilson':>15s}} {{'balanced':>9s}}\")\n"
                "for name, row in comparison.items():\n"
                "    correct = round(row['accuracy'] * N_HELD)\n"
                "    low, high = wilson_interval(correct, N_HELD)\n"
                "    held_out_intervals[name] = {{'correct': correct, 'n': N_HELD, 'accuracy': row['accuracy'], 'wilson_95': [round(low, 4), round(high, 4)]}}\n"
                "    print(f\"{{name:<28s}} {{f'{{correct}}/{{N_HELD}}':>9s}} {{row['accuracy']:>9.3f}} {{f'[{{low:.3f}}, {{high:.3f}}]':>15s}} {{row['balanced_accuracy']:>9.3f}}\")\n"
                "print('confusion (rows true, columns predicted):', adapted['confusion_matrix'])\n\n"
                "# DEIT-m2: a reading computed from the counts and intervals, not a fixed conclusion.\n"
                "zs, ft = held_out_intervals['zero-shot ImageNet mapping'], held_out_intervals['fine-tuned']\n"
                "if zs['correct'] >= N_HELD - 1:\n"
                "    comparison_verdict = (f\"at ceiling: the zero-shot mapping already answers {{zs['correct']}}/{{N_HELD}} held-out images correctly, \"\n"
                "                          f\"so the fine-tune ({{ft['correct']}}/{{N_HELD}}) can at best tie it; this run measures no fine-tuning gain\")\n"
                "elif ft['wilson_95'][0] > zs['wilson_95'][1]:\n"
                "    comparison_verdict = f\"fine-tuned ({{ft['correct']}}/{{N_HELD}}) is above zero-shot ({{zs['correct']}}/{{N_HELD}}): the 95% Wilson intervals do not overlap\"\n"
                "elif ft['wilson_95'][1] < zs['wilson_95'][0]:\n"
                "    comparison_verdict = f\"fine-tuned ({{ft['correct']}}/{{N_HELD}}) is below zero-shot ({{zs['correct']}}/{{N_HELD}}): the 95% Wilson intervals do not overlap\"\n"
                "else:\n"
                "    comparison_verdict = f\"fine-tuned {{ft['correct']}}/{{N_HELD}} and zero-shot {{zs['correct']}}/{{N_HELD}}: the 95% Wilson intervals overlap, so this split cannot tell them apart\"\n"
                "print('reading:', comparison_verdict)\n\n"
                "RUN_HISTORY = globals().get('RUN_HISTORY', [])\n"
                "RUN_HISTORY.append({{'run': len(RUN_HISTORY), 'freeze_backbone': run['freeze_backbone'], 'per_class': PER_CLASS, 'epochs': EPOCHS,\n"
                "                    'learning_rate': LEARNING_RATE, 'trainable_parameters': run['trainable_parameters'], 'held_out_n': adapted['n'],\n"
                "                    'zero_shot_accuracy': round(zero_shot['accuracy'], 4), 'fine_tuned_accuracy': round(adapted['accuracy'], 4),\n"
                "                    'final_loss': round(run['final_loss'], 4), 'train_seconds': train_seconds}})\n"
                "for row in RUN_HISTORY:\n"
                "    print(row)"
            ),
        },
        # ---------------------------------------------------------------- 10. Unseen data
        {
            "md": (
                "**What to notice (Section 9):** the four rows with their counts and Wilson intervals, the confusion matrix and the reading line.\n\n"
                "<details><summary>Check your reasoning</summary>The recorded run (local CPU run of this revision, 2026-10-04, real weights, every field at its default): majority class 30/60, 0.500 [0.377, 0.623]; zero-shot 59/60, 0.983 [0.911, 0.997]; untrained head 36/60, 0.600 [0.474, 0.714]; fine-tuned 60/60, 1.000 [0.940, 1.000], confusion `[[30, 0], [0, 30]]`. The reading was **at ceiling**: the zero-shot mapping already answers 59 of 60, so the fine-tune can at best tie it, and this run measures no fine-tuning gain. The fine-tune is far above the majority baseline and the untrained head, but its one-image edge over the zero-shot mapping is not evidence of a gain. A GPU run can differ by an image, which does not change that reading.</details>\n\n"
                "## 10. Inference on the unseen split and the degenerate probes again\n\n"
                "The unseen split was never used for training or for any comparison above, and no photograph in it has a copy in the training "
                "or held-out split (Section 5 checked). The adapted pipeline classifies it, and the cell repeats the blank and noise probes: the "
                "two-class model must now answer `frog` or `truck` for them, whatever they contain.\n\n"
                "**Predict before running:** what will the adapted model answer for the blank image, and with a score near 0.5 or near 1?"
            ),
            "code": (
                "unseen_metrics = adapter.evaluate(unseen, majority=TRAIN_MAJORITY)\n"
                "unseen_correct = round(unseen_metrics['accuracy'] * unseen_metrics['n'])\n"
                "unseen_metrics['wilson_95'] = [round(v, 4) for v in wilson_interval(unseen_correct, unseen_metrics['n'])]\n"
                "print({{'correct': unseen_correct, **{{k: unseen_metrics[k] for k in ('n', 'accuracy', 'wilson_95', 'balanced_accuracy')}}}})\n"
                "unseen_rows = []\n"
                "for record, pred in zip(unseen, adapter.predict([r['image'] for r in unseen[:MAX_BATCH]], top_k=2)['predictions'], strict=False):\n"
                "    unseen_rows.append({{'id': record['id'], 'truth': record['label'], 'predicted': pred['predicted_label'], 'score': round(pred['top_k'][0]['score'], 4)}})\n"
                "print(json.dumps(unseen_rows[:6], indent=2))\n"
                "adapted_degenerate = {{name: adapter.predict(image, top_k=2)['predictions'][0]['top_k'] for name, image in (('blank', blank_image()), ('noise', noise_image(0)))}}\n"
                "print({{name: [(i['label'], round(i['score'], 3)) for i in top] for name, top in adapted_degenerate.items()}})"
            ),
        },
        # ---------------------------------------------------------------- 11. Export, reload & verify
        {
            "md": (
                "**What to notice (Section 10):** the unseen `accuracy`, and the label and score the adapted model gives the blank and noise "
                "images.\n\n"
                "<details><summary>Check your reasoning</summary>The recorded run (local CPU run of this revision, 2026-10-04, real weights, every field at its default) classified the unseen split 60/60, 1.000 [0.940, 1.000]. The adapted head answered `truck` 0.547 for the blank image and `frog` 0.851 for noise (the Kaggle T4 record of the earlier revision: `frog` 0.598 and `frog` 0.912). A two-class head has no 'neither' answer, so it must pick one of its two classes for a white square, and which one can change from run to run.</details>\n\n"
                "## 11. Adapter export, fresh reload, and equivalence check\n\n"
                "`save_artifact` writes `outputs/deit_adapter.safetensors`: every tensor the fine-tune could change, plus a metadata header "
                "naming the base model, its pinned revision, the base `pytorch_model.bin` SHA-256, the class names and the frozen prefixes. With the "
                "default full fine-tune that is the whole state dict; with `FREEZE_BACKBONE = True` it is the head only.\n\n"
                "`load_artifact` then builds a **fresh** pipeline from the verified base snapshot, loads the adapter tensors onto it, and refuses an "
                "adapter whose format, base identity, base digest or tensor set does not fit. `reload_equivalence` compares the reloaded predictions "
                "with the in-memory model's on the unseen split, with a stated tolerance: loading succeeding is not the check, reproducing the "
                "predictions is. A mismatch stops the cell with a message naming the image, the two answers and what to do next. The same helper "
                "checks the BYOD adapter in Section 13."
            ),
            "code": (
                "artifact_path = OUTPUTS / 'deit_adapter.safetensors'\n"
                "descriptor = adapter.save_artifact(artifact_path, notes='DeiT-Small CIFAR-10 frog/truck tutorial adapter')\n"
                "print(json.dumps(descriptor, indent=2))\n\n"
                "reloaded = DeiTPipeline.load_artifact(artifact_path, weights_dir=WEIGHTS_DIR)\n"
                "print({{'reloaded_source': reloaded.source, 'adapted': reloaded.adapted, 'class_names': list(reloaded.class_names)}})\n\n"
                "TOLERANCE = 1e-4\n\n"
                "def reload_equivalence(first, second, images, tolerance=TOLERANCE):\n"
                "    \"\"\"Both pipelines must give every image the same label and a top score within `tolerance`.\"\"\"\n"
                "    advice = ('the export does not reproduce the evaluated model. If you changed a Section 8 field, re-run from Section 8 '\n"
                "              '(Runtime -> Run after) so the adapter is exported from the model that was evaluated; otherwise restart the '\n"
                "              'session and choose Run all')\n"
                "    first_preds = first.predict(images, top_k=2)['predictions']\n"
                "    second_preds = second.predict(images, top_k=2)['predictions']\n"
                "    for index, (a, b) in enumerate(zip(first_preds, second_preds, strict=True)):\n"
                "        if a['predicted_label'] != b['predicted_label']:\n"
                "            raise RuntimeError(f\"image {{index}}: the reloaded adapter answers {{b['predicted_label']!r}}, the in-memory model {{a['predicted_label']!r}}: {{advice}}\")\n"
                "        if abs(a['top_k'][0]['score'] - b['top_k'][0]['score']) > tolerance:\n"
                "            raise RuntimeError(f\"image {{index}}: top score {{b['top_k'][0]['score']:.6f}} after reload against {{a['top_k'][0]['score']:.6f}}, beyond tolerance {{tolerance}}: {{advice}}\")\n"
                "    return {{'images_compared': len(images), 'tolerance': tolerance, 'equivalent': True}}\n\n"
                "reload_check = reload_equivalence(adapter, reloaded, [r['image'] for r in unseen[:MAX_BATCH]])\n"
                "print(reload_check)"
            ),
        },
        # ---------------------------------------------------------------- 12. Outputs & provenance
        {
            "md": (
                "**What to notice (Section 11):** the number of tensors in the adapter, its size, and `equivalent: True`.\n\n"
                "<details><summary>Check your reasoning</summary>The recorded run (local CPU run of this revision, 2026-10-04, real weights, every field at its default) wrote 200 tensors (the whole network, 86,688,984 bytes) and the reload check compared 60 unseen images within 1e-4: `equivalent: True`. The header's `base_state_digest` is `7eadf70de01e…`, the SHA-256 of the base `pytorch_model.bin` in the Section 3 manifest. With `FREEZE_BACKBONE = True` the adapter holds only the 2 head tensors (3,632 bytes).</details>\n\n"
                "## 12. Write machine-readable outputs and provenance\n\n"
                "The cell writes:\n"
                "- `outputs/deit_classification_input_manifest.json`\n"
                "- `outputs/deit_classification_evaluation_report.json`\n"
                "- `outputs/deit_classification_result.json` (identity, runtime versions, device, dataset provenance and manifest, the "
                "duplicate grouping and the cross-split duplicate counts, split, baselines, fine-tuning configuration and time, held-out and "
                "unseen metrics with their Wilson intervals, the comparison reading, the run history, adapter descriptor and reload check)\n"
                "- `outputs/deit_classification_predictions.csv`\n"
                "- `outputs/deit_adapter.safetensors` (written in Section 11)"
            ),
            "code": (
                "import csv\n\n"
                "with open(OUTPUTS / '{stem}_input_manifest.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(input_manifest, f, indent=2)\n\n"
                "with open(OUTPUTS / '{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(imagenet_report, f, indent=2)\n\n"
                "def write_predictions(path, pipeline, parts):\n"
                "    with open(path, 'w', newline='', encoding='utf-8') as f:\n"
                "        writer = csv.writer(f)\n"
                "        writer.writerow(['split', 'id', 'group', 'truth', 'predicted', 'score'])\n"
                "        for split_name, part in parts:\n"
                "            for start in range(0, len(part), MAX_BATCH):\n"
                "                chunk = part[start:start + MAX_BATCH]\n"
                "                for record, pred in zip(chunk, pipeline.predict([r['image'] for r in chunk], top_k=1)['predictions'], strict=True):\n"
                "                    writer.writerow([split_name, record['id'], record.get('group', ''), record['label'], pred['predicted_label'], f\"{{pred['top_k'][0]['score']:.4f}}\"])\n\n"
                "write_predictions(OUTPUTS / '{stem}_predictions.csv', adapter, (('held_out', held_out), ('unseen', unseen)))\n\n"
                "result_export = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'torchvision': torchvision.__version__,\n"
                "                'transformers': transformers.__version__, 'cuda': torch.cuda.is_available()}},\n"
                "    'device': pipe.device,\n"
                "    'dataset': {{'archive': archive_info, 'manifest': dataset_manifest, 'per_class': PER_CLASS, 'seed': DATASET_SEED,\n"
                "                'duplicate_groups': duplicate_summary}},\n"
                "    'split': {{'train': len(train_records), 'held_out': len(held_out), 'unseen': len(unseen), 'seed': SEED, 'train_majority': TRAIN_MAJORITY,\n"
                "              'group_key': 'group', 'duplicates_across_splits': leakage}},\n"
                "    'imagenet_top_k': imagenet_result['predictions'],\n"
                "    'degenerate_probes': {{'imagenet_head': degenerate, 'adapted': adapted_degenerate}},\n"
                "    'baselines': {{'zero_shot': zero_shot, 'untrained_head': untrained}},\n"
                "    'finetune': {{**run, 'train_seconds': train_seconds}},\n"
                "    'held_out': adapted,\n"
                "    'comparison_verdict': comparison_verdict,\n"
                "    'held_out_intervals': held_out_intervals,\n"
                "    'run_history': RUN_HISTORY,\n"
                "    'unseen': unseen_metrics,\n"
                "    'artifact': descriptor,\n"
                "    'reload_check': reload_check,\n"
                "}}\n"
                "with open(OUTPUTS / '{stem}_result.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(result_export, f, indent=2, default=str)\n\n"
                "for p in sorted(OUTPUTS.iterdir()):\n"
                "    if p.is_file():\n"
                "        print(f'  {{p.name:<48s}} {{p.stat().st_size:>12,d}} bytes')"
            ),
        },
        # ---------------------------------------------------------------- 13. BYOD
        {
            "md": (
                "## 13. Optional: Bring Your Own Data (BYOD)\n\n"
                "Both branches are off by default, so `Run all` never stops here. Before you turn one on, read the contract:\n\n"
                "- **Image branch** (`USE_BYOD_IMAGE`): one image file PIL can open, each side between `MIN_IMAGE_SIDE` (8) and `MAX_IMAGE_SIDE` "
                "(4096) px. It runs through `validate_inputs`, the ImageNet head's `predict` and `evaluation_report`; the report is `not-measurable`, "
                "because no label comes with the image.\n"
                "- **Dataset branch** (`USE_BYOD_DATASET`): a directory or a `.zip` of class folders, `<class>/<image>`, with at least 2 classes and "
                "at least 2 images per class that are not copies of each other, at most 5,000 images. Archive members must stay inside the archive; "
                "folder files must not link outside the directory. The class names are the folder names; images directly at the top level are "
                "ignored, and the cell says how many. A layout with `train/` and `val/` folders is **merged by class name and re-split**, and the "
                "cell says so. The branch runs the same validate → group duplicates → split by group → baselines → fine-tune → evaluate → export → "
                "reload-and-compare stages as the sample, and classifies an unseen split when every class has enough independent images for one "
                "(4 or more); the zero-shot ImageNet baseline needs a class-to-ImageNet mapping, so it is skipped here. It writes "
                "`outputs/byod/byod_deit_classification_result.json` and `outputs/byod/byod_deit_classification_predictions.csv`.\n\n"
                "Set `BYOD_IMAGE_PATH` or `BYOD_DATASET_PATH` to read from a mounted or local location; leave them empty on Colab to get an upload "
                "dialog instead (outside Colab an empty location stops with a message naming the field to set). Uploaded files are written under "
                "`outputs/byod/` in this runtime, a new upload replaces the previous one, and nothing is sent anywhere else. The first lines of "
                "the cell show the validator refusing two malformed inputs with messages that name the failed rule."
            ),
            "code": (
                "import shutil\n\n"
                'USE_BYOD_IMAGE = False  # @param {{type:"boolean"}}\n'
                'BYOD_IMAGE_PATH = ""  # @param {{type:"string"}}\n'
                'USE_BYOD_DATASET = False  # @param {{type:"boolean"}}\n'
                'BYOD_DATASET_PATH = ""  # @param {{type:"string"}}\n\n'
                "for desc, probe in (\n"
                "    ('non-image object', lambda: validate_inputs('/not/an/image.png')),\n"
                "    ('class with one image', lambda: validate_dataset([{{'image': blank_image(32, 32), 'label': 'frog'}}, {{'image': noise_image(1, 32, 32), 'label': 'truck'}}], SAMPLE_CLASSES)),\n"
                "):\n"
                "    try:\n"
                "        probe()\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print(f'refused as expected: {{desc}} -> {{type(exc).__name__}}: {{exc}}')\n\n"
                "BYOD_DIR = OUTPUTS / 'byod'\n\n"
                "def _upload_into(target, field):\n"
                "    try:\n"
                "        from google.colab import files  # type: ignore[import-not-found]\n"
                "    except ImportError:\n"
                "        raise FileNotFoundError(f'{{field}} is empty and this runtime has no Colab upload dialog: set {{field}} to a local path') from None\n"
                "    shutil.rmtree(target, ignore_errors=True)  # a new upload replaces the previous one\n"
                "    target.mkdir(parents=True, exist_ok=True)\n"
                "    for name, data in files.upload().items():\n"
                "        (target / Path(name).name).write_bytes(data)\n"
                "    uploaded = sorted(target.iterdir())\n"
                "    if not uploaded:\n"
                "        raise FileNotFoundError(f'nothing was uploaded; upload a file or set {{field}}')\n"
                "    return uploaded[0]\n\n"
                "if USE_BYOD_IMAGE:\n"
                "    image_path = Path(BYOD_IMAGE_PATH) if BYOD_IMAGE_PATH else _upload_into(BYOD_DIR / 'image', 'BYOD_IMAGE_PATH')\n"
                "    try:\n"
                "        with Image.open(image_path) as handle:\n"
                "            byod_image = handle.convert('RGB')\n"
                "    except (OSError, SyntaxError) as exc:\n"
                "        raise ValueError(f'{{image_path}} is not an image PIL can open ({{exc}}); set BYOD_IMAGE_PATH to a .png, .jpg, .bmp or .webp file') from None\n"
                "    print(validate_inputs(byod_image, top_k=TOP_K, names=[image_path.name])['verdict'])\n"
                "    byod_result = pipe.predict(byod_image, top_k=TOP_K)\n"
                "    print([(item['label'].split(',')[0], round(item['score'], 3)) for item in byod_result['predictions'][0]['top_k']])\n"
                "    print(evaluation_report(byod_result, None, sample_kind='byod')['verdict'])\n"
                "else:\n"
                "    print('BYOD image branch is off; set USE_BYOD_IMAGE = True to classify your own image.')\n\n"
                "def _byod_layout(source):\n"
                "    \"\"\"Top-level image files (ignored) and top-level folder names of a class-folder directory or .zip.\"\"\"\n"
                "    if source.is_dir():\n"
                "        names = [p.name + ('/' if p.is_dir() else '') for p in source.iterdir()]\n"
                "    else:\n"
                "        import zipfile\n"
                "        with zipfile.ZipFile(source) as archive:\n"
                "            names = sorted({{n.split('/')[0] + ('/' if '/' in n else '') for n in archive.namelist()}})\n"
                "    loose = [n for n in names if not n.endswith('/') and Path(n).suffix.lower() in IMAGE_SUFFIXES]\n"
                "    folders = sorted(n.rstrip('/') for n in names if n.endswith('/'))\n"
                "    return loose, folders\n\n"
                "if USE_BYOD_DATASET:\n"
                "    source = Path(BYOD_DATASET_PATH) if BYOD_DATASET_PATH else _upload_into(BYOD_DIR / 'dataset', 'BYOD_DATASET_PATH')\n"
                "    if not source.exists():\n"
                "        raise FileNotFoundError(f'{{source}} does not exist; set BYOD_DATASET_PATH to a directory or a .zip of class folders')\n"
                "    if source.is_file() and source.suffix.lower() != '.zip':\n"
                "        raise ValueError(f'{{source.name}} is not a .zip: the dataset must be a directory or a .zip of <class>/<image> folders (unpack or re-zip a .tar)')\n"
                "    loose, folders = _byod_layout(source)\n"
                "    if loose:\n"
                "        print(f'note: {{len(loose)}} image(s) at the top level were ignored (e.g. {{loose[:3]}}); only images inside class folders are read')\n"
                "    if {{f.lower() for f in folders}} & {{'train', 'val', 'valid', 'validation', 'test'}}:\n"
                "        print(f'note: top-level folders {{folders}} look like a pre-made split; images are merged by class name and re-split by group below')\n"
                "    byod_records = read_class_archive(source) if source.suffix.lower() == '.zip' else read_class_folder(source)\n"
                "    byod_names = sorted({{r['label'] for r in byod_records}})\n"
                "    if len(byod_names) < 2:\n"
                "        raise ValueError(f'the dataset needs at least 2 class folders (<class>/<image>); found {{byod_names}}' + (f' and {{len(loose)}} loose top-level image(s); move them into class folders' if loose else ''))\n"
                "    byod_manifest = validate_dataset(byod_records, byod_names, epochs=EPOCHS)\n"
                "    byod_records, byod_duplicates = assign_duplicate_groups(byod_records)\n"
                "    print(json.dumps({{'manifest': byod_manifest, 'duplicate_groups': byod_duplicates}}, indent=2))\n"
                "    byod_train, byod_rest = split_dataset(byod_records, train_fraction=0.7, seed=SEED, group_key='group')\n"
                "    try:\n"
                "        byod_held, byod_unseen = split_dataset(byod_rest, train_fraction=0.5, seed=SEED, group_key='group')\n"
                "    except ValueError as exc:\n"
                "        byod_held, byod_unseen = byod_rest, []\n"
                "        print(f'note: no unseen split ({{exc}}); every class needs 4 or more independent images for one')\n"
                "    byod_parts = {{'held_out': byod_held, **({{'unseen': byod_unseen}} if byod_unseen else {{}})}}\n"
                "    byod_leakage = cross_split_duplicates(byod_train, byod_parts)\n"
                "    print({{'train': len(byod_train), **{{name: len(part) for name, part in byod_parts.items()}}, 'duplicates_across_splits': byod_leakage}})\n"
                "    byod_majority = majority_class(byod_train)\n"
                "    byod_pipe = DeiTPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, class_names=byod_names, seed=SEED)\n"
                "    byod_before = byod_pipe.evaluate(byod_held, majority=byod_majority)\n"
                "    byod_run = byod_pipe.finetune(byod_train, epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, seed=SEED, freeze_backbone=FREEZE_BACKBONE)\n"
                "    byod_after = byod_pipe.evaluate(byod_held, majority=byod_majority)\n"
                "    byod_unseen_metrics = byod_pipe.evaluate(byod_unseen, majority=byod_majority) if byod_unseen else None\n"
                "    print(f\"{{'held-out (n=' + str(byod_after['n']) + ')':<28s}} {{'accuracy':>9s}} {{'balanced':>9s}}\")\n"
                "    for name, acc, bal in (('majority class', byod_after['majority_baseline_accuracy'], 1 / len(byod_names)),\n"
                "                           ('untrained head', byod_before['accuracy'], byod_before['balanced_accuracy']),\n"
                "                           ('fine-tuned', byod_after['accuracy'], byod_after['balanced_accuracy'])):\n"
                "        print(f'{{name:<28s}} {{acc:>9.3f}} {{bal:>9.3f}}')\n"
                "    if byod_unseen_metrics:\n"
                "        print({{'unseen_accuracy': byod_unseen_metrics['accuracy'], 'unseen_balanced_accuracy': byod_unseen_metrics['balanced_accuracy'], 'n': byod_unseen_metrics['n']}})\n"
                "    (BYOD_DIR).mkdir(parents=True, exist_ok=True)\n"
                "    byod_artifact = BYOD_DIR / 'byod_deit_adapter.safetensors'\n"
                "    byod_descriptor = byod_pipe.save_artifact(byod_artifact, notes='BYOD adaptation adapter')\n"
                "    byod_reloaded = DeiTPipeline.load_artifact(byod_artifact, weights_dir=WEIGHTS_DIR)\n"
                "    byod_reload_check = reload_equivalence(byod_pipe, byod_reloaded, [r['image'] for r in (byod_unseen or byod_held)[:MAX_BATCH]])\n"
                "    write_predictions(BYOD_DIR / 'byod_{stem}_predictions.csv', byod_pipe, tuple(byod_parts.items()))\n"
                "    byod_export = {{\n"
                "        'notebook_source': NOTEBOOK_SOURCE, 'model_id': MODEL_ID, 'model_revision': MODEL_REVISION, 'model_license': MODEL_LICENSE,\n"
                "        'source': source.name, 'class_names': byod_names, 'manifest': byod_manifest, 'duplicate_groups': byod_duplicates,\n"
                "        'ignored_top_level_images': len(loose),\n"
                "        'split': {{'train': len(byod_train), **{{name: len(part) for name, part in byod_parts.items()}}, 'seed': SEED, 'group_key': 'group',\n"
                "                  'duplicates_across_splits': byod_leakage}},\n"
                "        'baselines': {{'majority_class_accuracy': byod_after['majority_baseline_accuracy'], 'untrained_head': byod_before}},\n"
                "        'finetune': byod_run, 'held_out': byod_after, 'unseen': byod_unseen_metrics,\n"
                "        'artifact': byod_descriptor, 'reload_check': byod_reload_check,\n"
                "    }}\n"
                "    with open(BYOD_DIR / 'byod_{stem}_result.json', 'w', encoding='utf-8') as f:\n"
                "        json.dump(byod_export, f, indent=2, default=str)\n"
                "    print('BYOD adapter exported, reloaded and compared:', byod_reload_check, byod_descriptor['sha256'][:16])\n"
                "else:\n"
                "    print('BYOD dataset branch is off; set USE_BYOD_DATASET = True to adapt DeiT-Small on your own class folders.')"
            ),
        },
    ],
    "closing": (
        "## 14. Your turn — change one thing: freeze the backbone\n\n"
        "Optional; **Predict → Change → Run → Observe → Explain**. It re-runs the fine-tune with only the new head trainable.\n\n"
        "1. **Predict:** with `FREEZE_BACKBONE = True`, will held-out accuracy go up, stay or go down against the default run, and will the "
        "fine-tune take more or less time? Write your guess next to row 0 of the Section 9 run history.\n"
        "2. **Change:** in Section 8 set `FREEZE_BACKBONE = True`. Change nothing else.\n"
        "3. **Run:** select the Section 8 cell and choose **Runtime → Run after**. Section 8 rebuilds the re-headed model from the verified "
        "snapshot before training (it prints `rebuilt_from_verified_snapshot`), Section 9 adds a row to the run history, and Sections 10–12 "
        "re-run on the new adapter. (If you re-run only the Section 8 cell, it still trains from the re-headed model; re-run Sections 9–12 "
        "afterwards so the comparison, the export and the outputs describe the same model.)\n"
        "4. **Observe:** in Section 8, `trainable_parameters` falls from every parameter to the head alone. In Section 9, compare rows 0 and "
        "1 of the run history: `fine_tuned_accuracy`, `zero_shot_accuracy`, `final_loss` and `train_seconds`. In Section 11, count the "
        "adapter's tensors.\n"
        "5. **Explain:** in one sentence, why can training 0.004% of the parameters reach nearly the same held-out accuracy as training all of "
        "them on this task?\n\n"
        "<details><summary>Check your reasoning</summary>In a local CPU run of this revision (2026-10-04, real weights, `FREEZE_BACKBONE = True` and Section 8 re-run after the default run, every other field at its default) Section 8 printed `rebuilt_from_verified_snapshot`, trained 770 of 21,666,434 parameters (0.004%), and its losses went 0.6671 → 0.2374 (the full fine-tune: 0.1378 → 0.0003). Section 11 straight after exported 2 tensors and reloaded equivalently. The run history then read: full fine-tune 60/60 held-out in 215.7 s; head only 60/60 [0.940, 1.000] in 50.9 s; zero-shot 59/60 in both rows; on the unseen split the head-only model scored 59/60. The pretrained features already separate frogs from trucks — the zero-shot row is at ceiling — so a linear head on top of them matches the full fine-tune on this split in five short epochs, in about a quarter of the time; the much higher final loss says the head-only model is less confident on the training images, not that it classifies worse. Identical held-out scores at the ceiling cannot rank the two runs.</details>\n\n"
        "## Interpretation and limits\n\n"
        "Start from your own run: the Section 9 table and its reading line, the Section 4 grouping summary and the Section 5 cross-split "
        "counts. Then read the reference answer.\n\n"
        "<details><summary>Reference answer from the recorded runs</summary>The ImageNet head gave the sample thumbnails non-frog, non-truck top-1 labels at 32 px, yet its summed frog and truck classes separated the two labels, and it spread its scores thinly over blank and noise images. The archive's 400 images are 200 photographs, each held twice; grouped, they split 280 / 60 / 60 with no copy across splits. On the 60 held-out images the zero-shot mapping scored 59/60 [0.911, 0.997]; the untrained head 36/60, which moves with the seed between 0.200 and 0.933; the full fine-tune and the head-only fine-tune both 60/60 [0.940, 1.000] — at ceiling, every interval overlaps the zero-shot one, so this run measures **no reliable fine-tuning gain** over what the pretrained model already does. The unseen split scored 60/60 and the adapter reloaded to the same predictions within 1e-4 (local CPU run of this revision, 2026-10-04; the Kaggle T4 run of the earlier revision, whose split was not duplicate-aware, also scored zero-shot 0.983 and fine-tuned 1.000 there).</details>\n\n"
        "**What this notebook established, in this runtime.** The pinned `facebook/deit-small-patch16-224` snapshot was verified against a committed "
        "SHA-256 manifest before loading, and the pinned CIFAR-10 subset was verified against its recorded digest before it was read. Its "
        "duplicate and darkened copies were grouped, the split kept every group on one side, and the held-out and unseen splits were checked "
        "from the pixels to hold no copy of a training image. The ImageNet head was run on sample thumbnails and on two structure-free probes, "
        "and its frog and truck classes gave a zero-shot baseline. The head was then replaced for the two tutorial classes and fine-tuned, and "
        "the result was compared with the majority-class, zero-shot and untrained baselines on the same held-out split and checked on an unseen "
        "split. The adapter was exported, reloaded onto a fresh base model and checked against the in-memory predictions.\n\n"
        "**What the comparison shows.** On frog versus truck the pretrained model is already at or near ceiling: the zero-shot mapping "
        "classifies 59 of the 60 held-out images correctly without any training, and the fine-tune's 60 of 60 is a one-image difference "
        "inside overlapping 95% intervals. This run shows that the fine-tune *works* — it trains, it matches the zero-shot result with a "
        "dedicated two-class head, and it exports faithfully — but not that it *improves* on what the pretrained model already does. The untrained head's score is a property of a random starting point, not a floor that the fine-tune lifts the model from. "
        "Measuring a fine-tuning gain needs a task the pretrained head does not already solve: your own classes through the BYOD dataset branch, "
        "for example.\n\n"
        "**What a green run proves.** Successful execution proves that the recorded repository revision, the pinned dependency set, the pinned "
        "checkpoint and the pinned dataset together reproduce these stages in a fresh runtime, without the repository being cloned or installed "
        "and without any DIMER worker or service. It does **not** establish benchmark superiority, fitness for any deployment, or that the "
        "adapted model recognises frogs or trucks outside CIFAR-10's 32×32 thumbnails. The held-out scores come from a few dozen images and carry "
        "no dispersion estimate. The scores are not calibrated probabilities, and there is no reject option.\n\n"
        "**Reproducibility.** Seeds are form fields (`DATASET_SEED`, `SEED`), the run is float32 with no data augmentation, and the new head is "
        "initialised under `SEED`. GPU kernels are not forced to be deterministic, so repeated GPU runs can differ in the last digits of the loss "
        "and the scores.\n\n"
        "**More experiments (optional).** Each one names the field to change and where to re-run from; a re-run of Section 8 always starts "
        "from the re-headed model, and Section 9's run history keeps the earlier rows.\n\n"
        "- **A smaller sample:** set `PER_CLASS = 10` in Section 4, select Section 4 and choose **Runtime → Run after**. With 10 images per class the grouping finds 19 groups (one darkened pair), the split is 14 / 2 / 4, and the held-out split holds 2 images, so one image is worth 50 points of accuracy. Does the fine-tune still beat zero-shot? If both score 2/2, the 95% Wilson interval of each is [0.342, 1.000]: a perfect score that says almost nothing (this split was measured locally; the scores at this size were not). Watch the interval, not the accuracy.\n"
        "- **A different starting point:** set `SEED = 1` in Section 5 and **Run after** from Section 5. The split, the untrained head and the "
        "fine-tune all change with it; compare how far the untrained row moves with how far the zero-shot and fine-tuned rows move.\n"
        "- **Your own classes:** turn on `USE_BYOD_DATASET` in Section 13 and re-run that cell only. Pick classes the ImageNet head does not "
        "already name, so the untrained-to-fine-tuned change has room to mean something.\n\n"
        "## Troubleshooting\n\n"
        "- **Section 1 stops with \"needs a Linux x86_64 runtime\".** The locked environment is built from manylinux wheels; use Google Colab, "
        "Kaggle or a Linux Jupyter server.\n"
        "- **Section 1 fails while downloading.** The `uv` wheel, the managed Python and the locked packages come from PyPI and "
        "python-build-standalone; run the cell again. A size or SHA-256 mismatch is refused on purpose — if it repeats, the download is being altered.\n"
        "- **A restart prompt.** This notebook never needs one: nothing is installed into the kernel. If the isolated process exits (usually out "
        "of memory), restart the session and choose **Run all**.\n"
        "- **Section 3 or 4 stops on a download or a digest mismatch.** The checkpoint and the sample archive are fetched from the Hugging Face "
        "Hub at pinned revisions and re-hashed; a mismatch is never loaded. Run the cell again; if it repeats, delete `weights/` or `data/` and "
        "run from that section.\n"
        "- **Section 5 stops with \"copies of one photograph straddle the split\".** The split was made without `group_key='group'` or before "
        "Section 4 grouped the records; run from Section 4.\n"
        "- **Out of memory in Section 8.** Lower `BATCH_SIZE` to 8 in Section 8, or set `FREEZE_BACKBONE = True`, and **Run after** from "
        "Section 8. On a CPU runtime the fine-tune is slow, not broken.\n"
        "- **Section 11 stops with \"the export does not reproduce the evaluated model\".** The model in memory was changed after it was "
        "evaluated; re-run from Section 8 (**Runtime → Run after**).\n"
        "- **BYOD is refused.** Each refusal names the rule: fewer than 2 class folders, a class with fewer than 2 independent images, a path "
        "with `..`, an undecodable image, an image side outside 8..4,096 px, a file that is not a directory or a `.zip`, or an empty location "
        "outside Colab. Fix the input or the location field, then re-run Section 13.\n\n"
        "## Glossary\n\n"
        "- **DeiT / patch embedding / `[CLS]` token:** a Vision Transformer (ViT) that cuts the 224 px image into 196 patches of 16×16 px, "
        "projects each to a 384-wide token, adds position embeddings and one extra learned `[CLS]` token, and passes the sequence through "
        "12 transformer layers; the head reads only the final `[CLS]` token.\n"
        "- **Data-efficient training:** DeiT's recipe (strong augmentation and regularisation) for training a ViT on ImageNet-1k alone, "
        "without a larger pre-training set. The DeiT paper also adds a distillation token; this checkpoint is the plain DeiT-Small "
        "without it.\n"
        "- **Softmax score:** the head's output turned into numbers that sum to 1 over its classes; the label is the largest. Not a calibrated "
        "probability, and never \"none of these\".\n"
        "- **Zero-shot mapping:** using the ImageNet head unchanged, adding its scores for the ImageNet classes that correspond to each task "
        "label; no weight changes.\n"
        "- **Re-heading:** replacing the 1000-class layer with a freshly initialised one for the task's classes.\n"
        "- **Duplicate group:** records that are pixel-identical or near duplicates of one photograph; a split keeps each group on one side.\n"
        "- **Held-out split:** images never shown to the optimizer, never used to choose a setting, and with no copy in the training split; "
        "the task evidence.\n"
        "- **Accuracy / balanced accuracy:** the fraction answered correctly, and the mean of the per-class recalls.\n"
        "- **At ceiling:** a baseline already scores (nearly) every test image correctly, so no method can measurably beat it on that test set.\n"
        "- **Cross-entropy / AdamW:** the training loss for one label per image, and the optimizer that minimises it with decoupled weight decay.\n"
        "- **Adapter / reload equivalence:** the SafeTensors file of the tensors the fine-tune changed, and the check that a fresh base "
        "model with the adapter loaded gives the same predictions.\n\n"
        "## Conclusion (your notes)\n\n"
        "Optional. Fill in from your own run, one sentence each:\n\n"
        "1. The 400 records formed ___ groups; ___ held-out images had a copy in the training split after the group split.\n"
        "2. The zero-shot mapping scored ___ on the held-out split, the untrained head ___, and the fine-tuned head ___ (with Wilson intervals ___); the reading line said ___.\n"
        "3. From these numbers, the fine-tune's measured gain over zero-shot is ___, because ___.\n"
        "4. In Section 14, freezing the backbone changed held-out accuracy by ___ and the training time by ___.\n"
        "5. What I would need before claiming a fine-tuning gain: ___.\n\n"
        "## References\n\n"
        "- Touvron, H., Cord, M., Douze, M., Massa, F., Sablayrolles, A. and Jégou, H. (2021). *Training data-efficient image transformers & distillation through attention.* ICML 2021. [arXiv:2012.12877](https://arxiv.org/abs/2012.12877).\n"
        "- Dosovitskiy, A. et al. (2021). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* ICLR 2021. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929).\n"
        "- Hugging Face checkpoint: [facebook/deit-small-patch16-224](https://huggingface.co/facebook/deit-small-patch16-224) — Apache-2.0; code: [facebookresearch/deit](https://github.com/facebookresearch/deit).\n"
        "- Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images.* Technical report, University of Toronto — the CIFAR-10 dataset.\n"
        "- Sample archive: [Cleanlab/cifar-10-subset](https://huggingface.co/datasets/Cleanlab/cifar-10-subset) — MIT licence.\n"
        "- Repository model card: https://github.com/kurtvalcorza/deit-classification-pipeline/blob/main/MODEL_CARD.md\n"
        "- [`kurtvalcorza/deit-classification-pipeline`](https://github.com/kurtvalcorza/deit-classification-pipeline) — source repository for this pipeline."
    ),
}
