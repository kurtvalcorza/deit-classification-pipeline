"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.1 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded package, and the
model pin/stage/verify cells are produced by the generator from repository sources so they cannot
drift from the package.

This is an `E2E` template, so it must state `run_all` itself, and its default path really adapts:
NOTEBOOK_SPEC 2.1 RUN7/FT2 make a bounded fine-tune mandatory rather than optional for this profile.
Every value a reader can change is a `# @param` form field, and each file-reading BYOD branch has a
location field that bypasses the upload dialog when set (EXE1, EXE2).
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "deit_classification_pipeline",
    "repo_name": "deit-classification-pipeline",
    "stem": "deit_classification",
    "notebook_name": "deit_classification_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
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
        "**The default path really adapts the model:** it downloads a pinned 400-image CIFAR-10 subset (`frog` and `truck`), measures two "
        "baselines on a held-out split — always answering the training majority, and mapping the ImageNet head's frog and truck classes to the "
        "two labels without any training — then replaces the head with a two-class layer, fine-tunes, scores the result on the same held-out "
        "split, classifies a further unseen split, exports the changed tensors as a SafeTensors adapter, and reloads that adapter onto a fresh "
        "copy of the verified base model to check that it reproduces the same predictions. Every number you see is measured in this notebook "
        "runtime."
    ),
    "learning_objectives": (
        "install the pinned runtime; read what the carried package guarantees; stage and digest-verify the immutable upstream model revision; "
        "download and digest-verify a pinned labelled dataset and validate it before any model runs; read ImageNet top-5 predictions; build a "
        "zero-shot baseline by mapping ImageNet classes onto task labels; see what a closed-set classifier answers for blank and noise images; "
        "split a dataset with stratification; fine-tune with cross-entropy; compare accuracy and balanced accuracy against both baselines on a "
        "held-out split; classify unseen images; and export, reload and verify the adapter."
    ),
    "exclusions": (
        "ImageNet benchmark results (nothing here re-scores the ImageNet validation set); CIFAR-10 benchmark results (the tutorial uses two "
        "classes and a few hundred images); real-world frog or vehicle recognition (CIFAR-10 images are 32×32 thumbnails, upsampled here to "
        "224 px); open-set recognition or a reject option (every image receives one of the known classes); calibrated probabilities; object "
        "detection, segmentation or multi-label tagging."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). A CUDA GPU such as a Colab or Kaggle T4 is the documented runtime for the fine-tuning stages and is used automatically when present; the notebook also runs on CPU, more slowly. Runtimes are not measured in this revision. The pinned `torch==2.14.0` wheel is the largest download; the checkpoint is about 88 MB.",
        "- **Knowledge:** basic Python and PIL; what a softmax over classes is; the idea of an image transformer operating on patches; accuracy, balanced accuracy and a confusion matrix; why a baseline is needed before a score means anything.",
        "- **Data:** the default path downloads one pinned archive, `CIFAR-10-subset.zip` from the Hugging Face dataset `Cleanlab/cifar-10-subset` (MIT licence, 986,707 bytes, verified by SHA-256 before it is opened), and keeps its `frog` and `truck` folders. BYOD is optional and off by default. Expected BYOD input: one image, or a directory (or `.zip`) of class folders, `<class>/<image>`, with at least 2 images per class. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so; uploaded inputs stay in this runtime and are not sent to any inference API.",
    ],
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime installs the pinned dependencies, stages and digest-verifies the pinned checkpoint, "
        "downloads and digest-verifies the pinned CIFAR-10 subset, validates and splits it, measures the majority-class and zero-shot ImageNet "
        "baselines, probes the model with a blank and a noise image, **runs the bounded fine-tune**, evaluates the held-out split, classifies an "
        "unseen split, exports the adapter, reloads it onto a fresh base model to verify the predictions, and writes machine-readable outputs "
        "with provenance. Nothing is skipped behind a default-off flag, and no clone or DIMER worker is required (NOTEBOOK_SPEC 2.1 §5, RUN7, FT2)."
    ),
    "byod": (
        "Two optional BYOD branches are included, and both are off by default (`USE_BYOD_IMAGE = False`, `USE_BYOD_DATASET = False`). "
        "`USE_BYOD_IMAGE` runs your own image through the same validation, ImageNet prediction and evaluation-report stages as the sample. "
        "`USE_BYOD_DATASET` takes your own class folders through the full adaptation workflow — validate, split, baselines, fine-tune, evaluate, "
        "export and reload — under NOTEBOOK_SPEC 2.1 DAT14. Set `BYOD_IMAGE_PATH` or `BYOD_DATASET_PATH` to read from a location without an "
        "upload dialog (EXE2)."
    ),
    "cells": [
        # ---------------------------------------------------------------- 4. Dataset
        {
            "md": (
                "## 4. Download, verify and validate the labelled sample\n\n"
                "`fetch_sample_archive` downloads `CIFAR-10-subset.zip` from the `Cleanlab/cifar-10-subset` dataset **at a pinned commit** and "
                "checks its byte size and SHA-256 before any member is opened. There is no fallback: a mismatch stops the notebook. "
                "`read_class_archive` then reads the `frog` and `truck` class folders. It refuses absolute member paths and `..` segments, and it "
                "bounds the member count and the uncompressed size before decompressing anything.\n\n"
                "`validate_dataset` checks every record before any model runs: record keys, image type and size, and labels. It reports classes "
                "with too few images as errors, and imbalance and pixel-identical duplicates as findings.\n\n"
                "**What to look for:** CIFAR-10 images are 32×32 pixels. The model's preprocessing upsamples them to 224×224, so every image the "
                "network sees here is a blurred thumbnail, far from the photographs ImageNet was collected from."
            ),
            "code": (
                "import hashlib\n"
                "import json\n"
                "import os\n"
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
                "## 5. Split, and measure two baselines before any training\n\n"
                "The records are split per class with a fixed seed: 70% for training, and the rest halved into a **held-out** split, which scores "
                "every method below, and an **unseen** split, used only in Section 10. No hyperparameter is chosen on either. A random split is "
                "valid here because CIFAR-10 images are independent thumbnails; images that share a photograph or a camera session must be split "
                "by that group instead.\n\n"
                "Two baselines give the fine-tune something to beat:\n\n"
                "- **Majority class:** always answer the most frequent training label. On a balanced split this is 50%, and any useful model must "
                "clear it.\n"
                "- **Zero-shot ImageNet mapping:** the pretrained 1000-class head already has frog and truck classes. `zero_shot_evaluate` sums the "
                "softmax mass over ImageNet's three frog classes and over its big-truck classes (fire engine, garbage truck, moving van, tow truck, "
                "trailer truck; CIFAR-10's `truck` excludes pickups), and answers whichever group is larger. No weight changes.\n\n"
                "The cell also prints ImageNet top-5 predictions for four images and a `sample-sanity` evaluation report: does the true group appear "
                "in the top-1 or the top-5?"
            ),
            "code": (
                'SEED = 0  # @param {{type:"integer"}}\n'
                'TOP_K = 5  # @param {{type:"integer"}}\n\n'
                "train_records, rest = split_dataset(records, train_fraction=0.7, seed=SEED)\n"
                "held_out, unseen = split_dataset(rest, train_fraction=0.5, seed=SEED)\n"
                "ids = [{{r['id'] for r in part}} for part in (train_records, held_out, unseen)]\n"
                "assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]), 'split leaked records'\n"
                "TRAIN_MAJORITY = majority_class(train_records)\n"
                "print({{'train': len(train_records), 'held_out': len(held_out), 'unseen': len(unseen), 'train_majority': TRAIN_MAJORITY}})\n\n"
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
                "## 6. Degenerate input probes: blank canvas and noise\n\n"
                'A closed-set classifier has no "none of these" answer: every image gets one of its classes, with a score that sums to 1 across '
                "them. The cell shows what the ImageNet head answers for a white image and for uniform noise, and how high its top score is.\n\n"
                "**What to look for:** a confident label on structure-free input is a property of softmax classification, not evidence about the "
                "image. The same probe is repeated on the adapted two-class model in Section 10."
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
                "## 7. Replace the head and measure the untrained two-class model\n\n"
                "`from_pretrained(class_names=SAMPLE_CLASSES)` loads the verified checkpoint again and replaces only the classification layer "
                "(`classifier`, 384 features to 1000 classes) with a new 384-to-2 layer initialised under `SEED`. Every other weight keeps its "
                "ImageNet value.\n\n"
                "**Expect roughly chance.** A randomly initialised head carries no information about frogs or trucks. This row shows the floor, and "
                "that the fine-tune — not the re-heading — is what moves the score."
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
                "## 8. Bounded fine-tuning\n\n"
                "This cell runs the real adaptation step in this runtime: gradient fine-tuning with cross-entropy over the two classes.\n\n"
                "- **What trains:** with `FREEZE_BACKBONE = False` (the default) every weight trains. With `True`, only the new head trains and "
                "the transformer (`vit.`) keeps its weights; ViT normalises with LayerNorm, so there are no running statistics to hold. The cell "
                "prints the trainable and total parameter counts.\n"
                "- **Schedule:** `EPOCHS` epochs of AdamW at learning rate `1e-4` with weight decay `0.01`, batch size 16, float32, seed `SEED`, "
                "and no data augmentation: training images go through the same evaluation transform as test images.\n\n"
                "**Read the loss as optimisation evidence only.** A falling training loss says the optimizer is fitting the training images; the "
                "held-out comparison in the next section is the task evidence."
            ),
            "code": (
                'LEARNING_RATE = 1e-4  # @param {{type:"number"}}\n'
                'BATCH_SIZE = 16  # @param {{type:"integer"}}\n'
                'FREEZE_BACKBONE = False  # @param {{type:"boolean"}}\n\n'
                "run = adapter.finetune(\n"
                "    train_records,\n"
                "    epochs=EPOCHS,\n"
                "    batch_size=BATCH_SIZE,\n"
                "    learning_rate=LEARNING_RATE,\n"
                "    seed=SEED,\n"
                "    freeze_backbone=FREEZE_BACKBONE,\n"
                "    progress=lambda row: print(f\"epoch {{row['epoch']}}/{{row['epochs']}}  loss {{row['loss']:.4f}}\"),\n"
                ")\n"
                "print(json.dumps({{key: run[key] for key in ('freeze_backbone', 'trainable_parameters', 'total_parameters', 'epochs',\n"
                "                                         'batch_size', 'learning_rate', 'optimizer', 'augmentation', 'precision', 'device')}}, indent=2))"
            ),
        },
        # ---------------------------------------------------------------- 9. Evaluate held-out
        {
            "md": (
                "## 9. Evaluate on the held-out split\n\n"
                "All four rows are scored on the same held-out images. `accuracy` is the fraction answered correctly; `balanced_accuracy` averages "
                "the per-class recall, so it cannot be inflated by favouring the larger class. The confusion matrix shows which way the errors go. "
                "These are tutorial metrics from one pass over a few dozen thumbnails, with no dispersion estimate: a difference of one or two "
                "images is within noise."
            ),
            "code": (
                "adapted = adapter.evaluate(held_out, majority=TRAIN_MAJORITY)\n"
                "comparison = {{\n"
                "    'majority class': {{'accuracy': adapted['majority_baseline_accuracy'], 'balanced_accuracy': 1 / len(SAMPLE_CLASSES)}},\n"
                "    'zero-shot ImageNet mapping': zero_shot,\n"
                "    'untrained two-class head': untrained,\n"
                "    'fine-tuned': adapted,\n"
                "}}\n"
                "print(f\"{{'method':<28s}} {{'accuracy':>9s}} {{'balanced':>9s}}\")\n"
                "for name, row in comparison.items():\n"
                "    print(f\"{{name:<28s}} {{row['accuracy']:>9.3f}} {{row['balanced_accuracy']:>9.3f}}\")\n"
                "print('confusion (rows true, columns predicted):', adapted['confusion_matrix'])"
            ),
        },
        # ---------------------------------------------------------------- 10. Unseen data
        {
            "md": (
                "## 10. Inference on the unseen split and the degenerate probes again\n\n"
                "The unseen split was never used for training or for any comparison above. The adapted pipeline classifies it, and the cell repeats "
                "the blank and noise probes: the two-class model must now answer `frog` or `truck` for them, whatever they contain."
            ),
            "code": (
                "unseen_metrics = adapter.evaluate(unseen, majority=TRAIN_MAJORITY)\n"
                "print({{k: unseen_metrics[k] for k in ('accuracy', 'balanced_accuracy', 'n')}})\n"
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
                "## 11. Adapter export, fresh reload, and equivalence check\n\n"
                "`save_artifact` writes `outputs/deit_adapter.safetensors`: every tensor the fine-tune could change, plus a metadata header "
                "naming the base model, its pinned revision, the base `model.safetensors` SHA-256, the class names and the frozen prefixes. With the "
                "default full fine-tune that is the whole state dict; with `FREEZE_BACKBONE = True` it is the head only.\n\n"
                "`load_artifact` then builds a **fresh** pipeline from the verified base snapshot, loads the adapter tensors onto it, and refuses an "
                "adapter whose format, base identity, base digest or tensor set does not fit. The cell compares the reloaded predictions with the "
                "in-memory model's on the unseen split, with a stated tolerance: loading succeeding is not the check, reproducing the predictions is."
            ),
            "code": (
                "artifact_path = OUTPUTS / 'deit_adapter.safetensors'\n"
                "descriptor = adapter.save_artifact(artifact_path, notes='DeiT-Small CIFAR-10 frog/truck tutorial adapter')\n"
                "print(json.dumps(descriptor, indent=2))\n\n"
                "reloaded = DeiTPipeline.load_artifact(artifact_path, weights_dir=WEIGHTS_DIR)\n"
                "print({{'reloaded_source': reloaded.source, 'adapted': reloaded.adapted, 'class_names': list(reloaded.class_names)}})\n\n"
                "TOLERANCE = 1e-4\n"
                "check_images = [r['image'] for r in unseen[:MAX_BATCH]]\n"
                "first = adapter.predict(check_images, top_k=2)['predictions']\n"
                "second = reloaded.predict(check_images, top_k=2)['predictions']\n"
                "for a, b in zip(first, second, strict=True):\n"
                "    assert a['predicted_label'] == b['predicted_label']\n"
                "    assert abs(a['top_k'][0]['score'] - b['top_k'][0]['score']) <= TOLERANCE\n"
                "reload_check = {{'images_compared': len(check_images), 'tolerance': TOLERANCE, 'equivalent': True}}\n"
                "print(reload_check)"
            ),
        },
        # ---------------------------------------------------------------- 12. Outputs & provenance
        {
            "md": (
                "## 12. Write machine-readable outputs and provenance\n\n"
                "The cell writes:\n"
                "- `outputs/deit_classification_input_manifest.json`\n"
                "- `outputs/deit_classification_evaluation_report.json`\n"
                "- `outputs/deit_classification_result.json` (identity, runtime versions, device, dataset provenance and manifest, split, "
                "baselines, fine-tuning configuration, held-out and unseen metrics, adapter descriptor and reload check)\n"
                "- `outputs/deit_classification_predictions.csv`\n"
                "- `outputs/deit_adapter.safetensors` (written in Section 11)"
            ),
            "code": (
                "import csv\n\n"
                "with open(OUTPUTS / '{stem}_input_manifest.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(input_manifest, f, indent=2)\n\n"
                "with open(OUTPUTS / '{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(imagenet_report, f, indent=2)\n\n"
                "with open(OUTPUTS / '{stem}_predictions.csv', 'w', newline='', encoding='utf-8') as f:\n"
                "    writer = csv.writer(f)\n"
                "    writer.writerow(['split', 'id', 'truth', 'predicted', 'score'])\n"
                "    for split_name, part in (('held_out', held_out), ('unseen', unseen)):\n"
                "        for start in range(0, len(part), MAX_BATCH):\n"
                "            chunk = part[start:start + MAX_BATCH]\n"
                "            for record, pred in zip(chunk, adapter.predict([r['image'] for r in chunk], top_k=1)['predictions'], strict=True):\n"
                "                writer.writerow([split_name, record['id'], record['label'], pred['predicted_label'], f\"{{pred['top_k'][0]['score']:.4f}}\"])\n\n"
                "result_export = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'torchvision': torchvision.__version__,\n"
                "                'transformers': transformers.__version__, 'cuda': torch.cuda.is_available()}},\n"
                "    'device': pipe.device,\n"
                "    'dataset': {{'archive': archive_info, 'manifest': dataset_manifest, 'per_class': PER_CLASS, 'seed': DATASET_SEED}},\n"
                "    'split': {{'train': len(train_records), 'held_out': len(held_out), 'unseen': len(unseen), 'seed': SEED, 'train_majority': TRAIN_MAJORITY}},\n"
                "    'imagenet_top_k': imagenet_result['predictions'],\n"
                "    'degenerate_probes': {{'imagenet_head': degenerate, 'adapted': adapted_degenerate}},\n"
                "    'baselines': {{'zero_shot': zero_shot, 'untrained_head': untrained}},\n"
                "    'finetune': run,\n"
                "    'held_out': adapted,\n"
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
                "at least 2 images per class, at most 5,000 images. Archive members must stay inside the archive; folder files must not link outside "
                "the directory. The class names are the folder names. The branch runs the same validate → split → baselines → fine-tune → evaluate "
                "→ export → reload stages as the sample; the zero-shot ImageNet baseline needs a class-to-ImageNet mapping, so it is skipped here.\n\n"
                "Set `BYOD_IMAGE_PATH` or `BYOD_DATASET_PATH` to read from a mounted or local location; leave them empty on Colab to get an upload "
                "dialog instead. Uploaded files are written under `outputs/byod/` in this runtime and are not sent anywhere else. The first lines of "
                "the cell show the validator refusing two malformed inputs with messages that name the failed rule."
            ),
            "code": (
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
                "def _upload_into(target):\n"
                "    from google.colab import files  # type: ignore[import-not-found]\n"
                "    target.mkdir(parents=True, exist_ok=True)\n"
                "    for name, data in files.upload().items():\n"
                "        (target / Path(name).name).write_bytes(data)\n"
                "    return target\n\n"
                "if USE_BYOD_IMAGE:\n"
                "    image_path = Path(BYOD_IMAGE_PATH) if BYOD_IMAGE_PATH else next(iter(sorted(_upload_into(BYOD_DIR / 'image').iterdir())))\n"
                "    with Image.open(image_path) as handle:\n"
                "        byod_image = handle.convert('RGB')\n"
                "    print(validate_inputs(byod_image, top_k=TOP_K, names=[image_path.name])['verdict'])\n"
                "    byod_result = pipe.predict(byod_image, top_k=TOP_K)\n"
                "    print([(item['label'].split(',')[0], round(item['score'], 3)) for item in byod_result['predictions'][0]['top_k']])\n"
                "    print(evaluation_report(byod_result, None, sample_kind='byod')['verdict'])\n"
                "else:\n"
                "    print('BYOD image branch is off; set USE_BYOD_IMAGE = True to classify your own image.')\n\n"
                "if USE_BYOD_DATASET:\n"
                "    source = Path(BYOD_DATASET_PATH) if BYOD_DATASET_PATH else next(iter(sorted(_upload_into(BYOD_DIR / 'dataset').iterdir())))\n"
                "    byod_records = read_class_archive(source) if source.suffix.lower() == '.zip' else read_class_folder(source)\n"
                "    byod_names = sorted({{r['label'] for r in byod_records}})\n"
                "    print(json.dumps(validate_dataset(byod_records, byod_names, epochs=EPOCHS), indent=2))\n"
                "    byod_train, byod_held = split_dataset(byod_records, train_fraction=0.7, seed=SEED)\n"
                "    byod_majority = majority_class(byod_train)\n"
                "    byod_pipe = DeiTPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, class_names=byod_names, seed=SEED)\n"
                "    byod_before = byod_pipe.evaluate(byod_held, majority=byod_majority)\n"
                "    byod_pipe.finetune(byod_train, epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, seed=SEED, freeze_backbone=FREEZE_BACKBONE)\n"
                "    byod_after = byod_pipe.evaluate(byod_held, majority=byod_majority)\n"
                "    print({{'majority_baseline': byod_after['majority_baseline_accuracy'], 'untrained': byod_before['balanced_accuracy'], 'fine_tuned': byod_after['balanced_accuracy']}})\n"
                "    byod_descriptor = byod_pipe.save_artifact(OUTPUTS / 'byod_deit_adapter.safetensors', notes='BYOD adaptation adapter')\n"
                "    DeiTPipeline.load_artifact(OUTPUTS / 'byod_deit_adapter.safetensors', weights_dir=WEIGHTS_DIR)\n"
                "    print('BYOD adapter exported and reloaded:', byod_descriptor['sha256'][:16])\n"
                "else:\n"
                "    print('BYOD dataset branch is off; set USE_BYOD_DATASET = True to adapt DeiT-Small on your own class folders.')"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "**What this notebook established, in this runtime.** The pinned `facebook/deit-small-patch16-224` snapshot was verified against a committed "
        "SHA-256 manifest before loading, and the pinned CIFAR-10 subset was verified against its recorded digest before it was read. The "
        "ImageNet head was run on sample thumbnails and on two structure-free probes, and its frog and truck classes gave a zero-shot baseline. "
        "The head was then replaced for the two tutorial classes and fine-tuned, and the result was compared with the majority-class, zero-shot "
        "and untrained baselines on the same held-out split and checked on an unseen split. The adapter was exported, reloaded onto a fresh base "
        "model and checked against the in-memory predictions.\n\n"
        "**What a green run proves.** Successful execution proves that the recorded repository revision, the pinned dependency set, the pinned "
        "checkpoint and the pinned dataset together reproduce these stages in a fresh runtime, without the repository being cloned or installed "
        "and without any DIMER worker or service. It does **not** establish benchmark superiority, fitness for any deployment, or that the "
        "adapted model recognises frogs or trucks outside CIFAR-10's 32×32 thumbnails. The held-out scores come from a few dozen images and carry "
        "no dispersion estimate. The scores are not calibrated probabilities, and there is no reject option.\n\n"
        "**Reproducibility.** Seeds are form fields (`DATASET_SEED`, `SEED`), the run is float32 with no data augmentation, and the new head is "
        "initialised under `SEED`. GPU kernels are not forced to be deterministic, so repeated GPU runs can differ in the last digits of the loss "
        "and the scores.\n\n"
        "**Try next.** Set `FREEZE_BACKBONE = True` and compare the held-out scores and runtime of a head-only fine-tune, or lower `PER_CLASS` "
        "and watch how quickly the fine-tuned model falls back toward the zero-shot baseline. To transfer the workflow, point `BYOD_DATASET_PATH` "
        "at a small set of class folders from your own domain.\n\n"
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
