---
license: apache-2.0
model_card_spec: "1.2"
pipeline_tag: image-classification
base_model: facebook/deit-small-patch16-224
date_published: "2020-12"
date_published_source: "month of the DeiT paper and first code release (arXiv:2012.12877, submitted 2020-12-23, with the weights in facebookresearch/deit); the date the Hugging Face conversion was first published is not established by this repository"
---

# DeiT-Small (patch 16, 224 px, ImageNet-1k) — Image Classification with Bounded Fine-Tuning

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-facebook%2Fdeit--small--patch16--224-ffcc4d?style=flat)](https://huggingface.co/facebook/deit-small-patch16-224)
[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-facebookresearch%2Fdeit-181717?style=flat&logo=github&logoColor=white)](https://github.com/facebookresearch/deit)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2012.12877-b31b1b.svg)](https://arxiv.org/abs/2012.12877)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are redistributed unmodified under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebooks are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, or safety-critical use. Running the notebooks downloads third-party weights and datasets governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any damages arising from their use. Hosting implies no affiliation with or endorsement by the original authors.

> [!IMPORTANT]
> The upstream snapshot is pinned to Hub commit `164deee347853469b97442b3817f22eece80c7e3`, and the manifest records every file's SHA-256. No execution with the pinned weights has been recorded yet, so this card claims no measured value for this repository.

---

## Interactive Colab Tutorials

- **End-to-end classification and adaptation tutorial**:
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/deit-classification-pipeline/blob/main/tutorials/deit_classification_colab.ipynb) [`deit_classification_colab.ipynb`](https://github.com/kurtvalcorza/deit-classification-pipeline/blob/main/tutorials/deit_classification_colab.ipynb)
  *A pinned CIFAR-10 frog/truck subset, ImageNet top-5 predictions, majority-class and zero-shot ImageNet baselines, degenerate-input probes, a bounded fine-tune of a new two-class head, held-out and unseen-split scores, and SafeTensors adapter export and reload.*

---

#### Description

`facebook/deit-small-patch16-224` is DeiT-Small, a Vision Transformer (ViT) trained on ImageNet-1k with the data-efficient recipe of "Training data-efficient image transformers & distillation through attention" (Touvron et al., arXiv:2012.12877). This is the non-distilled model. The upstream card states that the weights were converted from the `timm` repository and that the model was trained on a single 8-GPU node for 3 days, and it reports about 22M parameters. A test in this repository counts 22,050,664 parameters in the architecture it builds.

The model cuts a 224×224 image into 196 patches of 16×16 pixels, embeds each patch linearly, prepends a `[CLS]` token and adds learned position embeddings. Twelve transformer layers of width 384 with 6 attention heads process the 197 tokens. A linear head on the final `[CLS]` token outputs a softmax over the 1000 ImageNet-1k classes. The snapshot's `config.json` declares `ViTForImageClassification`, which is how the pipeline loads it.

Preprocessing is a documented discrepancy. The snapshot's `preprocessor_config.json` specifies a plain resize to 224×224 with mean and standard deviation 0.5, and the `transformers` DeiT image processor also defaults to 0.5. The DeiT training code (`datasets.py`, `build_transform`, at the commit the upstream card links) resizes the shorter side to 256 px with bicubic interpolation, centre-crops 224 and normalises with the ImageNet mean (0.485, 0.456, 0.406) and standard deviation (0.229, 0.224, 0.225); the upstream card describes the same. The pipeline applies that transform. The effect of the difference on accuracy has not been measured here.

This repository adds gradient fine-tuning on a caller's labelled images. `from_pretrained(class_names=...)` replaces the 384-to-1000 `classifier` layer with a new layer for the caller's classes. `finetune` then trains with cross-entropy, either the whole network (the default) or the new head alone with the transformer frozen.

What this repository adds to the upstream weights:

- `verify_snapshot` and `stage_missing_files`: manifest checks and staging of the pinned files, both refusing to run if `MODEL_REVISION` is ever reset to the `"unpinned"` sentinel;
- `DeiTPipeline.from_pretrained`: construction of `ViTForImageClassification` from the verified `config.json`, then `torch.load(..., weights_only=True)` and `load_state_dict(strict=True)` from the verified `pytorch_model.bin`, and a check that the config's labels at the zero-shot group indices are the expected ImageNet classes;
- `predict`: input checks and top-k softmax scores; `zero_shot_evaluate`: a baseline that maps groups of ImageNet classes onto task labels without training;
- `fetch_sample_archive`, `read_class_archive`, `read_class_folder`, `validate_dataset`, `split_dataset`, `validate_inputs` and `evaluation_report`: the data acquisition, validation and single-batch evaluation stages;
- `finetune`, `evaluate` (accuracy, balanced accuracy, per-class recall, confusion matrix, majority-class baseline), `save_artifact`, `apply_artifact` and `load_artifact`: the bounded adaptation workflow and its SafeTensors adapter;
- `tools/pin_snapshot.py`, `tools/build_notebook.py` and `tools/validate_release_assets.py`: pinning, notebook generation and static release checks.

#### Intended Use and Limitations

The uses below are the ones the package was built to support. Everything else is out of scope (Out-of-scope use cases) or prohibited (Use cases).

###### Primary Intended Uses

The task is closed-set, single-label image classification. `predict` takes one image or a batch of up to 64 and returns, per image, the top-k classes with their softmax scores and the argmax label.

The pretrained head fits photographs of the objects, animals and scenes ImageNet-1k names. The adaptation path fits a small labelled set of a few classes, for example product categories, plant or specimen types, or document page types, where a new head on a pretrained backbone is enough.

The intended role is a reference Vision Transformer classifier and a teaching baseline for transfer learning. DeiT-Small is small enough to fine-tune on a free GPU in minutes. A reader's own application can embed `DeiTPipeline` as a classifier or, with the head removed, as a `[CLS]`-token image feature extractor.

###### Primary Intended Users

Intended users are machine-learning engineers, computer-vision researchers, students and instructors. Settings envisioned are research prototypes, teaching, and self-hosted applications that run the code in this repository.

A user is expected to know the following before relying on the output:

- the pretrained vocabulary is the 1000 ImageNet-1k classes; anything else receives the nearest of them;
- a `score` is a softmax value under the model's training distribution, not a calibrated probability on the user's images;
- the classifier has no reject option: every image, including a blank one, receives a class;
- accuracy is only meaningful next to the majority-class baseline of the same data, and balanced accuracy is the fairer summary when classes are imbalanced;
- sketches, screenshots, medical, aerial and thermal images, and very small images upsampled to 224 px, are distribution shifts from ImageNet photographs;
- a fine-tune on a few hundred images demonstrates the workflow and does not produce a deployable classifier.

###### Out-of-scope use cases

1. **Capability boundary:** no classes outside the loaded vocabulary, no text prompts, no multi-label output, no localisation (detection or segmentation), and no calibrated confidence or open-set rejection.
2. **Input boundary:** `predict` and `validate_inputs` reject anything that is not a `PIL.Image.Image` (`TypeError`), batches above `MAX_BATCH = 64`, and image sides below `MIN_IMAGE_SIDE = 8` px or above `MAX_IMAGE_SIDE = 4096` px (`ValueError`). `top_k` must be between 1 and the number of classes.
3. **Input boundary:** every image is resized so its shorter side is 256 px and centre-cropped to 224×224, and the position embeddings are fixed for that size. Content outside the central crop is not seen, and small images are upsampled. The tutorial's 32×32 CIFAR-10 images are an example of the latter, not a recommended input size.
4. **Data boundary for adaptation:** `validate_dataset` accepts up to 5,000 records (`MAX_RECORDS`) over 2 to 1,000 classes, with at least 2 images per class. The bounded tutorial fine-tune (5 epochs, no augmentation) is not a training recipe for a production classifier.
5. **Decision boundary:** not for decisions that act on labels without a person reviewing them, including medical, safety, legal, hiring, credit or law-enforcement decisions. Any use also requires accuracy and per-class recall measured locally on the deployment's own labelled images.

#### Factors

###### Groups

ImageNet-1k names only three person classes (`ballplayer`, `groom` and `scuba diver`), but many of its images show people, and classes such as clothing, sports equipment and musical instruments correlate with them. The pretrained model can therefore produce labels whose errors differ with the skin tone, age, gender presentation or dress of the people in an image; this was not evaluated here or, as far as this repository knows, upstream.

ImageNet's photographs were collected from web search in a limited set of languages and regions. Objects, foods, animals and scenes from under-represented regions may be recognised less often; that is unmeasured.

An adapted model inherits whatever group structure the caller's labelled data has. The operator who classifies images of people, with the pretrained or an adapted model, owns a per-group audit on their own images before relying on the output.

###### Instrumentation

ImageNet-1k images are web photographs of varied resolution, mostly taken with consumer cameras. Inference images arrive from whatever produced them: phones, scanners, microscopes, drones or rendering engines.

Resolution, blur, compression, exposure, colour balance and framing all change the evidence. The resize-and-crop preprocessing discards the border of every non-square image and resamples every image to 224 px. The pipeline checks only type and size; it cannot detect a rotated scan, a thumbnail, a rendered image or an empty frame.

The tutorial sample is itself an instrument: CIFAR-10 images are 32×32 pixels, collected by Krizhevsky and labelled by hand, and are upsampled about sevenfold here. Labels a caller supplies carry whatever error their annotation process has, and a fine-tune learns a systematic labelling error as if it were correct.

###### Environment

**Operating environment.** Python 3.12 with the pins in `pyproject.toml`: `torch==2.14.0`, `torchvision==0.29.0`, `transformers==4.57.6`, `safetensors==0.8.0`, `numpy==2.5.3`, `pillow==11.3.0`, `huggingface-hub==0.36.2`. Computation is float32. The code runs on CPU and uses CUDA automatically when available. No run with the pinned weights has been recorded yet, so no runtime, memory or throughput figure is given.

**Data environment.** The pretrained head assumes a photograph centred on one ImageNet object or scene. An adapted model assumes inference images that resemble its training images in source, framing and resolution. The tutorial's adaptation data is CIFAR-10 thumbnails, so a model adapted on it transfers to images of that kind and to little else. When these assumptions fail, the model still returns a class. The pipeline reports no signal that the distribution has shifted.

#### Metrics

###### Performance Measures

`evaluate(records, majority=...)` reports, for this pipeline's head on labelled records:

- `accuracy`: the fraction of images whose argmax label is correct;
- `balanced_accuracy`: the mean of the per-class recalls, which a model cannot raise by favouring the larger class;
- `per_class_recall` and `confusion_matrix` (rows true, columns predicted), which show where the errors go;
- `majority_baseline_accuracy`: the accuracy of always answering `majority`, which the caller sets to the training split's most frequent class.

`zero_shot_evaluate(records, groups)` scores the unmodified ImageNet head on task labels. It assigns each image to the label whose group of ImageNet classes holds the most softmax mass, and reports the same measures. For the tutorial, `frog` maps to ImageNet's three frog classes and `truck` to its big-truck classes; pickups are left out because CIFAR-10's `truck` excludes them.

`evaluation_report(result, truth, groups=...)` covers one batch of ImageNet-head predictions. It reports how often the true group appears in the top-1 and the top-k, with the verdict `sample-sanity`. Without labels it returns `not-measurable` and names the labelled data that would be needed.

The upstream card reports ImageNet top-1 accuracy 79.9 and top-5 accuracy 95.0 for DeiT-Small. Those values are upstream-reported, and this repository does not reproduce them. No value from this repository has been recorded yet.

###### Decision thresholds

There is no score threshold. The reported label is the argmax over the head's classes, so every image receives exactly one of them. `top_k` changes how many ranked classes are returned, not which one is reported.

No acceptance threshold on accuracy is set anywhere in the repository. A deployment that needs to abstain must add its own rule, for example a minimum top-1 score chosen on labelled images from the deployment, and must measure what that rule costs in coverage. Re-check any such rule after a change of camera, image source, class set or adapter.

###### Approaches to uncertainty and variability

Every score is one pass over one split: no repeated runs, no cross-validation, no bootstrap and no confidence interval. With the default 200 images per class, the tutorial's held-out split has 60 images, so one image moves accuracy by about 1.7 percentage points, and differences of a few points between methods are within noise.

Sources of run-to-run variability:

- the per-class sample drawn from the archive and the split, controlled by `DATASET_SEED` and `SEED`;
- the head initialisation and the batch order, controlled by the `seed` argument;
- none from dropout or normalisation statistics: the snapshot's config sets every dropout probability to 0.0, and ViT uses LayerNorm;
- GPU kernel selection, which is not forced to be deterministic, so repeated GPU runs can differ slightly.

A `score` is a softmax output, not a calibrated probability. A caller who needs calibrated confidence must fit a calibration map on labelled images from the deployment. A caller who needs an uncertainty estimate must evaluate on more images, with repeated runs or bootstrap resampling.

#### Ethical considerations and biases

No external ethics board, red team, or population-specific review has examined this repository or, to our knowledge, the upstream checkpoint. Nothing below implies that one did.

###### Data

The upstream card states that the model was trained on ImageNet-1k, which it describes as about 1 million images in 1,000 classes. ImageNet's images were gathered by web search and labelled by crowd workers; they include identifiable people and private settings, and published audits of the wider ImageNet collection have documented offensive and stereotyped labels in its person categories. Personal data is present in the training corpus by construction; it was not audited here.

The tutorial downloads `CIFAR-10-subset.zip` from the `Cleanlab/cifar-10-subset` dataset at commit `bb5a7aabf1d14d2d1e3e49d0d8f917bda3622f75` (MIT licence, 986,707 bytes, SHA-256 `66f90a4f87d865e8eb653b62f10e754684075a32314177de76832349d4b1fb19`). It keeps the `frog` and `truck` folders. CIFAR-10 is a set of 32×32 colour images labelled with ten object classes.

This repository distributes code, tests, documentation and one configuration file copied from the upstream snapshot (`preprocessor_config.json`, kept so a test can record the preprocessing discrepancy). It does not distribute `pytorch_model.bin` or the sample archive; both are downloaded at runtime and git-ignored.

An exported adapter contains weights fitted to the caller's training images. It does not contain the images, but it can reflect them. The operator must audit the images they classify, or fine-tune on, for personal, proprietary or restricted content; the pipeline performs no such check.

###### Human Life

The pipeline is not intended for decisions in health, safety, criminal justice, employment, credit or housing. Neither this repository, the upstream authors, nor any regulator has validated or certified it for any of them.

Some sensitive uses are foreseeable although not intended: triage of medical or dermatology photographs, sorting of images of people, content moderation, and quality inspection that stops a production line. Any of them would be admissible only with human review of every acted-on label. They would also need locally measured accuracy and per-class recall stratified by the groups named above, a documented abstention and re-validation policy, and any regulatory clearance the domain requires.

###### Mitigations

- **Supply-chain integrity:** if `MODEL_REVISION` were reset to `"unpinned"`, `verify_snapshot`, `stage_missing_files` and `from_pretrained` would raise before any download or model import. At the pinned revision, `stage_missing_files` refuses a manifest whose `modelId` or `revision` differs from the package constants. It fetches only manifest-listed files, and only with `allow_download=True`. `verify_snapshot` checks every file's byte size and SHA-256 and refuses an entry with no recorded digest. `from_pretrained` builds the architecture from the verified `config.json` and reads the verified `pytorch_model.bin` with `torch.load(..., weights_only=True)`, which refuses arbitrary pickled objects, then loads it with `strict=True`. The Hub repository holds no SafeTensors file. Its `tf_model.h5` is never staged or loaded; the pin tool records its Hub LFS digest for provenance only.
- **Data integrity:** `fetch_sample_archive` downloads the sample at a fixed dataset commit and checks its size and SHA-256 before it is opened, with no fallback. `read_class_archive` refuses absolute member names and `..` segments and bounds the member count and the uncompressed size before decompressing anything; `read_class_folder` refuses files that link outside the directory.
- **Tests of those refusals:** tests assert that an unpinned package, a missing snapshot and a tampered digest are all refused before `torch`, `transformers` or `safetensors` is imported. Others assert that a full-size checkpoint strict-loads and that a missing tensor or a drifted config is refused, that the label check refuses a shifted ImageNet order, that the pipeline's transform is DeiT's and not the committed `preprocessor_config.json`, and that a frozen fine-tune leaves every transformer weight unchanged.
- **Input integrity:** `validate_inputs` and `predict` share one checker for type, batch size, image size and `top_k`. `validate_dataset` rejects a record with missing keys, a non-image, an out-of-range image, an unknown label or a class with fewer than 2 images. It reports class imbalance and pixel-identical duplicates as findings.
- **Adapter integrity:** `save_artifact` writes SafeTensors, not pickle, with the base identity, base-weights digest, class names and frozen prefixes in its header. `apply_artifact` refuses a different format, base identity or model key. It also refuses a different vocabulary, a different base digest, tensors the model does not have, and any missing trainable tensor.
- **Reproducibility:** exact `==` pins in `pyproject.toml`, carried into the notebook and checked by the parity tests. Seeds for sampling, split, head initialisation and batch order. Every result and adapter records `model_id` and `model_revision`.
- **Refusals:** no download without the explicit flag, no unrestricted pickle deserialisation, no remote model code, and no export of an unadapted pipeline.
- **Statistical mitigations:** none is implemented. There is no class balancing, re-sampling or augmentation; `validate_dataset` reports imbalance and does not change it.

###### Risks and harms

- **A class for every image.** Blank, corrupted and out-of-domain images receive a label, often with a high score. Whoever acts on that label bears the harm; the tutorial probes a blank and a noise image and records what it finds.
- **Mislabelling within a closed vocabulary.** An object outside the vocabulary that resembles a class is labelled as that class, and nothing flags it.
- **Unequal error across groups.** Errors on images of people, and on objects from under-represented regions, may be more frequent; the people in the images bear that harm. It is unmeasured.
- **Overfitting in adaptation.** A fine-tune on a few hundred images can score well on a held-out split from the same source and fail on anything else. The operator who deploys it bears the harm whenever training and deployment images differ.
- **Misleading accuracy.** Accuracy on an imbalanced set can exceed the majority baseline by little while looking high. Reporting it without the baseline and balanced accuracy overstates quality.
- **Automation bias.** High softmax scores invite trust that an uncalibrated score has not earned. Operators who skip review turn a model error into a decision error.
- **Leakage through adaptation data.** A random split of records that share a source photograph or session puts near-duplicates on both sides. The resulting held-out score overstates quality; `validate_dataset` reports exact duplicates, and `split_dataset` documents that grouped data must be split by group.

###### Use cases

The following uses are prohibited even where the model would work:

- classifying people in order to surveil, track, profile or score them, or to infer sensitive attributes;
- unlawful discrimination in employment, housing, credit, insurance, education, healthcare access or law enforcement;
- processing images the operator has no right to process, or in breach of consent, privacy or data-protection obligations;
- deceptive uses that present labels as verified facts or as evidence;
- autonomous physical control or safety interlocks driven by unreviewed labels;
- any use that violates the upstream Apache-2.0 licence, the MIT licence of the sample archive, or the terms of the deployment running the pipeline.

## Immutable provenance

- Model: `facebook/deit-small-patch16-224`
- Revision: `164deee347853469b97442b3817f22eece80c7e3` (pinned 2026-09-25 by `python tools/pin_snapshot.py`, which resolved the Hub's `main` to this commit, downloaded every manifest file at it, recorded each file's SHA-256, and recorded the Hub's LFS SHA-256 of `tf_model.h5` without downloading it).
- Snapshot manifest: `weights/deit-small-patch16-224/dimer-base-manifest.json`, 4 staged files, `totalBytes` 88360671, plus one reference file. The byte sizes and digests describe the files at the pinned commit.
- `pytorch_model.bin` (executed artifact): 88,283,631 bytes; SHA-256 `7eadf70de01e439569e1bbba8b5ef233b4bf70f25a98d9b33460095672bce38c` (matches the Hub's LFS record).
- `tf_model.h5` (hosted TensorFlow checkpoint, reference only): 88,505,432 bytes; never staged or loaded; Hub LFS SHA-256 `43160787f644b779af11e010a117a1b074edfb1bc7cfc1454d2a24f6dc7a1051`.
- `config.json`: 69,593 bytes; `ViTForImageClassification`, hidden size 384, 12 layers, 6 heads, patch 16, image size 224, 1000 labels.
- `preprocessor_config.json`: 160 bytes; plain 224 px resize, mean and standard deviation 0.5 (not used; see Description).
- `README.md`: 7,287 bytes; the upstream model card.
- Loader: `ViTForImageClassification(ViTConfig.from_pretrained(<verified dir>, local_files_only=True))`, then `load_state_dict(torch.load(<verified pytorch_model.bin>, map_location="cpu", weights_only=True), strict=True)`.
- Sample dataset: `Cleanlab/cifar-10-subset` at commit `bb5a7aabf1d14d2d1e3e49d0d8f917bda3622f75`, `CIFAR-10-subset.zip`, 986,707 bytes, SHA-256 `66f90a4f87d865e8eb653b62f10e754684075a32314177de76832349d4b1fb19`.

## Input/output contract

- `DeiTPipeline.from_pretrained(device=None, weights_dir=None, allow_download=False, class_names=None, seed=20260925)`: stage (only with `allow_download=True`), verify, load; with `class_names`, replace the head deterministically under `seed`.
- `predict(images, top_k=None) -> dict`: keys `predictions` (per image: `predicted_label`, `predicted_index`, `top_k` as a list of `{"label", "index", "score"}` in descending score), `top_k`, `decision_rule` (`"argmax"`), `class_names_count`, `adapted`, `device`, `model_id`, `model_revision`. `top_k` defaults to 5, or to the number of classes if smaller.
- `zero_shot_evaluate(records, groups=IMAGENET_GROUPS, *, majority=None) -> dict`: the measures of `evaluate`, plus `rule`, `groups` and `mean_group_mass`; ImageNet head only.
- `finetune(records, *, epochs=5, batch_size=16, learning_rate=1e-4, weight_decay=0.01, seed=20260925, freeze_backbone=False, progress=None) -> dict`: AdamW, cross-entropy, float32, no augmentation; returns the configuration, parameter counts and per-epoch losses.
- `evaluate(records, *, majority=None) -> dict`: `accuracy`, `balanced_accuracy`, `per_class_recall`, `support`, `confusion_matrix`, `n`, `mean_top1_score`, `majority_baseline_accuracy` (with `majority`), `estimation`.
- `save_artifact(path, *, notes=None) -> dict`; `read_artifact_metadata(path) -> dict`; `apply_artifact(path)`; `load_artifact(path, *, weights_dir=None, device=None)`. The adapter format is `deit-adapter-v1`.
- Records: `{"id": str, "image": PIL.Image.Image, "label": str}`; `read_class_archive(zip)` and `read_class_folder(directory)` read them from `<class>/<image>` layouts.
- Constants: `MIN_IMAGE_SIDE = 8`, `MAX_IMAGE_SIDE = 4096`, `MAX_BATCH = 64`, `NUM_IMAGENET_CLASSES = 1000`, `DEFAULT_TOP_K = 5`, `IMAGENET_GROUPS` (frog: 30, 31, 32; truck: 555, 569, 675, 864, 867), `MAX_RECORDS = 5000`, `MIN_PER_CLASS = 2`.

## Verification records

No execution with the pinned weights has been recorded. The offline test suite runs a two-layer `ViTForImageClassification` with random weights on 32 px inputs through prediction, the zero-shot baseline, full and frozen fine-tuning, evaluation and adapter reload; that exercises the code path and is not a result about this model. `docs/release-verification.md` holds the release gate and the record table.

## References

- Touvron, Cord, Douze, Massa, Sablayrolles and Jégou. Training data-efficient image transformers & distillation through attention. ICML 2021. https://arxiv.org/abs/2012.12877
- Dosovitskiy et al. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR 2021. https://arxiv.org/abs/2010.11929
- Krizhevsky. Learning Multiple Layers of Features from Tiny Images. Technical report, University of Toronto, 2009.
- Upstream code: https://github.com/facebookresearch/deit (evaluation transform: `datasets.py`, `build_transform`, commit `ab5715372db8c6cad5740714b2216d55aeae052e`)
- Upstream card: https://huggingface.co/facebook/deit-small-patch16-224
- Sample dataset: https://huggingface.co/datasets/Cleanlab/cifar-10-subset
