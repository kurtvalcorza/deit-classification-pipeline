# Weight and sample-data provenance and hosting

- Upstream: `facebook/deit-small-patch16-224`
- Revision: **not yet pinned** (`MODEL_REVISION = "unpinned"`). Run `python tools/pin_snapshot.py` to resolve the Hub's `main` to a 40-hex commit, download every manifest-listed file at that commit, cross-check each LFS file against the SHA-256 the Hub records, record the Hub's LFS SHA-256 of the reference file, and write the commit and digests into the manifest and `src/deit_classification_pipeline/pipeline.py`.
- Executed artifact: `pytorch_model.bin` (88,283,631 bytes as the Hub reported for `main` when this repository was built), a PyTorch pickle of the state dict. The Hub repository holds no SafeTensors file, so the loader reads it with `torch.load(..., weights_only=True)`, which restricts unpickling to tensors and plain containers.
- Hosted, not executed: `tf_model.h5` (88,505,432 bytes), the TensorFlow checkpoint. It is listed under `referenceFiles` in the manifest; the pin tool records its Hub LFS SHA-256 without downloading it, so the card can cite the digest of the hosted file and of the executed one.
- Manifest: `weights/deit-small-patch16-224/dimer-base-manifest.json` (4 staged files: `README.md`, `config.json`, `preprocessor_config.json`, `pytorch_model.bin`; `totalBytes` 88360671; plus the reference file). Every `sha256` is `null` until the pin tool runs.
- Committed copy: `preprocessor_config.json` (160 bytes) is committed as the Hub served it, so a test can record that it differs from the transform the pipeline uses. The pin tool replaces it with the bytes downloaded at the pinned commit before hashing.
- Upstream weight license: Apache-2.0 (the checkpoint's `README.md` front matter). The model was trained on ImageNet-1k, whose images carry their own terms of access.
- Hosting: Apache-2.0 permits use, modification, distribution and commercial use, subject to keeping the licence and notices. The Git repository does not vendor the checkpoint (`weights/**/*.bin` is git-ignored).
- Fresh clone, once pinned: `stage_missing_files(allow_download=True)` fetches only the manifest-listed files that are absent, at the pinned revision; `verify_snapshot()` then checks every file before any load. `weights/**` is marked `-text` in `.gitattributes`, so Windows `core.autocrlf` cannot rewrite the committed files and break their digests.
- Loader trust boundary: `ViTConfig.from_pretrained(<verified dir>, local_files_only=True)` and `ViTForImageClassification(config)` build the architecture without contacting the Hub or running repository code; `load_state_dict(..., strict=True)` refuses a missing, unexpected or mis-shaped tensor; and the loader refuses a config whose labels at the zero-shot group indices are not the expected ImageNet classes.

## Tutorial sample data

- Dataset: `Cleanlab/cifar-10-subset`, file `CIFAR-10-subset.zip`, at commit `bb5a7aabf1d14d2d1e3e49d0d8f917bda3622f75`.
- Size and digest: 986,707 bytes, SHA-256 `66f90a4f87d865e8eb653b62f10e754684075a32314177de76832349d4b1fb19`. `fetch_sample_archive` refuses any other bytes and has no fallback.
- Licence: MIT (the dataset card). The images are CIFAR-10 images (Krizhevsky, 2009); the tutorial keeps the `frog` and `truck` folders.
- Hosting: the archive is downloaded at runtime into a working directory (`data/`, git-ignored) and is not redistributed by this repository.
