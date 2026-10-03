# FiDi: laryngeal videostroboscopy segmentation

Associated manuscript: **Boundary-Preserving and Phase-Specific Segmentation of the Vocal Folds and Glottis in Laryngeal Videostroboscopy: Development and Technical Evaluation of FiDi.**

FiDi provides a seven-class segmentation model, training and inference commands,
image-level metrics, and five-model summaries for videostroboscopic research.
The implementation uses a ResNet-18 encoder with boundary supervision,
content-aware upsampling, and grouped attention. Synthetic examples are included
so the workflow can be checked without clinical images.

**Software release:** version 0.1.3. The package provides model training,
checkpoint inference, pixel-level evaluation and participant-grouped data checks.
Training defaults and calculation conventions are documented below and in
[execution and reporting](docs/reproducibility.md).
The FiDi authors release their contributions under the
[MIT License](LICENSE); upstream notices are retained separately.

## Contents

| Path | Purpose |
| --- | --- |
| `fidi/` | Model builder, preprocessing, manifest validation, metrics and provenance |
| `scripts/train_fold.py` | Train a fold and select the best validation-mIoU checkpoint |
| `scripts/predict_masks.py` | Save class-index predictions from a state-dict checkpoint |
| `scripts/evaluate_masks.py` | Compute per-image metrics and their summaries |
| `scripts/aggregate_models.py` | Summarize five models evaluated on the same test set |
| `scripts/run_smoke_test.py` | Synthetic training, inference and evaluation check |
| `configs/fidi.json` | Training defaults for this research implementation |
| `patches/smp-0.3.2/` | FiDi model changes to segmentation_models.pytorch |
| `examples/synthetic/` | Generated RGB/mask examples, with no patient data |
| `docs/` | Model, labels, metric formulas and execution and reporting |

## Installation

The verified Mac CPU environment uses Python 3.9.20 (x86_64 under Rosetta),
PyTorch 2.2.2 and torchvision 0.17.2. It is a development environment, distinct
from the paper's reported GPU/CUDA setup. Native Apple GPU, Windows/CUDA,
and other Python versions have not been validated for this release.
The pinned SMP 0.3.2 dependency uses timm 0.6.12, which fails to import with
Python 3.12 in our check.

Create an isolated Python 3.9 environment and install the tested dependencies:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-smoke.txt
python scripts/prepare_smp.py
```

`prepare_smp.py` verifies the known SMP 0.3.2 source hashes and creates a patched
copy under `.local/`. It does not overwrite the installed library. The library's
MIT notice is retained. The package versions from the tested environment are
recorded in [tested-environment.txt](docs/tested-environment.txt).
For a different platform, install a compatible PyTorch/torchvision pair before
`requirements.txt` and verify execution in that environment.

## Quick execution check

```bash
python scripts/run_smoke_test.py
```

This trains one synthetic fold for one epoch at 64 x 64, saves/reloads its
checkpoint, predicts masks, and evaluates them. To exercise all five folds and
their aggregation:

```bash
python scripts/run_smoke_test.py --all-folds
```

Logs, checkpoints and JSON reports are saved under `runs/smoke_cpu/` and excluded
from Git. The examples deliberately repeat generated images. Their scores have
no clinical interpretation. No encoder weights are downloaded for this check.

## Prepare research data

Use the pseudonymous CSV manifest described in [data format](docs/data-format.md).
Masks must contain integer labels 0-6. Use participant and recording IDs with the intended development/test and fold assignments.

Images and reference masks are center-cropped directly at native resolution
to 448 x 448. No resizing or padding is applied. RGB channels are normalized
using mean and standard deviation 0.5. Images smaller than the crop are rejected.
Development folds follow the explicit CSV assignments. The fixed test set is
excluded from checkpoint selection. The validator requires participant separation
between development and test and requires all development images and recordings
from each participant to remain in the same fold. These checks are mandatory.
The CSV records participant-to-image assignments explicitly.

## Train and summarize

```bash
python scripts/train_fold.py --config configs/fidi.json --manifest PATH_TO_MANIFEST.csv --fold 1 --output runs/fold_1
```

Repeat for folds 2-5. Defaults are ImageNet-pretrained ResNet-18, 20 epochs,
batch size 16, Adam, cosine learning-rate decay from 1e-4 to 5e-5, and boundary
loss weight 0.1. ImageNet initialization requires an encoder-weight download.
The default random seed is 2026 and can be changed with `--seed`.

Each run saves `best_state_dict.pth`, `report.json` and `test_report.json`.
Reports include the fold, seed, per-image scores, code/environment identity,
checkpoint hash, test-image hashes and metric rules. Use identical model and
training configurations for the five-model summary:

```bash
python scripts/aggregate_models.py runs/fold_1/test_report.json runs/fold_2/test_report.json runs/fold_3/test_report.json runs/fold_4/test_report.json runs/fold_5/test_report.json --output runs/five_model_summary.json
```

The summary is the mean and sample SD of five model means. It does not estimate
patient-level uncertainty. A metric missing from any model has a null five-model
mean/SD, with defined/undefined counts retained.

## Inference and metrics

```bash
python scripts/predict_masks.py --images IMAGE_DIR --checkpoint CHECKPOINT.pth --output PREDICTION_DIR
python scripts/evaluate_masks.py --truth MASK_DIR --prediction PREDICTION_DIR --crop-reference --output runs/mask_report.json
```

RGB files use `*_rgb.png`, reference masks `*_gt.png`, and predictions
`*_pred.png`. The default output is the 448 x 448 center crop.
`--crop-reference` applies the same native-resolution crop to reference masks; omit it when
references are already aligned. For synthetic-size or other inputs, supply the
same `--input-size` to both commands. Checkpoints must be
trusted state-dict files created for the same model switches.

Pixel segmentation formulas, class membership, empty-mask handling and averaging rules
are specified in [metric definitions](docs/metric-definitions.md). Pixel F1 outputs are named
`pixel-mF1`, `pixel-F1-G` and `pixel-F1-V`. They measure pixel-label agreement;
the paper's F1 scores are supplementary measures of phase-related confusion.
These pixel metrics and the reported supplementary study F1 scores have separate scopes. Independent image-phase classification is outside this evaluator.

## Ablations and additional evaluation

```bash
python scripts/train_fold.py --manifest PATH_TO_MANIFEST.csv --fold 1 --output runs/no_adaup --disable-adaup
python scripts/train_fold.py --manifest PATH_TO_MANIFEST.csv --fold 1 --output runs/lambda_02 --boundary-weight 0.2
```

`--disable-cafe` and `--disable-indis` are also supported. See
[model implementation](docs/model-implementation.md) for their precise behavior.
The optional [Genova evaluator](docs/genova.md) handles a separate six-class task;
prepare aligned prediction/reference masks using the documented Genova label map.

## Tests

For the lightweight metric, manifest and aggregation tests, install
`requirements-test.txt` in a separate Python 3.9 environment. These tests do
not load the segmentation model. The GitHub Actions workflow runs this suite.

```bash
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -v
```

Tests cover hand-checkable metrics, partition leakage and five-model summaries.
The synthetic smoke command checks the model execution path separately.

## Data, citation and contributions

Clinical images, participant records, trained study checkpoints and per-image study predictions are not included.
Study data access must be arranged with the study team under its approved terms.
The software authors and copyright holder are listed in
[AUTHORS.md](AUTHORS.md), and [CITATION.cff](CITATION.cff) contains the citation
names. The FiDi contributions are under the [MIT License](LICENSE); see
[third-party notices](THIRD_PARTY_NOTICES.md) for the upstream patch and runtime
dependencies. Once released, cite its version and immutable commit or release
identifier alongside the paper. See [contributing](CONTRIBUTING.md) for
updating this research implementation.
