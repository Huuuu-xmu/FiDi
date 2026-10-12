# FiDi: fine and disambiguous segmentation

FiDi segments the left vocal fold, right vocal fold and glottis in laryngeal
videostroboscopic images, using separate labels for the open and closed phases.
The model uses an ImageNet-pretrained ResNet-18 encoder and three modules:

- **CaFe — cross-scale gated fine-edge perception:** combines shallow features across scales through gated refinement and predicts an auxiliary glottis boundary map.
- **AdaUp — adaptive upsampling:** reconstructs higher-resolution features using content-aware reassembly following CARAFE.
- **InDis — intra-scale gated organ-structure disambiguation:** applies ULSAM-based grouped spatial attention to deep features. Channel groups have no predefined anatomical assignments.

Related manuscript: *Boundary-Preserving and Phase-Specific Segmentation of the
Vocal Folds and Glottis in Laryngeal Videostroboscopy: Development and Technical
Evaluation of FiDi* (not yet published).

This repository contains the version 0.1.4 software package: model implementation,
preprocessing, training, inference and evaluation. Clinical images, annotations,
participant records, study checkpoints and per-image study predictions are not
included. See [implementation details](docs/model-implementation.md) for the
available source and the scope of this implementation.

## Installation

Use Python 3.9 for the pinned SMP 0.3.2 dependencies. For the locally tested CPU
environment (Python 3.9.20, PyTorch 2.2.2 and torchvision 0.17.2):

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.txt
python scripts/prepare_smp.py
```

For CUDA, install a compatible PyTorch/torchvision pair, then `requirements.txt`.
The manuscript reports experiments with PyTorch 1.9.1, CUDA 11.1 and a single
NVIDIA GeForce RTX 3090. The local CPU environment is for checking this code
package and differs from that study environment. SMP 0.3.2 depends on timm
0.6.12, which did not import successfully under Python 3.12 in our checks.

`prepare_smp.py` checks source hashes and places the FiDi model files in a local
SMP copy under `.local/`. The installed library is unchanged. Dependency versions
are recorded in [tested-environment.txt](docs/tested-environment.txt).

## Data preparation

The LSLE task uses six foreground classes plus background:

| Value | Class |
| ---: | --- |
| 0 | Background |
| 1 | Closed-phase glottis |
| 2 | Closed-phase left vocal fold |
| 3 | Closed-phase right vocal fold |
| 4 | Open-phase glottis |
| 5 | Open-phase left vocal fold |
| 6 | Open-phase right vocal fold |

Use `*_rgb.png` for RGB images and `*_gt.png` for single-channel class-index
masks. Prepare a CSV manifest with image, mask, participant, recording, partition
and fold fields as described in [data format](docs/data-format.md).

Images and masks are center-cropped to 448 × 448 pixels at native resolution,
without resizing, padding or data augmentation. RGB channels are normalized
with mean and standard deviation 0.5. All images and recordings from a participant
must remain in one development/test partition and one development fold.

## Training

```bash
python scripts/train_fold.py --config configs/fidi.json --manifest PATH_TO_MANIFEST.csv --fold 1 --output runs/fold_1
```

Repeat for folds 2–5, keeping the same fixed test set. Defaults are 20 epochs,
batch size 16, Adam, cosine learning-rate decay from 1e-4 to 5e-5, and boundary-loss
weight λ=0.1. Boundary targets are Canny edges of the glottis mask (labels 1 and 4).
ImageNet initialization downloads encoder weights. `--seed` sets the random seed;
the package default is 2026.

Each fold selects the checkpoint with the highest validation mIoU and evaluates
it on the fixed test set. Outputs are `best_state_dict.pth`, `report.json` and
`test_report.json`. To summarize the five test reports:

```bash
python scripts/aggregate_models.py runs/fold_1/test_report.json runs/fold_2/test_report.json runs/fold_3/test_report.json runs/fold_4/test_report.json runs/fold_5/test_report.json --output runs/five_model_summary.json
```

The summary reports the mean and sample SD across five model scores. These
values describe variation across model fits; they are not patient-level
confidence intervals. See [execution and reporting](docs/reproducibility.md).

## Inference and evaluation

```bash
python scripts/predict_masks.py --images IMAGE_DIR --checkpoint CHECKPOINT.pth --output PREDICTION_DIR
python scripts/evaluate_masks.py --truth MASK_DIR --prediction PREDICTION_DIR --crop-reference --output runs/mask_report.json
```

Predictions use `*_pred.png`. Load checkpoints with the same model switches used
during training. `--crop-reference` applies the same center crop to reference
masks; omit it if the references are already aligned. For a different crop size,
pass the same `--input-size` to both commands.

The evaluator computes mIoU, Dice-C, ASSD-C, HD95-C and pixel-label F1 scores.
The `pixel-mF1`, `pixel-F1-G` and `pixel-F1-V` outputs are separate from the
supplementary phase-related F1 results in the manuscript. Metric formulas,
eligibility rules and undefined values are documented in
[metric definitions](docs/metric-definitions.md).

## Ablation and Genova evaluation

`train_fold.py` supports `--disable-cafe`, `--disable-adaup`, `--disable-indis`
and `--boundary-weight` for component ablation and loss-weight sensitivity.

The [Genova evaluator](docs/genova.md) uses a separate six-class mapping and
reports class-specific IoU and Dice with unweighted means. The LSLE training and
inference commands do not provide the Genova training pipeline or checkpoint.

## Tests

```bash
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -v
```

The tests check metric arithmetic, participant separation, crop coordinates and
five-model aggregation. They run in GitHub Actions and require no study data.

## Citation and license

Use [CITATION.cff](CITATION.cff) to cite the software version and commit used.
The manuscript citation will be added when publication details are available.
Software authors are listed in [AUTHORS.md](AUTHORS.md). FiDi contributions use
the [MIT License](LICENSE); upstream notices and method sources are listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
