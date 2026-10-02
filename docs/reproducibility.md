# Reproducibility scope

## Source and implementation

This release was reconstructed from recovered development scripts, a locally
modified SMP 0.3.2 package and the manuscript methods. The archived final
experiment version could not be recovered. Model/metric differences are
explained in `model-implementation.md` and `metric-definitions.md`.

## Alignment with the manuscript

| Manuscript method | This release |
| --- | --- |
| Seven classes, ResNet-18 encoder, CaFe, AdaUp and InDis | Components are implemented; the exact final experiment wiring was not recovered. |
| ImageNet initialization, 20 epochs, batch size 16, Adam, cosine learning rate and boundary-loss weight 0.1 | Set as training defaults; full clinical training was not rerun. |
| 448 × 448 center crop without augmentation | Uses the center crop and no augmentation; the loader first resizes to 600 × 448, a step not specified in the manuscript. |
| Five development folds and a fixed participant-exclusive test set | Supported by the manifest format; historical participant assignments and image mappings are unavailable. |
| Segmentation and boundary metrics | Implemented with explicit rules; historical class inclusion and aggregation rules are unverified. |
| Genova evaluation | A separate six-class evaluator is included; historical label mapping, sequence split, training protocol and results are unavailable. |

The numerical results in the manuscript are historical study findings, not
outputs from this release.

## Evidence levels

| Item | Current evidence |
| --- | --- |
| Hand-checkable metrics and manifest checks | Seven unit tests passed locally |
| Synthetic model training and evaluation | Five synthetic folds, inference, mask evaluation and aggregation passed |
| Full 448 x 448 model shape | Batch-1 CPU forward pass verified; logits and boundary shapes checked |
| Full-resolution LSLE training, batch 16 and GPU use | Not verified |
| Original participant/recording-to-image mapping | Not recovered |
| Historical fixed development/test partition | Participant separation cannot be established from recovered directories alone |
| Historical development folds | Recovered loader sliced an image list by index; no patient-grouped fold evidence |
| All five historical FiDi checkpoints and outputs | Not recovered |
| Exact historical F1 and five-model aggregation | Not recovered |
| Reported baselines, ablations and Genova tables | Not reproduced by this release |

A synthetic successful run establishes execution of the tested code path only.
It does not confirm the historical architecture, clinical accuracy, patient
separation or agreement with the paper's tables.

## New clinical runs

Prepare an independently verified pseudonymous manifest and record its origin.
Run all five folds, preserve selected checkpoints and outputs, and reconcile
new tables with those saved reports. Where the protocol differs from historical
records, identify the work as a new experiment. Never infer participant counts
or disease-group split assignments from a desired train/test ratio.

The manuscript's code-availability statement should describe this release as a
reconstructed research implementation unless further evidence establishes its
relationship to the original experiments.

## Execution checks on 2026-09-28

The code-only candidate was prepared from the installed PyPI SMP 0.3.2 package
and tested on macOS with Python 3.9.20 (x86_64 under Rosetta), PyTorch 2.2.2
and torchvision 0.17.2, using CPU.

- Seven unit tests passed.
- All five synthetic folds completed one epoch at 64 x 64, each using four
  training rows, one validation row and the same two fixed-test rows.
- Checkpoint-based inference, per-image evaluation and five-model aggregation
  completed through `run_smoke_test.py --all-folds`.
- Disabling CaFe, AdaUp or InDis separately passed finite-loss forward and backward checks at 64 x 64.
- A batch-1 forward pass at 448 x 448 returned finite logits of shape
  `(1, 7, 448, 448)` and an edge map of shape `(1, 1, 448, 448)`.

Core Python source SHA-256 recorded by that 2026-09-28 workflow:
`2e35a1796a696a55ce4c8bd175c8df0759b6f9a3b6e780f85eaf673d7d6b8024`.
The later GitHub preparation changed a preparation-script docstring only; its
source fingerprint is `4adbee76a02a9d2eb544c6f6a75e24bbebe45cbb35b102f1e45ae214c2212faa`.
The seven unit tests passed again on 2026-09-30. The full model smoke check
was not rerun after this documentation-only source change.

These generated examples and random initializations were not used to estimate
clinical performance. Full-resolution training/backward at batch 16, GPU
execution, LSLE experiments and historical-result agreement remain unverified.
