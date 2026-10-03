# Execution and reporting

## Package contents

Version 0.1.3 provides the FiDi model, data loading, training, checkpoint
inference, pixel-level metric evaluation and five-model aggregation. It also
includes a separate six-class Genova mask evaluator and generated examples.

The clinical images, participant-level source records, trained study checkpoints
and per-image study predictions are supplied separately. Published experimental
tables are reported in the article and supplementary results.

## Configuration and records

- Record the image/mask paths, participant IDs, recording IDs, development/test split and development fold in the data manifest.
- Keep each participant within one partition and one development fold.
- Record model switches, random seed, training configuration and package versions for each run.
- The training defaults are ResNet-18 with ImageNet initialization, 20 epochs, batch size 16, Adam, cosine learning-rate decay and boundary-loss weight 0.1.
- Inputs are directly center-cropped to 448 × 448 at native resolution.
- Reports include the configuration, metric conventions, per-image values, checkpoint identity and reference-image identity.
- Preserve the selected checkpoints and evaluation reports with the experiment records.

## Metric scope

The software evaluates pixel-label segmentation. Its F1 outputs carry the
`pixel-` prefix; supplementary phase-related study F1 is reported separately
in the manuscript. Per-image values are averaged with equal weights over
defined cases, and five-model summaries contain a mean and sample SD.
These summaries describe variation across model fits, not a patient-level
confidence interval. Detailed definitions are in `metric-definitions.md`.

## Execution checks

The version 0.1.2 workflow passed eleven unit tests and all five synthetic
training folds, checkpoint inference, mask evaluation and aggregation on
2026-10-03. The tested source SHA-256 was
`85236f07db1745e08e635392f9df4f2e1daf595b77d95de4ebfe042069d7b71c`.
The local CPU environment used Python 3.9.20, PyTorch 2.2.2 and torchvision
0.17.2. The dependency list is in `tested-environment.txt`.

Synthetic examples are installation and execution checks; their scores have
no clinical performance interpretation. For a study analysis, supply the
clinical manifest and task data, train the selected models, and preserve the
generated outputs with the configuration used to obtain them.

## Version 0.1.3 checks on 2026-10-03

- Eleven unit tests passed.
- All five synthetic training folds, inference, mask evaluation and aggregation passed.
- The six-class Genova evaluator returned IoU and Dice of 1.0 for identical synthetic masks containing all six foreground labels.
- Execution source SHA-256: `80a096bd3e2a0d9b1951ae2365cef31c49a9971f000c60af6f87934da7ceb01c`.
- These checks use generated examples for software execution.
