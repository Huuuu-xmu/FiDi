# Changelog

## 0.1.3 — documentation and reporting, 2026-10-03

- Include Pillow in unit-test dependencies for native-crop tests.
- Use Node 24 GitHub Actions and a fixed Ubuntu 24.04 test runner.

- Present ordinary function, input and output documentation.
- Add the original Genova dataset source and provider label mapping.
- Use a completed-evaluation status for the Genova mask evaluator.
- Retain explicit pixel F1 names and all metric calculations.
- Leave model architecture, preprocessing, training and numerical evaluation behavior unchanged.


## 0.1.2 — manuscript protocol alignment, 2026-10-03

- Center-crop at native resolution; remove the intermediate 600 x 448 resize. Reject undersized inputs rather than resize or pad them.
- Distinguish pixel-label F1 outputs from the paper's supplementary phase-related F1 by explicit `pixel-` names. No image-phase prediction rule is inferred.
- Use the ordinary research software release name and synchronize the manuscript title. Retain verification scope and participant-group protections.


## 0.1.1 — final submission package, 2026-10-03

- Require participant separation across development folds as well as the fixed test split. Reject configurations that disable this protection.
- Add a defensive participant-leakage check to fold partitioning.
- Update manuscript metadata and document the distinction between historical results and the implementation metric rules.
- Preserve the model architecture, preprocessing, metric formulas and historical manuscript numbers.


## Unreleased — GitHub repository preparation

- Made the code directory the repository root and separated local submission
  checks, logs, model weights and draft files from the publishable tree.
- Added a Python 3.9 unit-test workflow and a small test dependency list.
- Clarified setup and the scope of the historical execution checks. Added
  confirmed software authors, copyright holder and the MIT License. No model or
  metric behavior changed in this preparation.

## 0.1.0 — submission candidate

- Prepared model, data, training and inference workflows from recovered
  source and manuscript specifications.
- Added explicit seven-label metrics, per-image outputs and five-model summaries.
- Added code/environment identity and checkpoint/reference hashes to reports.
- Added isolated SMP preparation, synthetic examples and one-command smoke checks.
- Documented model settings, metric calculations and data manifest requirements.
- Preserved null metric values rather than silently dropping models from summaries.

Software authors, publication permission and the MIT License were confirmed after the code candidate was prepared.
