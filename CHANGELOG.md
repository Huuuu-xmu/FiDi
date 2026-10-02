# Changelog

## Unreleased — GitHub repository preparation

- Made the code directory the repository root and separated local submission
  checks, logs, model weights and draft files from the publishable tree.
- Added a Python 3.9 unit-test workflow and a small test dependency list.
- Clarified setup and the scope of the historical execution checks. Added
  confirmed software authors, copyright holder and the MIT License. No model or
  metric behavior changed in this preparation.

## 0.1.0 — submission candidate

- Reconstructed model, data, training and inference workflows from recovered
  source and manuscript specifications.
- Added explicit seven-label metrics, per-image outputs and five-model summaries.
- Added code/environment identity and checkpoint/reference hashes to reports.
- Added isolated SMP preparation, synthetic examples and one-command smoke checks.
- Documented remaining architecture, metric and historical split uncertainty.
- Preserved null metric values rather than silently dropping models from summaries.

This version has no verified LSLE performance results or historical-result
reproduction claim. Software authors, publication permission and the MIT License
were confirmed after the code candidate was prepared.
