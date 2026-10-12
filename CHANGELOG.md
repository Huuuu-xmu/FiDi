# Changelog

## Documentation update — 2026-10-12

- Remove geometric image examples, their generators and the demonstration training script.
- Rename the pinned CPU dependency file to `requirements-cpu.txt`.
- Update module names and training descriptions to match the current manuscript.
- Clarify manuscript status, software scope and the local test environment.
- Remove obsolete comments and internal preparation notes. Model operations, training defaults and metric formulas are unchanged.

## 0.1.4 — 2026-10-05

- Implement ULSAM-based grouped attention with vectorized grouped convolutions.
- Convert earlier per-group attention checkpoint tensors when loading.
- Update method attribution and third-party notices.

## 0.1.3 — 2026-10-03

- Document inputs, outputs and the Genova class mapping.
- Add Pillow to unit-test dependencies and update the GitHub Actions runner.

## 0.1.2 — 2026-10-03

- Use native-resolution center cropping without intermediate resizing.
- Name pixel-label F1 outputs with a `pixel-` prefix.

## 0.1.1 — 2026-10-03

- Require participant separation across development folds and the fixed test set.
- Check participant separation again when constructing fold partitions.

## 0.1.0

- Assemble model, preprocessing, training and inference commands from available source and method specifications.
- Add pixel-level metrics, per-image outputs and five-model summaries.
- Record software versions and checkpoint/reference hashes in reports.
