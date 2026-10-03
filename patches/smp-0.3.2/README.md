# SMP model patch

The two replacement files derive from the study team's modified SMP 0.3.2
source. The decoder activates content-aware upsampling, boundary-feature
modulation and grouped attention as documented in `docs/model-implementation.md`.

`manifest.json` records hashes of the recovered baseline, PyPI SMP 0.3.2 source,
recovered modifications and current derivative. The recovered baseline
`base/model.py` already returned an auxiliary gate, unlike PyPI. Both recognized
starting files are replaced by the same derivative after verification.
`scripts/prepare_smp.py` copies the installed package to `.local/` before
replacing the model files.

The upstream MIT license is preserved in `LICENSE.upstream`. The study team's
later modifications are covered by the repository's root `LICENSE`.
