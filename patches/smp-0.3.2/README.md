# SMP model patch

The two replacement files derive from the study team's modified SMP 0.3.2
source. The decoder activates content-aware upsampling, boundary-feature
modulation and grouped attention as documented in `docs/model-implementation.md`.

`manifest.json` records SHA-256 hashes for the supported SMP source files and
the FiDi replacement files. `scripts/prepare_smp.py` verifies the hashes, copies
the installed SMP package to `.local/`, and replaces the two model files in
that local copy.

The upstream MIT license is preserved in `LICENSE.upstream`. The study team's
later modifications are covered by the repository's root `LICENSE`.
