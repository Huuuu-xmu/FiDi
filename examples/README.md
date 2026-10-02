# Example data

`synthetic/` contains two generated RGB image and seven-label mask pairs. They document the expected `*_rgb.png` and `*_gt.png` naming convention only. They contain no patient data and are not suitable for training or validating FiDi. The images can be regenerated with `python examples/generate_synthetic.py` after installing Pillow.

Mask values are `0` for background, `1` for closed-phase glottis, `2` and `3` for closed-phase left and right vocal folds, `4` for open-phase glottis, and `5` and `6` for open-phase left and right vocal folds. This mapping follows the recovered dataset-conversion description and requires verification against the complete annotation pipeline before release.

`python examples/generate_smoke_data.py` copies the two synthetic pairs into
seven pseudonymous rows with five development folds and two fixed-test rows
(one open phase and one closed phase). The
generated `smoke_data/` directory is ignored by Git. Duplicate image content
is intentional for an installation check and carries no performance meaning.
