# Data manifest and class labels

The reconstructed training command requires a CSV manifest. This is a proposed
new format because the original participant-to-image mapping was not recovered.
The CSV must contain one row per image and these columns:

| Column | Meaning |
| --- | --- |
| `image_id` | Unique pseudonymous image ID used in evaluation reports. |
| `image_path` | Relative or absolute path to an RGB image. |
| `mask_path` | Matching single-channel class-index PNG. |
| `participant_id` | Pseudonymous participant ID, never a name. |
| `recording_id` | Pseudonymous recording ID, unique across participants. |
| `split` | `development` or `test`. |
| `fold` | 1–5 for development images; blank for test images. |

All images from one participant must stay within one development/test split.
The manuscript does not specify participant-grouped development folds, and the
recovered loader split an image list by index. Accordingly, the default
manifest validator allows one development participant in multiple folds.
`--group-folds-by-participant` adds a stricter rule for a newly designed run,
but that run must be described as different from the recovered fold method.
The loader rejects missing IDs, repeated images, mismatched recording
ownership, nonexistent files, and leakage into the fixed test set. For fold
*k*, development rows marked *k* are validation; the other development rows
are training. Test rows are not used for model selection.

Labels recovered from `json_to_dataset.py` are:

| Value | Label |
| ---: | --- |
| 0 | background |
| 1 | closed-phase glottis (`cglottis`) |
| 2 | closed-phase left vocal fold (`cvfl`) |
| 3 | closed-phase right vocal fold (`cvfr`) |
| 4 | open-phase glottis (`oglottis`) |
| 5 | open-phase left vocal fold (`ovfl`) |
| 6 | open-phase right vocal fold (`ovfr`) |

By default, the image loader resizes RGB and mask images to 600 × 448, then
crops the central 448 × 448 square. RGB channels are normalized using mean and
standard deviation 0.5. Masks use nearest-neighbor interpolation and retain
integer class values. The `--input-size` and `--resize-width` options are for
synthetic execution checks or separately specified imaging protocols. The
reconstruction configuration uses 448 and 600.
The manuscript states a 448 × 448 center crop but does not specify this
intermediate resize. Its use in the historical experiments needs confirmation
from original records before a result can be called a reproduction.
The training boundary target follows the manuscript wording: a Canny edge map
of the combined glottis mask (classes 1 and 4). The recovered development
script instead restricted edges to class 1; this change requires retraining.
