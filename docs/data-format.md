# Data manifest and class labels

The training command requires a CSV manifest with one row per image and these columns:

| Column | Meaning |
| --- | --- |
| `image_id` | Unique pseudonymous image ID used in evaluation reports. |
| `image_path` | Relative or absolute path to an RGB image. |
| `mask_path` | Matching single-channel class-index PNG. |
| `participant_id` | Pseudonymous participant ID, never a name. |
| `recording_id` | Pseudonymous recording ID, unique across participants. |
| `split` | `development` or `test`. |
| `fold` | 1–5 for development images; blank for test images. |

All images and recordings from one participant must stay within one
development/test split and, for development, within one fold. The loader
checks these participant-level partitions before training.
The loader rejects missing IDs, repeated images, mismatched recording
ownership, nonexistent files, and leakage into the fixed test set. For fold
*k*, development rows marked *k* are validation; the other development rows
are training. Test rows are not used for model selection.

Class-index masks use the following labels:

| Value | Label |
| ---: | --- |
| 0 | background |
| 1 | closed-phase glottis (`cglottis`) |
| 2 | closed-phase left vocal fold (`cvfl`) |
| 3 | closed-phase right vocal fold (`cvfr`) |
| 4 | open-phase glottis (`oglottis`) |
| 5 | open-phase left vocal fold (`ovfl`) |
| 6 | open-phase right vocal fold (`ovfr`) |

By default, RGB images and masks are center-cropped directly at native
resolution to 448 × 448. No resizing or padding is applied, and smaller source
images are rejected. RGB channels are normalized using mean and standard
deviation 0.5. The crop preserves the integer mask labels exactly. `--input-size`
selects another crop size when required by the task.
The configuration `configs/fidi.json` uses 448.
The training boundary target is a Canny edge map
of the combined glottis mask (classes 1 and 4). 
