# Genova segmentation task

The public dataset contains 536 annotated 512 × 512-pixel images in eight
sequences from two patients. Obtain the images and annotations from the
[original provider](https://github.com/imesluh/vocalfolds); the dataset paper is
[Laves et al](https://doi.org/10.1007/s11548-018-01910-0).
The manuscript treats this task as a separate supplementary technical evaluation,
following the setup described for [RANT](https://doi.org/10.1016/j.bspc.2022.103890).

The provider's class-index masks use this mapping:

| Index | Class |
| ---: | --- |
| 0 | void, excluded from the foreground mean |
| 1 | vocal folds |
| 2 | other tissue |
| 3 | glottal space |
| 4 | pathology |
| 5 | surgical tool |
| 6 | intubation |

Prepare aligned `*_gt.png` and `*_pred.png` masks with this mapping, then run:

```bash
python scripts/evaluate_genova.py --truth REFERENCE_DIR --prediction PREDICTION_DIR --output runs/genova_report.json
```

The evaluator pools pixel counts across the supplied images for each foreground
class, computes class-specific IoU and Dice, and reports their unweighted means
across classes 1–6. It records image counts and the calculation conventions.

For checkpoint inference, use a separately trained Genova checkpoint with the
Genova label mapping and `--input-size 512`. `train_fold.py` provides the LSLE
participant-grouped five-fold workflow; Genova training requires its own task
configuration and sequence partition. Input images, masks, training manifests
and checkpoints are supplied separately and are not distributed in this code package.
