# Separate Genova task

The manuscript describes a separate 512 × 512-pixel endoscopic dataset with
six foreground classes and an unweighted mean of six class-specific IoU and
Dice scores. `scripts/evaluate_genova.py` implements that class-wise mean for
class-index masks 0–6, pooling pixels over images before computing each class
score. The manuscript does not specify whether the historical calculation
pooled pixels, so this convention needs confirmation.

The model and prediction command can accept `--input-size 512 --resize-width
512`. A new Genova checkpoint would require training on the same sequence
partition as RANT and a verified mapping from the six semantic classes to
numeric mask values. The current `train_fold.py` still uses LSLE-specific
validation metrics and a five-fold manifest, so it is not a verified Genova
training protocol. Neither the RANT partition manifest nor the local Genova
label map is available in the recovered files. This repository does not claim
that the LSLE checkpoint transfers to Genova or that the reported Genova
numbers can yet be regenerated.
