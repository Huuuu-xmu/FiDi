# Metric definitions and aggregation

These definitions describe version 0.1.0. The original F1 and five-model
aggregation implementation was not recovered, so correspondence with historical
manuscript tables remains unverified. New values must be identified as new runs.

## Classes and overlap counts

For class k in one image, let P_k and T_k be predicted and reference pixel sets.
TP_k = |P_k intersect T_k|, FP_k = |P_k minus T_k| and
FN_k = |T_k minus P_k|. Background (0) is excluded from all foreground macro
scores. Phase and side labels are distinct classes.

| Metric | Formula / calculation | Classes or eligible images |
| --- | --- | --- |
| F1-G | 2 sum(TP_k) / sum(2 TP_k + FP_k + FN_k) | k in {1,4}; per-image micro F1 over distinct phase labels |
| F1-V | Same grouped micro formula | k in {2,3,5,6}; distinct phase and side labels |
| mF1 | Arithmetic mean of 2 TP_k / (2 TP_k + FP_k + FN_k) | Foreground k=1..6 with at least one predicted or reference pixel in that image |
| mIoU | Arithmetic mean of TP_k / (TP_k + FP_k + FN_k) | Same active foreground classes as mF1 |
| Dice-C | 2 size(P_1 intersect T_1) / (size(P_1) + size(T_1)) | Images containing class 1 in the reference |
| ASSD-C | Sum of all directed nearest-boundary distances divided by the total number of boundary points | Same Dice-C images, additionally requiring nonempty prediction and reference |
| HD95-C | Maximum of the two directed 95th-percentile nearest-boundary distances | Same ASSD-C images |

A confusion between two labels inside an F1 group counts as an error, including
1 versus 4. These scores do not simply merge a group into one binary mask.
Dice-C eligibility uses class-1 presence in the reference, not independently
recorded phase metadata. Closed-phase images with an empty class-1 mask are
therefore excluded; this must be reconsidered if phase metadata are available.

## Surface distances

A boundary comprises mask pixels removed by binary erosion with four-neighbor
connectivity and outside-image value 0. For boundaries A and B,
`d(a,B) = min_b ||a-b||_2`. Distances are in pixels without physical spacing.

```text
ASSD = [sum_a d(a,B) + sum_b d(b,A)] / (|A| + |B|)
HD95 = max(percentile95({d(a,B)}), percentile95({d(b,A)}))
```

Percentiles use NumPy's default linear interpolation. HD95 uses the maximum of
two directed percentiles, not the percentile of a pooled distance set. These
ASSD and HD95 formulas follow the manuscript's metric table.

## Empty masks and undefined values

- A class/group empty in both prediction and reference is undefined and omitted
  for that image. If all foreground classes are empty, mF1 and mIoU are null.
- A class/group empty on one side only receives overlap score 0.
- A Dice-C-eligible image with no predicted class-1 pixels receives Dice-C 0;
  its surface distances are null.
- Surface distances require two nonempty masks. Undefined distances are retained
  as null and excluded from the image mean. Their exclusion counts must be
  reported: a surface-distance mean alone does not summarize complete misses.

JSON uses null rather than NaN. Reports retain the metric value for each image,
plus `n_defined` and `n_undefined` for each image-level mean.

## Within-model and five-model summaries

1. Compute each metric separately for every image. Pixels are never pooled
   across images or batches.
2. A model's reported score is the arithmetic mean of its defined image scores.
   Each eligible image has equal weight.
3. For five distinct model reports from the same test images, reference masks,
   preprocessing, metric rules and training configuration, compute the arithmetic
   mean and sample SD (denominator 5-1) of the five model means.
4. If a model mean is undefined for a metric, that metric's five-model mean and
   SD are null. The individual values and counts of defined/undefined models are
   retained; the script does not substitute a four-model mean.

Variation among five models is not a patient-level confidence interval. Patient
or recording clustering is not accounted for by these image-level summaries.

## Historical rules still requiring confirmation

The original class grouping, background inclusion, empty-mask policy,
image/batch/pixel pooling, phase eligibility, surface definitions and five-model
aggregation need confirmation or a newly documented analysis. This release
makes these choices explicit without asserting their historical equivalence.
