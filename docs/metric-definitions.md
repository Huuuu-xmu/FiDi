# Metric definitions and aggregation

This document specifies the pixel-level metrics computed by version 0.1.3.
The formulas and averaging terminology follow the sources listed below; class
selection, image eligibility and undefined-value handling are explicit release
conventions. The study uses F1 as a supplementary assessment of open/closed phase confusion.
Software outputs `pixel-mF1`, `pixel-F1-G` and `pixel-F1-V` measure pixel-label
agreement. Independent image-phase classification is outside this evaluator.

## Classes and overlap counts

For class k in one image, let P_k and T_k be predicted and reference pixel sets.
TP_k = |P_k intersect T_k|, FP_k = |P_k minus T_k| and
FN_k = |T_k minus P_k|. Background (0) is excluded from all foreground macro
scores. Phase and side labels are distinct classes.

| Metric | Formula / calculation | Classes or eligible images |
| --- | --- | --- |
| pixel-F1-G | 2 sum(TP_k) / sum(2 TP_k + FP_k + FN_k) | k in {1,4}; per-image micro F1 over distinct phase labels |
| pixel-F1-V | Same grouped micro formula | k in {2,3,5,6}; distinct phase and side labels |
| pixel-mF1 | Arithmetic mean of 2 TP_k / (2 TP_k + FP_k + FN_k) | Foreground k=1..6 with at least one predicted or reference pixel in that image |
| mIoU | Arithmetic mean of TP_k / (TP_k + FP_k + FN_k) | Same active foreground classes as pixel-mF1 |
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
two directed percentiles, not the percentile of a pooled distance set. The ASSD convention agrees with MedPy's pooled bidirectional surface-distance
average. The HD95 convention agrees with MONAI's undirected percentile
Hausdorff calculation. Both are documented in Table S4. MedPy's HD95 uses a
different pooled-percentile convention, so package names alone do not specify
identical metrics.

## Empty masks and undefined values

- A class/group empty in both prediction and reference is undefined and omitted
  for that image. If all foreground classes are empty, pixel-mF1 and mIoU are null.
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

## Relation between pixel-level F1 and IoU

For identical prediction/reference sets, F1 = 2 IoU / (1 + IoU).
For identical class/image weights, concavity implies
mean(F1) <= 2 mean(IoU) / (1 + mean(IoU)). This comparison is valid only when
evaluation units, eligible cases and weights agree. An image-level phase F1
would be a different metric and would require explicit phase labels and a
prediction rule; this release does not compute it.

## Sources

- Taha AA, Hanbury A. Metrics for evaluating 3D medical image segmentation: analysis, selection, and tool. BMC Med Imaging. 2015;15:29. https://doi.org/10.1186/s12880-015-0068-x
- Maier-Hein L, Reinke A, Godau P, et al. Metrics reloaded: recommendations for image analysis validation. Nat Methods. 2024;21:195-212. https://doi.org/10.1038/s41592-023-02151-z
- Scikit-learn F1 score documentation (macro/micro weighting and undefined values): https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
- MedPy ASSD documentation: https://loli.github.io/medpy/generated/medpy.metric.binary.assd.html
- MONAI Hausdorff source (undirected maximum of directed percentiles): https://monai-dev.readthedocs.io/en/stable/_modules/monai/metrics/hausdorff_distance.html

These sources describe metric formulas and aggregation terminology.
