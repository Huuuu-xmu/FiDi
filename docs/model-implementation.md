# Model implementation

FiDi means **fine and disambiguous segmentation**. Its modules are CaFe
(**cross-scale gated fine-edge perception**), AdaUp (**adaptive upsampling**)
and InDis (**intra-scale gated organ-structure disambiguation**).

The builder fixes an SMP U-Net with a ResNet-18 encoder and seven output channels.
The segmentation head returns logits. The auxiliary branch returns a sigmoid
boundary probability map. Preprocessing and labels are described in
[data format](data-format.md).

| Component | Implementation in this version | Disabled behavior |
| --- | --- | --- |
| CaFe | `ShapeStream` combines three encoder scales using gated convolutions, predicts a boundary map and produces a spatial cue. The cue modulates the final decoder feature as `x * (1 + feat)`. | Skip the boundary branch, feature modulation and auxiliary boundary loss. |
| AdaUp | CARAFE-style content-aware upsampling at three coarse decoder stages and one step in each of the three CaFe paths. Other resolution alignment uses interpolation. | Use nearest interpolation in the decoder and bilinear alignment in CaFe. |
| InDis | ULSAM-style grouped attention on encountered 512- and 256-channel decoder inputs for the ResNet-18 path. | Bypass the grouped-attention calls. |

CARAFE uses scale 2, reassembly kernel 3 and encoder kernel 3. Decoder channel
compression is `min(input_channels, 64)`. CaFe paths use compression widths
16, 16 and 64. The two fine decoder stages use nearest interpolation.
These settings specify the model constructed by this package.

The target boundary is Canny(`glottis_mask * 255`, 1, 3), using the union of
labels 1 and 4. The training objective is pixel-mean cross-entropy plus lambda
times weighted boundary BCE. For the boundary map, the positive-pixel weight is
the fraction of negative pixels in the batch, and vice versa; weighted BCE is
summed and divided by the number of boundary pixels. Lambda defaults to 0.1.

The training command uses 20 epochs by default and selects the checkpoint with the highest validation mIoU. Checkpoints should be loaded with the same model switches and class mapping used for training.

## Available source

The SMP model files derive from the study team's available development source.
The final original CaFe-to-decoder fusion code was not retained. This package
uses `x * (1 + feat)` for that fusion, based on the available features and method
description. This operation has not been verified against the final original
training code. The executable training and evaluation commands were assembled
for this package; their execution does not reproduce the manuscript tables
without the study inputs and experiment records.

## Method sources

AdaUp uses content-aware feature reassembly as described by Wang et al.,
“CARAFE: Content-Aware ReAssembly of FEatures,” ICCV 2019, pp. 3007–3016,
[doi:10.1109/ICCV.2019.00310](https://doi.org/10.1109/ICCV.2019.00310).

InDis adapts the grouped subspace-attention operations described by Saini,
Jha, Das, Mittal and Mohan, “ULSAM: Ultra-Lightweight Subspace Attention
Module for Compact Convolutional Neural Networks,” WACV 2020, pp. 1616–1625,
[doi:10.1109/WACV45572.2020.9093341](https://doi.org/10.1109/WACV45572.2020.9093341).
The release implements these published operations in `GroupedSpatialAttention`
using depthwise and grouped pointwise convolutions. Version 0.1.4 uses grouped parameter tensors. The loader converts earlier
FiDi per-group attention tensors to this representation when loading a checkpoint.
Feature groups are learned channel subspaces; they are not fixed anatomical
class assignments.
