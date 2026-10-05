# Third-party notices

## segmentation_models.pytorch

The model patch derives from segmentation_models.pytorch 0.3.2, copyright
2019 Pavel Iakubovskii, licensed under MIT. Its notice is preserved at
`patches/smp-0.3.2/LICENSE.upstream`.

## Runtime libraries

PyTorch, torchvision, timm, pretrainedmodels, efficientnet-pytorch, NumPy,
SciPy, Pillow, OpenCV, matplotlib and other installed dependencies retain their
own licenses. Their installed distributions are not bundled in this repository.
ImageNet encoder weights are downloaded separately when requested by training.

## Study-team modifications

The model's local CaFe/AdaUp/InDis code was supplied in a modified SMP package.
The study team has confirmed permission to publish its modifications under the
repository's MIT License. The upstream MIT notice remains in
`patches/smp-0.3.2/LICENSE.upstream`. This repository includes no recovered
project archive, clinical files or model checkpoints.

## CARAFE and ULSAM method attribution

The AdaUp content-aware reassembly operation follows CARAFE by Jiaqi Wang,
Kai Chen, Rui Xu, Ziwei Liu, Chen Change Loy and Dahua Lin (ICCV 2019;
doi:10.1109/ICCV.2019.00310).

The InDis method follows the grouped subspace-attention formulation by
Rajat Saini, Nandan Kumar Jha, Bedanta Das, Sparsh Mittal and C. Krishna Mohan
(WACV 2020; doi:10.1109/WACV45572.2020.9093341), equations 2-4.
Version 0.1.4 implements these operations in GroupedSpatialAttention with
PyTorch depthwise and grouped pointwise convolutions. The previous SubSpace
and ULSAM source classes are not included in this release. The new implementation
is part of the FiDi contributions covered by this repository's MIT License.
Method attribution does not grant or imply a license to the authors' separate
reference code.
