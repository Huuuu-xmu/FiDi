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

The FiDi team's CaFe, AdaUp and InDis modifications are released under the
repository's MIT License. The SMP notice is preserved in
`patches/smp-0.3.2/LICENSE.upstream`.

## CARAFE and ULSAM method attribution

The AdaUp content-aware reassembly operation follows CARAFE by Jiaqi Wang,
Kai Chen, Rui Xu, Ziwei Liu, Chen Change Loy and Dahua Lin (ICCV 2019;
doi:10.1109/ICCV.2019.00310).

The InDis method follows the grouped subspace-attention formulation by
Rajat Saini, Nandan Kumar Jha, Bedanta Das, Sparsh Mittal and C. Krishna Mohan
(WACV 2020; doi:10.1109/WACV45572.2020.9093341), equations 2-4.
`GroupedSpatialAttention` implements these operations using PyTorch depthwise
and grouped pointwise convolutions. This implementation is covered by the
repository's MIT License; the cited authors' reference code retains its own
license.
