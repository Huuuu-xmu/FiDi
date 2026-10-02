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
