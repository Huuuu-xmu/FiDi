# Execution and reporting

## Package scope

Version 0.1.4 contains the FiDi model, preprocessing, participant-grouped data
checks, training, checkpoint inference, LSLE pixel-level evaluation and
five-model aggregation. A separate evaluator handles Genova class-index masks.
Model architecture and loss functions are described in
[model implementation](model-implementation.md).

Prepare the image manifest, images and class-index masks before training.
For inference, provide a checkpoint with the required class mapping and model
settings. The repository contains code and documentation; clinical data and
trained study checkpoints are not included.

## Model and training settings

| Setting | Value |
| --- | --- |
| Encoder | ImageNet-pretrained ResNet-18 |
| Modules | CaFe, AdaUp and InDis |
| LSLE labels | Six phase-specific foreground classes plus background |
| Input | Native-resolution 448 × 448 center crop |
| Augmentation | None |
| Loss | Multiclass cross-entropy plus weighted auxiliary boundary BCE |
| Boundary target | Canny edges of the union of glottis labels 1 and 4 |
| Boundary-loss weight | 0.1 |
| Optimizer | Adam |
| Training | 20 epochs, batch size 16 |
| Learning rate | Cosine annealing from 1e-4 to 5e-5 |
| Checkpoint selection | Highest validation mIoU within each development fold |

Use participant-separated development/test partitions and five participant-level
development folds. Evaluate each selected checkpoint on the same fixed test set.
The package default seed is 2026; record any change made with `--seed`.

The manuscript reports PyTorch 1.9.1, CUDA 11.1 and an NVIDIA GeForce RTX 3090.
Local package checks used Python 3.9.20, PyTorch 2.2.2 and torchvision 0.17.2
on a Mac CPU. The local dependency list is recorded in
[tested-environment.txt](tested-environment.txt).

## Experiment records

Keep the data manifest, configuration, random seed, selected checkpoint and
reports for each fold. Reports record software versions, source hashes,
checkpoint hashes, image and reference-mask hashes, and metric conventions.
These records identify the inputs and implementation used for a run.

## Metric scope

The LSLE evaluator computes pixel-label segmentation metrics. Outputs with the
`pixel-` F1 prefix are separate from the supplementary phase-related F1 values
in the manuscript. [Metric definitions](metric-definitions.md) specifies the
formulas, eligible images and handling of empty masks.

Image means weight each eligible image equally. Five-model summaries report
the mean and sample SD of five scores from the same fixed test set. Genova uses
its own six-class mapping and pools pixel counts across images within each class.
