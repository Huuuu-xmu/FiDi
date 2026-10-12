"""Build FiDi with a ResNet-18 encoder and the local SMP model files."""

from __future__ import annotations

import sys
from pathlib import Path


def build_model(
    *,
    pretrained: bool = True,
    classes: int = 7,
    use_cafe: bool = True,
    use_adaup: bool = True,
    use_indis: bool = True,
):
    local = Path(__file__).resolve().parents[1] / ".local"
    if not (local / "segmentation_models_pytorch").is_dir():
        raise RuntimeError("Run `python scripts/prepare_smp.py` before building the model")
    if str(local) not in sys.path:
        sys.path.insert(0, str(local))
    import segmentation_models_pytorch as smp

    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=classes,
        activation=None,
    )
    model.decoder.configure(
        use_cafe=use_cafe,
        use_adaup=use_adaup,
        use_indis=use_indis,
    )
    return model
