"""Write class-index masks for *_rgb.png images using a trusted checkpoint."""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fidi.data import load_image
from fidi.model import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="Directory containing *_rgb.png")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trusted state_dict checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--disable-cafe", action="store_true", help="Use for a checkpoint trained without CaFe")
    parser.add_argument("--disable-adaup", action="store_true", help="Use for a checkpoint trained without AdaUp")
    parser.add_argument("--disable-indis", action="store_true", help="Use for a checkpoint trained without InDis")
    parser.add_argument("--trust-legacy-checkpoint", action="store_true", help="Permit torch.load on PyTorch without weights_only")
    args = parser.parse_args()

    import torch

    images = sorted(args.images.glob("*_rgb.png"))
    if not images:
        raise SystemExit("No *_rgb.png images found")
    if args.input_size < 32 or args.input_size % 32:
        parser.error("input-size must be a positive multiple of 32")
    load_kwargs = {"map_location": args.device}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = True
    elif not args.trust_legacy_checkpoint:
        raise SystemExit("This PyTorch lacks weights_only; pass --trust-legacy-checkpoint only for a trusted file")
    state = torch.load(args.checkpoint, **load_kwargs)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model = build_model(
        pretrained=False,
        use_cafe=not args.disable_cafe,
        use_adaup=not args.disable_adaup,
        use_indis=not args.disable_indis,
    ).to(args.device)
    model.load_state_dict(state, strict=True)
    model.eval()
    args.output.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for image_path in images:
            image = torch.from_numpy(load_image(
                image_path, input_size=args.input_size
            ).copy()).unsqueeze(0).to(args.device)
            scores, _ = model(image)
            pred = scores.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
            output = args.output / image_path.name.replace("_rgb.png", "_pred.png")
            Image.fromarray(pred, mode="L").save(output)
            print(output)


if __name__ == "__main__":
    main()
