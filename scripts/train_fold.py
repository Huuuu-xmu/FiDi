"""Train one research FiDi development fold from an explicit manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from statistics import mean

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fidi.manifest import fold_partition, read_manifest
from fidi.metrics import CONVENTIONS, aggregate_images, image_metrics
from fidi.provenance import execution_metadata


def boundary_loss(edge, boundary):
    """Weighted auxiliary boundary BCE following the recovered loss code."""
    import torch
    import torch.nn.functional as F

    edge = edge.squeeze(1)
    pos = boundary == 1
    neg = boundary == 0
    n = boundary.numel()
    weight = torch.zeros_like(boundary)
    weight[pos] = neg.sum() / n
    weight[neg] = pos.sum() / n
    return F.binary_cross_entropy(edge, boundary, weight=weight, reduction="sum") / n


def evaluate(model, loader, device, *, include_image_scores=False):
    import torch

    model.eval()
    all_images = []
    with torch.no_grad():
        for images, targets, _ in loader:
            scores, _ = model(images.to(device))
            predictions = scores.argmax(1).cpu().numpy()
            truths = targets.numpy()
            all_images.extend(image_metrics(p, t) for p, t in zip(predictions, truths))
    aggregate = aggregate_images(all_images)
    return (aggregate, all_images) if include_image_scores else aggregate


def main():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, help="JSON file of training defaults")
    early, _ = config_parser.parse_known_args()
    parser = argparse.ArgumentParser(description=__doc__, parents=[config_parser])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True, choices=range(1, 6))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--input-size", type=int, default=448, help="Square model input; manuscript value is 448")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no-pretrained", action="store_false", dest="pretrained", help="Skip manuscript-specified ImageNet initialization")
    parser.set_defaults(pretrained=True)
    parser.add_argument("--boundary-weight", type=float, default=0.1, help="Lambda for auxiliary boundary loss")
    parser.add_argument("--boundary-classes", default="1,4", help="Comma-separated class IDs used for the glottis edge target")
    parser.add_argument("--disable-cafe", action="store_true", help="Ablate CaFe and its boundary loss")
    parser.add_argument("--disable-adaup", action="store_true", help="Ablate adaptive upsampling")
    parser.add_argument("--disable-indis", action="store_true", help="Ablate intra-scale disambiguation")
    parser.add_argument("--group-folds-by-participant", action="store_true", default=True, help="Participant-grouped folds are mandatory; retained for CLI compatibility")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    if early.config:
        config = json.loads(early.config.read_text(encoding="utf-8"))
        allowed = {action.dest for action in parser._actions}
        unknown = set(config) - allowed
        if unknown:
            parser.error(f"Unknown configuration keys: {', '.join(sorted(unknown))}")
        parser.set_defaults(**config)
    args = parser.parse_args()
    if not args.group_folds_by_participant:
        parser.error("Participant grouping is mandatory and cannot be disabled by a configuration file")

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from fidi.data import ManifestDataset
    from fidi.model import build_model

    if args.epochs < 1 or args.batch_size < 1 or args.boundary_weight < 0:
        parser.error("epochs and batch-size must be positive; boundary-weight cannot be negative")
    if args.input_size < 32 or args.input_size % 32:
        parser.error("input-size must be a positive multiple of 32")
    try:
        boundary_classes = tuple(int(part) for part in args.boundary_classes.split(","))
    except ValueError:
        parser.error("boundary-classes must be comma-separated integers")
    if not boundary_classes or any(label not in range(1, 7) for label in boundary_classes):
        parser.error("boundary-classes must contain foreground labels 1..6")
    rows = read_manifest(args.manifest, group_folds_by_participant=args.group_folds_by_participant)
    train_rows, val_rows, test_rows = fold_partition(rows, args.fold)
    if not test_rows:
        parser.error("The manuscript workflow requires a nonempty fixed test set")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    data_options = {"input_size": args.input_size}
    dataset_options = {**data_options, "boundary_classes": boundary_classes}
    train_loader = DataLoader(ManifestDataset(train_rows, **dataset_options), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(ManifestDataset(val_rows, **dataset_options), batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(ManifestDataset(test_rows, **dataset_options), batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = build_model(
        pretrained=args.pretrained,
        use_cafe=not args.disable_cafe,
        use_adaup=not args.disable_adaup,
        use_indis=not args.disable_indis,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=5e-5)
    args.output.mkdir(parents=True, exist_ok=True)

    best_score = -1.0
    best_epoch = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for images, targets, boundary in train_loader:
            images, targets, boundary = images.to(device), targets.to(device), boundary.to(device)
            scores, edge = model(images)
            loss = F.cross_entropy(scores, targets)
            if edge is not None and args.boundary_weight:
                loss = loss + args.boundary_weight * boundary_loss(edge, boundary)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        validation = evaluate(model, val_loader, device)
        score = validation["mIoU"]["mean"]
        if score is None:
            raise RuntimeError("Validation mIoU undefined; inspect the labels")
        history.append({"epoch": epoch, "train_loss": mean(losses), "validation": validation})
        print(f"epoch={epoch} train_loss={mean(losses):.5f} val_mIoU={score:.5f}")
        if score > best_score:
            best_score, best_epoch = score, epoch
            torch.save(model.state_dict(), args.output / "best_state_dict.pth")

    # Load only a checkpoint written by this same run. The fixed test is not
    # involved in checkpoint selection.
    state = torch.load(args.output / "best_state_dict.pth", map_location=device)
    model.load_state_dict(state)
    test_metrics, test_image_metrics = evaluate(model, test_loader, device, include_image_scores=True)
    provenance = execution_metadata()
    checkpoint_hash = hashlib.sha256((args.output / "best_state_dict.pth").read_bytes()).hexdigest()
    test_report = {
        "status": "fixed_test_evaluation_completed",
        "fold": args.fold,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "checkpoint_sha256": checkpoint_hash,
        "execution": provenance,
        "n_images": len(test_rows),
        "image_ids": [row["image_id"] for row in test_rows],
        "image_sha256": [hashlib.sha256(row["image_path"].read_bytes()).hexdigest() for row in test_rows],
        "reference_sha256": [hashlib.sha256(row["mask_path"].read_bytes()).hexdigest() for row in test_rows],
        "crop_reference": True,
        "reference_preprocessing": data_options,
        "metric_conventions": CONVENTIONS,
        "model_config": {
            "architecture": "FiDi",
            "pretrained": args.pretrained,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "boundary_weight": 0.0 if args.disable_cafe else args.boundary_weight,
            "boundary_classes": boundary_classes,
            "input_size": args.input_size,
            "CaFe": not args.disable_cafe,
            "AdaUp": not args.disable_adaup,
            "InDis": not args.disable_indis,
        },
        "metrics": test_metrics,
        "per_image": [
            {"image_id": row["image_id"], "metrics": values}
            for row, values in zip(test_rows, test_image_metrics)
        ],
    }
    (args.output / "test_report.json").write_text(json.dumps(test_report, indent=2) + "\n", encoding="utf-8")
    report = {
        "status": "training_completed",
        "execution": provenance,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "checkpoint_sha256": checkpoint_hash,
        "config_path": str(args.config) if args.config else None,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest() if args.config else None,
        "fold": args.fold,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "input_size": args.input_size,
        "pretrained": args.pretrained,
        "modules": {
            "CaFe": not args.disable_cafe,
            "AdaUp": not args.disable_adaup,
            "InDis": not args.disable_indis,
        },
        "boundary_weight": 0.0 if args.disable_cafe else args.boundary_weight,
        "boundary_classes": boundary_classes,
        "group_folds_by_participant": args.group_folds_by_participant,
        "model_specification": "FiDi_implementation_documented_in_model_implementation_md",
        "metric_conventions": CONVENTIONS,
        "device": device,
        "counts": {"train_images": len(train_rows), "validation_images": len(val_rows), "test_images": len(test_rows)},
        "best_epoch": best_epoch,
        "best_validation_mIoU": best_score,
        "history": history,
        "test": test_metrics,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved best weights, run report, and fixed-test report to {args.output}")


if __name__ == "__main__":
    main()
