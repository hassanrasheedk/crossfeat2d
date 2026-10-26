"""``crossfeat2d train`` and ``crossfeat2d evaluate``."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from crossfeat2d.io import DEFAULT_LONG_SIDE


def _device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seed(seed: int | None) -> np.random.Generator:
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
    return np.random.default_rng(seed)


def cmd_train(args: argparse.Namespace) -> None:
    from torch.utils.data import DataLoader

    from crossfeat2d.data import SiamesePatchDataset, build_training_pairs, split_pairs
    from crossfeat2d.keypoints import build_superpoint
    from crossfeat2d.model import SiameseNetwork
    from crossfeat2d.train import train

    device = _device(args.device)
    rng = _seed(args.seed)
    extractor = build_superpoint(args.max_keypoints, device)
    positives, negatives = build_training_pairs(
        args.case_dir,
        extractor,
        keypoint_source=args.keypoints,
        synthetic_subdir=args.synthetic_subdir,
        long_side=args.long_side,
        patch_size=args.patch_size,
        shift=args.shift,
    )
    if not positives:
        raise SystemExit("No training pairs found; check the case directory layout.")
    train_pairs, train_labels, val_pairs, val_labels = split_pairs(positives, negatives, rng=rng)
    print(f"{len(train_pairs)} training pairs, {len(val_pairs)} validation pairs on {device}")

    generator = torch.Generator().manual_seed(args.seed) if args.seed is not None else None
    train_loader = DataLoader(
        SiamesePatchDataset(train_pairs, train_labels),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(SiamesePatchDataset(val_pairs, val_labels), batch_size=args.batch_size)

    model = SiameseNetwork().to(device)
    train(model, train_loader, val_loader, epochs=args.epochs, lr=args.lr, device=device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"saved {args.out}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    from crossfeat2d.evaluate import evaluate_case, summarize
    from crossfeat2d.keypoints import build_superpoint
    from crossfeat2d.matching import build_lightglue
    from crossfeat2d.model import SiameseNetwork

    device = _device(args.device)
    model = SiameseNetwork().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()
    results = evaluate_case(
        args.case_dir,
        model,
        build_superpoint(args.max_keypoints, device),
        matcher=args.matcher,
        lightglue=build_lightglue(device) if args.matcher == "lightglue" else None,
        long_side=args.long_side,
        patch_size=args.patch_size,
        margin=args.margin,
        device=device,
    )
    summary = summarize(results)
    for r in results:
        print(f"{r.slice}: {r.correct}/{r.matches} correct ({r.precision:.1f}%)")
    print(
        f"mean precision {summary['mean_precision']:.2f}% over {summary['slices']} slices "
        f"({summary['slices_without_matches']} without matches), "
        f"mean matches {summary['mean_matches']:.1f}"
    )
    if args.json:
        args.json.write_text(json.dumps(summary, indent=2))


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("case_dir", type=Path, help="case directory, e.g. data/mr_us/Case067")
    p.add_argument("--device", help="torch device (default: cuda if available, else cpu)")
    p.add_argument(
        "--long-side", type=int, default=DEFAULT_LONG_SIDE, help="resize images to this longer side"
    )
    p.add_argument(
        "--patch-size", type=int, default=64, help="patch size in px (the network expects 64)"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="crossfeat2d", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="train a patient-specific descriptor on one case")
    _common(t)
    t.add_argument("--out", type=Path, required=True, help="where to save the state dict (.pth)")
    t.add_argument(
        "--keypoints",
        choices=["synthetic", "t2"],
        default="synthetic",
        help="detect keypoints on each synthetic US image (default) or on the T2 slice",
    )
    t.add_argument(
        "--synthetic-subdir", default="synthetic_us", help="sub-directory with synthetic US"
    )
    t.add_argument("--max-keypoints", type=int, default=256, help="SuperPoint keypoints per image")
    t.add_argument("--shift", type=int, default=50, help="offset of negative patches in px")
    t.add_argument("--epochs", type=int, default=10)
    t.add_argument("--batch-size", type=int, default=512)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--seed", type=int, help="seed for the split, shuffling and initialisation")
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("evaluate", help="matching precision on registered real MR/US slices")
    _common(e)
    e.add_argument("--weights", type=Path, required=True, help="state dict saved by `train`")
    e.add_argument("--matcher", choices=["lightglue", "cosine"], default="lightglue")
    e.add_argument(
        "--margin", type=float, default=4.0, help="max x/y error of a correct match in px"
    )
    e.add_argument(
        "--max-keypoints",
        type=int,
        default=None,
        help="SuperPoint keypoints per image (default: all)",
    )
    e.add_argument("--json", type=Path, help="also write per-slice results to this JSON file")
    e.set_defaults(func=cmd_evaluate)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
