"""Training pairs from one case: T2 patches against synthetic-US patches."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from crossfeat2d.io import DEFAULT_LONG_SIDE, check_case_dir, list_slices, load_image, to_hwc
from crossfeat2d.keypoints import detect_keypoints, merge_keypoints
from crossfeat2d.patches import extract_patches, extract_pos_neg_patches

Pair = tuple[np.ndarray, np.ndarray]


def build_training_pairs(
    case_dir: str | Path,
    extractor,
    *,
    keypoint_source: Literal["synthetic", "t2"] = "synthetic",
    synthetic_subdir: str = "synthetic_us",
    long_side: int = DEFAULT_LONG_SIDE,
    patch_size: int = 64,
    shift: int = 50,
    dbscan_eps: float = 5.0,
) -> tuple[list[Pair], list[Pair]]:
    """Positive and negative (T2 patch, synthetic-US patch) pairs for every slice of a case.

    For slice ``NNN`` every ``<synthetic_subdir>/NNN-*.png`` is a synthetic US image
    pixel-aligned with ``t2/NNN.png``, so a keypoint has the same coordinates in both.
    Keypoints are detected with SuperPoint either on each synthetic image
    (``"synthetic"``, the main method) or once on the T2 slice (``"t2"``, the ablation),
    and merged with DBSCAN. The negative for each keypoint is the synthetic patch
    ``shift`` px away (see :func:`extract_pos_neg_patches`).
    """
    case_dir = check_case_dir(case_dir, ("t2", synthetic_subdir))
    synthetic_dir = case_dir / synthetic_subdir
    synthetic_names = list_slices(synthetic_dir)

    positives: list[Pair] = []
    negatives: list[Pair] = []
    for t2_name in list_slices(case_dir / "t2"):
        stem = Path(t2_name).stem
        t2_t = load_image(case_dir / "t2" / t2_name, long_side)
        t2 = to_hwc(t2_t)
        if keypoint_source == "t2":
            t2_kpts = merge_keypoints(detect_keypoints(extractor, t2_t), dbscan_eps)

        for name in (n for n in synthetic_names if n.startswith(f"{stem}-")):
            synthetic_t = load_image(synthetic_dir / name, long_side)
            if synthetic_t.shape != t2_t.shape:
                raise ValueError(f"{name} and t2/{t2_name} differ in size")
            if keypoint_source == "t2":
                kpts = t2_kpts
            else:
                kpts = merge_keypoints(detect_keypoints(extractor, synthetic_t), dbscan_eps)

            base, _ = extract_patches(t2, kpts, patch_size)
            pos, neg = extract_pos_neg_patches(to_hwc(synthetic_t), kpts, patch_size, shift)
            positives.extend(zip(base, pos, strict=True))
            negatives.extend(zip(base, neg, strict=True))
    return positives, negatives


def split_pairs(
    positives: list[Pair],
    negatives: list[Pair],
    train_fraction: float = 0.8,
    rng: np.random.Generator | None = None,
) -> tuple[list[Pair], list[int], list[Pair], list[int]]:
    """Label positives 1 and negatives 0, shuffle, and split into train and validation."""
    rng = rng if rng is not None else np.random.default_rng()
    pairs = positives + negatives
    labels = [1] * len(positives) + [0] * len(negatives)
    order = rng.permutation(len(pairs))
    cut = int(np.floor(train_fraction * len(pairs)))
    train, val = order[:cut], order[cut:]
    return (
        [pairs[i] for i in train],
        [labels[i] for i in train],
        [pairs[i] for i in val],
        [labels[i] for i in val],
    )


class SiamesePatchDataset(Dataset):
    """Pairs of (H, W, 3) float patches served as (3, H, W) tensors with a float label."""

    def __init__(self, pairs: list[Pair], labels: list[int]):
        self.pairs = list(pairs)
        self.labels = list(labels)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        a, b = self.pairs[index]
        return (patch_to_tensor(a), patch_to_tensor(b)), torch.tensor(
            self.labels[index], dtype=torch.float32
        )


def patch_to_tensor(patch: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(patch.transpose(2, 0, 1)))
