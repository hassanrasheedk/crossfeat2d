"""Matching precision on registered real MR/US slices of one case."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from crossfeat2d.io import DEFAULT_LONG_SIDE, check_case_dir, list_slices, load_image, to_hwc
from crossfeat2d.keypoints import detect_keypoints
from crossfeat2d.matching import correct_matches, describe, match_cosine, match_lightglue
from crossfeat2d.model import SiameseNetwork
from crossfeat2d.patches import extract_patches


@dataclass
class SliceResult:
    slice: str
    matches: int
    correct: int

    @property
    def precision(self) -> float:
        return 100.0 * self.correct / self.matches if self.matches else math.nan


def match_slice(
    model: SiameseNetwork,
    extractor,
    image1,
    image2,
    *,
    matcher: Literal["lightglue", "cosine"] = "lightglue",
    lightglue=None,
    patch_size: int = 64,
    device="cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Detect, describe and match keypoints between two (3, H, W) images.

    Returns the matched keypoints in image 1 and image 2, each of shape (M, 2).
    """
    patches1, kpts1 = extract_patches(
        to_hwc(image1), detect_keypoints(extractor, image1), patch_size
    )
    patches2, kpts2 = extract_patches(
        to_hwc(image2), detect_keypoints(extractor, image2), patch_size
    )
    desc1, desc2 = describe(model, patches1, device), describe(model, patches2, device)
    if matcher == "lightglue":
        idx = match_lightglue(lightglue, kpts1, desc1, kpts2, desc2, device)
    else:
        idx = match_cosine(desc1, desc2)
    return kpts1[idx[:, 0]], kpts2[idx[:, 1]]


def evaluate_case(
    case_dir: str | Path,
    model: SiameseNetwork,
    extractor,
    *,
    matcher: Literal["lightglue", "cosine"] = "lightglue",
    lightglue=None,
    long_side: int = DEFAULT_LONG_SIDE,
    patch_size: int = 64,
    margin: float = 4.0,
    device="cpu",
) -> list[SliceResult]:
    """Per-slice match counts for every ``t2/NNN.png`` that has a ``us/NNN.png``."""
    case_dir = check_case_dir(case_dir, ("t2", "us"))
    us_names = set(list_slices(case_dir / "us"))
    results = []
    for name in list_slices(case_dir / "t2"):
        if name not in us_names:
            continue
        m1, m2 = match_slice(
            model,
            extractor,
            load_image(case_dir / "t2" / name, long_side),
            load_image(case_dir / "us" / name, long_side),
            matcher=matcher,
            lightglue=lightglue,
            patch_size=patch_size,
            device=device,
        )
        results.append(SliceResult(name, len(m1), int(correct_matches(m1, m2, margin).sum())))
    return results


def summarize(results: list[SliceResult]) -> dict:
    """Mean per-slice precision over slices with at least one match."""
    precisions = [r.precision for r in results if r.matches]
    return {
        "slices": len(results),
        "slices_without_matches": len(results) - len(precisions),
        "mean_matches": float(np.mean([r.matches for r in results])) if results else math.nan,
        "mean_precision": float(np.mean(precisions)) if precisions else math.nan,
        "per_slice": [asdict(r) | {"precision": r.precision} for r in results],
    }
