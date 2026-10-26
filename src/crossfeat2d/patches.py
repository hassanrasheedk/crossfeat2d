"""Square patch extraction around keypoints."""

from __future__ import annotations

import numpy as np


def _in_bounds(x: int, y: int, size: int, h: int, w: int) -> bool:
    half = size // 2
    return x - half >= 0 and y - half >= 0 and x + half <= w and y + half <= h


def extract_patches(
    image: np.ndarray, keypoints, size: int = 64
) -> tuple[list[np.ndarray], np.ndarray]:
    """Cut ``size`` x ``size`` patches centred on ``keypoints`` from an (H, W, C) image.

    Keypoints whose patch would leave the image are skipped. Returns the patches (views
    into ``image``) and the (M, 2) keypoints that were kept, at their original sub-pixel
    position.
    """
    h, w = image.shape[:2]
    half = size // 2
    patches, kept = [], []
    for kp in keypoints:
        x, y = int(kp[0]), int(kp[1])
        if not _in_bounds(x, y, size, h, w):
            continue
        patches.append(image[y - half : y + half, x - half : x + half])
        kept.append((float(kp[0]), float(kp[1])))
    return patches, np.asarray(kept, dtype=np.float32).reshape(-1, 2)


def extract_pos_neg_patches(
    image: np.ndarray, keypoints, size: int = 64, shift: int = 50
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Positive patch at each keypoint and a negative patch ``shift`` px away diagonally.

    The negative is shifted towards +x/+y unless that would leave the image, in which
    case it is shifted the other way, then clamped inside the image.
    """
    h, w = image.shape[:2]
    half = size // 2
    pos, neg = [], []
    for kp in keypoints:
        x, y = int(kp[0]), int(kp[1])
        if not _in_bounds(x, y, size, h, w):
            continue
        pos.append(image[y - half : y + half, x - half : x + half])
        nx = x + shift if x + shift < w - half else x - shift
        ny = y + shift if y + shift < h - half else y - shift
        nx = max(min(nx, w - half), half)
        ny = max(min(ny, h - half), half)
        neg.append(image[ny - half : ny + half, nx - half : nx + half])
    return pos, neg
