"""SuperPoint keypoint detection and merging of nearby detections."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.cluster import DBSCAN


def build_superpoint(max_num_keypoints: int | None, device: torch.device | str):
    from lightglue import SuperPoint

    return SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(device)


@torch.no_grad()
def detect_keypoints(extractor, image: torch.Tensor) -> np.ndarray:
    """(N, 2) array of (x, y) keypoints detected on a (3, H, W) image."""
    from lightglue.utils import rbd

    device = next(extractor.parameters()).device
    feats = rbd(extractor.extract(image.to(device)))
    return feats["keypoints"].cpu().numpy()


def merge_keypoints(kpts: np.ndarray, eps: float = 5.0) -> np.ndarray:
    """Replace each DBSCAN cluster of keypoints (``min_samples=1``) by its mean."""
    if len(kpts) == 0:
        return kpts.reshape(0, 2)
    labels = DBSCAN(eps=eps, min_samples=1).fit(kpts).labels_
    return np.stack([kpts[labels == label].mean(axis=0) for label in np.unique(labels)])
