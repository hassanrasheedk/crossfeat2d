"""Describe keypoints with a trained network and match them across two images."""

from __future__ import annotations

import numpy as np
import torch

from crossfeat2d.data import patch_to_tensor
from crossfeat2d.model import SiameseNetwork


@torch.no_grad()
def describe(
    model: SiameseNetwork, patches: list[np.ndarray], device, batch_size: int = 1024
) -> np.ndarray:
    """(N, 128) descriptors for (H, W, 3) patches."""
    model.eval()
    if not patches:
        return np.zeros((0, model.embedding_net.fc1.out_features), dtype=np.float32)
    out = []
    for i in range(0, len(patches), batch_size):
        batch = torch.stack([patch_to_tensor(p) for p in patches[i : i + batch_size]])
        out.append(model.get_embedding(batch.to(device)).cpu().numpy())
    return np.concatenate(out)


def match_cosine(
    desc1: np.ndarray,
    desc2: np.ndarray,
    sim_threshold: float = 0.95,
    ratio: float = 0.99,
) -> np.ndarray:
    """Nearest neighbour by cosine similarity with a similarity floor and a ratio test.

    Returns an (M, 2) array of index pairs. A match is kept if its best similarity
    exceeds ``sim_threshold`` and second-best / best is below ``ratio``. As in the
    notebooks, several keypoints in image 1 may match the same keypoint in image 2.
    """
    if len(desc1) == 0 or len(desc2) < 2:
        return np.zeros((0, 2), dtype=np.int64)
    d1 = desc1 / np.linalg.norm(desc1, axis=1, keepdims=True)
    d2 = desc2 / np.linalg.norm(desc2, axis=1, keepdims=True)
    sim = d1 @ d2.T
    order = np.argsort(sim, axis=1)[:, ::-1]
    rows = np.arange(len(sim))
    best, second = sim[rows, order[:, 0]], sim[rows, order[:, 1]]
    keep = (best > sim_threshold) & (second / best < ratio)
    return np.stack([rows[keep], order[keep, 0]], axis=1)


def build_lightglue(device):
    from lightglue import LightGlue

    # The DISK weights expect 128-d descriptors, the size of our embeddings.
    return LightGlue(features="disk").eval().to(device)


@torch.no_grad()
def match_lightglue(
    matcher, kpts1: np.ndarray, desc1: np.ndarray, kpts2: np.ndarray, desc2: np.ndarray, device
) -> np.ndarray:
    """Match with LightGlue (DISK weights) using our descriptors. Returns (M, 2) indices.

    No ``image_size`` is passed, so LightGlue normalises keypoints by their extent,
    exactly as the evaluation notebook did.
    """
    if len(kpts1) == 0 or len(kpts2) == 0:
        return np.zeros((0, 2), dtype=np.int64)

    def feats(k, d):
        return {
            "keypoints": torch.as_tensor(k, dtype=torch.float32, device=device)[None],
            "descriptors": torch.as_tensor(d, dtype=torch.float32, device=device)[None],
        }

    out = matcher({"image0": feats(kpts1, desc1), "image1": feats(kpts2, desc2)})
    return out["matches"][0].cpu().numpy()


def correct_matches(m_kpts1: np.ndarray, m_kpts2: np.ndarray, margin: float = 4.0) -> np.ndarray:
    """Boolean mask: matched points within ``margin`` px in both x and y.

    MR and US slices are registered, so a correct match lands at (almost) the same pixel.
    """
    return np.all(np.abs(m_kpts1 - m_kpts2) <= margin, axis=1)
