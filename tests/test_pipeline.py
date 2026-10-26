"""End-to-end smoke tests on a tiny synthetic case directory (no ReMIND data, no downloads)."""

import imageio.v2 as imageio
import numpy as np
import pytest
import torch

from crossfeat2d.data import build_training_pairs, split_pairs
from crossfeat2d.evaluate import evaluate_case, summarize
from crossfeat2d.model import SiameseNetwork

KEYPOINTS = [[50.0, 60.0], [52.0, 61.0], [100.0, 100.0], [140.0, 70.0], [5.0, 5.0]]


class FakeExtractor:
    """Stands in for SuperPoint: returns fixed keypoints in LightGlue's batched format."""

    def extract(self, image):
        return {"keypoints": torch.tensor([KEYPOINTS])}


@pytest.fixture
def case_dir(tmp_path):
    rng = np.random.default_rng(0)
    for sub in ("t2", "us", "synthetic_us"):
        (tmp_path / sub).mkdir()
    for s in ("000", "001"):
        img = rng.integers(0, 255, (192, 192), dtype=np.uint8)
        imageio.imwrite(tmp_path / "t2" / f"{s}.png", img)
        imageio.imwrite(tmp_path / "us" / f"{s}.png", 255 - img)
        for style in ("t2_0.3", "flair_1"):
            imageio.imwrite(tmp_path / "synthetic_us" / f"{s}-{style}.png", img // 2)
    return tmp_path


@pytest.mark.parametrize("source", ["synthetic", "t2"])
def test_build_training_pairs(case_dir, source):
    pos, neg = build_training_pairs(case_dir, FakeExtractor(), keypoint_source=source)
    # 2 slices x 2 styles x 3 merged in-bounds keypoints ((50,60)+(52,61) merge; (5,5) is dropped)
    assert len(pos) == len(neg) == 12
    a, b = pos[0]
    assert a.shape == b.shape == (64, 64, 3)
    train, train_labels, val, val_labels = split_pairs(pos, neg, rng=np.random.default_rng(0))
    assert len(train) == 19 and len(val) == 5
    assert sorted(train_labels + val_labels) == [0] * 12 + [1] * 12


def test_missing_directory_has_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Expected layout"):
        build_training_pairs(tmp_path, FakeExtractor())


def test_evaluate_case_cosine(case_dir):
    torch.manual_seed(0)
    results = evaluate_case(case_dir, SiameseNetwork(), FakeExtractor(), matcher="cosine")
    assert [r.slice for r in results] == ["000.png", "001.png"]
    summary = summarize(results)
    assert summary["slices"] == 2
    assert all(0 <= r.correct <= r.matches for r in results)
