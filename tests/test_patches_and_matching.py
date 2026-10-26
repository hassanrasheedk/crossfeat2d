import numpy as np
import pytest

from crossfeat2d.keypoints import merge_keypoints
from crossfeat2d.matching import correct_matches, match_cosine
from crossfeat2d.patches import extract_patches, extract_pos_neg_patches


@pytest.fixture
def image():
    return np.arange(192 * 192 * 3, dtype=np.float32).reshape(192, 192, 3)


def test_extract_patches_skips_out_of_bounds(image):
    patches, kept = extract_patches(image, [(100.7, 90.2), (10, 10), (180, 100)], size=64)
    assert len(patches) == 1
    assert patches[0].shape == (64, 64, 3)
    np.testing.assert_allclose(kept, [[100.7, 90.2]], rtol=1e-6)
    # patch is centred on the integer part of the keypoint
    np.testing.assert_array_equal(patches[0][32, 32], image[90, 100])


def test_pos_neg_patches_are_aligned_and_shifted(image):
    pos, neg = extract_pos_neg_patches(image, [(40, 40), (150, 150)], size=64, shift=50)
    assert len(pos) == len(neg) == 2
    np.testing.assert_array_equal(pos[0][32, 32], image[40, 40])
    np.testing.assert_array_equal(neg[0][32, 32], image[90, 90])  # shifted +50
    np.testing.assert_array_equal(neg[1][32, 32], image[100, 100])  # shifted -50


def test_merge_keypoints():
    kpts = np.array([[10.0, 10.0], [12.0, 10.0], [100.0, 100.0]])
    merged = merge_keypoints(kpts, eps=5)
    assert sorted(map(tuple, merged)) == [(11.0, 10.0), (100.0, 100.0)]


def test_match_cosine_ratio_test():
    desc1 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    desc2 = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.99, 0.141]])
    matches = match_cosine(desc1, desc2)
    # row 0 matches desc2[1] unambiguously; row 1 is ambiguous (ratio ~0.99) and dropped
    np.testing.assert_array_equal(matches, [[0, 1]])


def test_correct_matches_margin():
    a = np.array([[10.0, 10.0], [10.0, 10.0]])
    b = np.array([[14.0, 6.0], [14.5, 10.0]])
    np.testing.assert_array_equal(correct_matches(a, b, margin=4), [True, False])
