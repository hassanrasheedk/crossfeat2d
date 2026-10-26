"""Image loading and case-directory layout checks."""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_LONG_SIDE = 192


def load_image(path: str | Path, long_side: int = DEFAULT_LONG_SIDE) -> torch.Tensor:
    """Load an 8-bit image as a float tensor of shape (3, H, W) in [0, 1].

    Grayscale images are repeated to three channels, alpha is dropped and the image is
    resized (bilinear) so that its longer side equals ``long_side``. This mirrors
    ``load_path_for_pytorch(path, long_side, force_scale=True)`` from the notebooks.
    """
    x = imageio.imread(path) / 255.0
    h, w = x.shape[:2]
    if x.ndim < 3:
        x = np.stack([x, x, x], 2)
    x = x[:, :, :3].astype(np.float32)
    t = torch.from_numpy(x).permute(2, 0, 1).contiguous()
    fac = float(long_side) / max(h, w)
    size = (int(h * fac), int(w * fac))
    return F.interpolate(t.unsqueeze(0), size, mode="bilinear", align_corners=False)[0]


def to_hwc(image: torch.Tensor) -> np.ndarray:
    """(3, H, W) tensor -> (H, W, 3) float32 array, the layout used for patch slicing."""
    return image.detach().cpu().numpy().transpose(1, 2, 0)


def check_case_dir(case_dir: str | Path, required: tuple[str, ...]) -> Path:
    """Raise a helpful error if ``case_dir`` lacks any of the ``required`` sub-directories."""
    case_dir = Path(case_dir)
    missing = [d for d in required if not (case_dir / d).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"{case_dir} is missing {', '.join(missing)}/. Expected layout:\n"
            "  <case>/t2/NNN.png                 T2 MR slices\n"
            "  <case>/us/NNN.png                 registered intraoperative US slices\n"
            "  <case>/synthetic_us/NNN-*.png     synthetic US generated from slice NNN\n"
            "See the 'Data' section of the README."
        )
    return case_dir


def list_slices(directory: Path) -> list[str]:
    """Sorted PNG file names in ``directory``."""
    return sorted(p.name for p in directory.glob("*.png"))
