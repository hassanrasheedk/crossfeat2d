"""Descriptor network, Siamese wrapper and contrastive loss with hard negatives."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingNet(nn.Module):
    """Maps a 3 x 64 x 64 patch to a 128-d descriptor.

    Called ``UNet`` in the notebooks; layer names are kept so their checkpoints load.
    """

    def __init__(self, output_size: int = 128):
        super().__init__()
        self.enc_conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)  # 32x32
        self.enc_conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # 16x16
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc_conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # 4x4 after pool
        self.enc_conv4 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)  # 2x2 after pool
        self.fc1 = nn.Linear(512 * 2 * 2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = self.max_pool(x)
        x = F.relu(self.enc_conv3(x))
        x = self.max_pool(x)
        x = F.relu(self.enc_conv4(x))
        return self.fc1(x.flatten(1))


class SiameseNetwork(nn.Module):
    def __init__(self, embedding_net: nn.Module | None = None):
        super().__init__()
        self.embedding_net = embedding_net if embedding_net is not None else EmbeddingNet()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        return self.get_embedding(x1), self.get_embedding(x2)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding_net(x)


class ContrastiveLoss(nn.Module):
    """Contrastive loss; ``target`` is 1 for corresponding pairs and 0 otherwise.

    If ``hard_negatives`` are given, the loss of a negative pair uses whichever of the
    pair's own negative and the mined hard negative is closer to the anchor.
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
        self.eps = 1e-9

    def _negative_loss(self, sq_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        hinge = F.relu(self.margin - (sq_dist + self.eps).sqrt())
        return 0.5 * (1 - target).float() * hinge.pow(2)

    def forward(self, output1, output2, target, hard_negatives=None, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)
        positive_losses = 0.5 * target.float() * distances
        negative_losses = self._negative_loss(distances, target)
        if hard_negatives is not None:
            hard_distances = (output1 - hard_negatives).pow(2).sum(1)
            negative_losses = torch.where(
                hard_distances < distances,
                self._negative_loss(hard_distances, target),
                negative_losses,
            )
        total = positive_losses + negative_losses
        return total.mean() if size_average else total.sum()


def mine_hard_negatives(anchors: torch.Tensor, others: torch.Tensor, target: torch.Tensor):
    """For each anchor, the closest embedding in ``others`` within the batch.

    Rows of positive anchors are masked out entirely (as in the original training code);
    their hard negative is then irrelevant because the loss ignores it for positives.
    """
    dist = torch.cdist(anchors, others)
    mask = target.view(-1, 1).expand(dist.size()).bool()
    dist = dist.masked_fill(mask, float("inf"))
    return others[torch.argmin(dist, dim=1)]
