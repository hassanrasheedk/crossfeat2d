"""Training loop: contrastive loss with in-batch hard-negative mining."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from crossfeat2d.model import ContrastiveLoss, SiameseNetwork, mine_hard_negatives


def train(
    model: SiameseNetwork,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    *,
    epochs: int = 10,
    lr: float = 1e-3,
    lr_step: int = 8,
    lr_gamma: float = 0.1,
    similarity_threshold: float = 0.5,
    device: torch.device | str = "cpu",
    log: Callable[[str], None] = print,
    log_interval: int = 10,
) -> list[dict]:
    """Train ``model`` in place and return per-epoch metrics.

    Defaults follow the thesis notebooks: Adam (lr 1e-3), StepLR (step 8, gamma 0.1),
    margin-1 contrastive loss, 10 epochs. Hard negatives follow Robinson et al.
    (ICLR 2021): the closest non-matching embedding in the batch. Validation uses the
    plain loss without hard negatives, as in the notebooks.
    """
    loss_fn = ContrastiveLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_step, gamma=lr_gamma)
    history = []

    for epoch in range(epochs):
        model.train()
        total_loss = total_acc = 0.0
        for batch_idx, ((img1, img2), target) in enumerate(train_loader):
            img1, img2, target = img1.to(device), img2.to(device), target.to(device)
            optimizer.zero_grad()
            out1, out2 = model(img1, img2)
            hard_negatives = mine_hard_negatives(out1, out2, target)
            with torch.no_grad():
                predicted = (F.pairwise_distance(out1, out2) < similarity_threshold).float()
                total_acc += (predicted == target).float().mean().item()
            loss = loss_fn(out1, out2, target, hard_negatives)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if batch_idx % log_interval == 0:
                log(f"epoch {epoch} batch {batch_idx} loss {loss.item():.4f}")
        scheduler.step()

        metrics = {
            "epoch": epoch,
            "train_loss": total_loss / max(len(train_loader), 1),
            "train_accuracy": total_acc / max(len(train_loader), 1),
        }
        if val_loader is not None and len(val_loader) > 0:
            metrics["val_loss"] = evaluate_loss(model, val_loader, loss_fn, device)
        log(", ".join(f"{k} {v:.4f}" for k, v in metrics.items() if k != "epoch"))
        history.append(metrics)
    return history


@torch.no_grad()
def evaluate_loss(model, loader, loss_fn, device) -> float:
    model.eval()
    total = 0.0
    for (img1, img2), target in loader:
        out1, out2 = model(img1.to(device), img2.to(device))
        total += loss_fn(out1, out2, target.to(device)).item()
    return total / len(loader)
