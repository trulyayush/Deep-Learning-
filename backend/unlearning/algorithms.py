"""PyTorch SISA sharded retraining and approximate gradient-ascent unlearning."""
from typing import List, Set
import torch
import torch.nn as nn


def gradient_ascent_unlearn(
    model: nn.Module,
    forget_loader: torch.utils.data.DataLoader,
    retain_loader: torch.utils.data.DataLoader,
    device: torch.device,
    steps: int = 10,
    lr_forget: float = 1e-3,
    lr_retain: float = 1e-3,
) -> nn.Module:
    """Bounded gradient ascent on forget-set paired with retain-set repair descent."""
    criterion = nn.CrossEntropyLoss()
    model.train()

    for _ in range(steps):
        for xf, yf in forget_loader:
            xf, yf = xf.to(device), yf.to(device)
            model.zero_grad()
            loss_f = -criterion(model(xf), yf)
            loss_f.backward()
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p -= lr_forget * p.grad

        for xr, yr in retain_loader:
            xr, yr = xr.to(device), yr.to(device)
            model.zero_grad()
            loss_r = criterion(model(xr), yr)
            loss_r.backward()
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p -= lr_retain * p.grad

    return model


def affected_sisa_shards(sample_shards: List[int], forget_indices: Set[int]) -> List[int]:
    return sorted({sample_shards[idx] for idx in forget_indices if idx < len(sample_shards)})
