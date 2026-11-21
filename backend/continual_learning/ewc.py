"""PyTorch Elastic Weight Consolidation (EWC) regularization utilities."""
from typing import Dict
import torch
import torch.nn as nn


def compute_diagonal_fisher(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    model.eval()
    fisher = {
        name: torch.zeros_like(param, device=device)
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    criterion = nn.CrossEntropyLoss()
    sample_count = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        model.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher[name] += param.grad.detach() ** 2 * x.size(0)
        sample_count += x.size(0)

    if sample_count > 0:
        for name in fisher:
            fisher[name] /= float(sample_count)
    return fisher


def ewc_penalty(
    model: nn.Module,
    fisher: Dict[str, torch.Tensor],
    anchor_params: Dict[str, torch.Tensor],
    ewc_lambda: float,
) -> torch.Tensor:
    penalty = torch.tensor(0.0, device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        if name in fisher and name in anchor_params:
            penalty = penalty + (fisher[name] * (param - anchor_params[name]) ** 2).sum()
    return 0.5 * ewc_lambda * penalty
