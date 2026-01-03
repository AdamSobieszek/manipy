from typing import Any, Dict, Iterable
import math

import torch
from torch.optim.lr_scheduler import LambdaLR

from .traversal_adamw_impl import TraversalAdamW


def _named_param_groups_for_adamw(model: torch.nn.Module, weight_decay: float) -> Iterable[Dict[str, Any]]:
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_bias = n.endswith(".bias")
        is_norm = any(s in n.lower() for s in ["norm", "ln", "bn", "layernorm", "batchnorm", "groupnorm"])
        (no_decay if (is_bias or is_norm) else decay).append(p)
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups if groups else [{"params": model.parameters(), "weight_decay": weight_decay}]


def _build_warmup_cosine(optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int, min_lr_frac: float):
    min_lr_frac = max(0.0, min(1.0, float(min_lr_frac)))

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_frac + (1.0 - min_lr_frac) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def configure_optimizers(model: Any, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    FOB expects a function with this signature.
    We return a Lightning-style dict: {"optimizer": opt, "lr_scheduler": {...}}.
    """
    lr = float(config.get("learning_rate", 1e-3))
    wd = float(config.get("weight_decay", 0.0))
    beta1 = float(config.get("beta1", 0.9))
    beta2 = float(config.get("beta2", 0.999))
    eps = float(config.get("eps", 1e-8))

    traversal_kwargs = dict(
        traversal_mode=config.get("traversal_mode", "none"),
        traversal_dims=config.get("traversal_dims", None),
        traversal_eps=float(config.get("traversal_eps", 1e-12)),
        traversal_gain_clamp=config.get("traversal_gain_clamp", None),
        moment1_source=config.get("moment1_source", "trav"),
        moment2_source=config.get("moment2_source", "trav"),
    )
    groups = _named_param_groups_for_adamw(model, wd)
    opt = TraversalAdamW(groups, lr=lr, betas=(beta1, beta2), eps=eps, **traversal_kwargs)

    min_lr_frac = float(config.get("minimal_lr_frac", 0.01))
    warmup_frac = float(config.get("warmup_frac", 0.01))
    total_steps = config.get("total_steps", None)

    if total_steps is None:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=int(config.get("max_epochs", 100)),
            eta_min=lr * min_lr_frac,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val/loss",
            },
        }

    warmup_steps = max(1, int(round(total_steps * warmup_frac)))
    scheduler = _build_warmup_cosine(opt, total_steps=total_steps, warmup_steps=warmup_steps, min_lr_frac=min_lr_frac)
    return {
        "optimizer": opt,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
            "monitor": "val/loss",
        },
    }
