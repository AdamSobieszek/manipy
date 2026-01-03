import math
from contextlib import contextmanager
from typing import Callable, Iterable, Optional, Sequence, Tuple, Union

import torch
from torch.optim import AdamW

DimSpec = Optional[Union[Tuple[int, ...], Sequence[int], Callable[[torch.Tensor], Tuple[int, ...]]]]

def _resolve_reduce_dims(x: torch.Tensor, mode: str, dims: DimSpec) -> Tuple[int, ...]:
    if dims is not None:
        if callable(dims):
            out = dims(x)
            return tuple(out) if isinstance(out, (list, tuple)) else (int(out),)
        return tuple(dims)
    if mode == "none":
        return ()
    if mode == "full":
        return tuple(range(x.dim()))
    if mode in ("row", "per_last"):
        return (x.dim() - 1,) if x.dim() > 0 else ()
    if mode in ("col", "per_first"):
        return (0,) if x.dim() > 0 else ()
    if mode in ("batch", "except_first"):
        return tuple(range(1, x.dim()))
    raise ValueError(f"Unknown traversal mode: {mode}")

def canonical_traversal(
    g: torch.Tensor,
    mode: str = "none",
    dims: DimSpec = None,
    eps: float = 1e-12,
    gain_clamp: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if g.is_sparse:
        raise RuntimeError("TraversalAdamW does not support sparse gradients.")
    reduce_dims = _resolve_reduce_dims(g, mode=mode, dims=dims)
    if len(reduce_dims) == 0:
        norm_sq = g.pow(2)
    else:
        norm_sq = g.pow(2).sum(dim=reduce_dims, keepdim=True)
    norm_sq = norm_sq.add(eps)
    inv = norm_sq.reciprocal()
    if gain_clamp is not None:
        inv = inv.clamp(max=gain_clamp)
    T = g * inv
    return T, norm_sq

@contextmanager
def _temporary_grad_swap(params: Iterable[torch.nn.Parameter], new_grads: Iterable[torch.Tensor]):
    saved = []
    for p, ng in zip(params, new_grads):
        saved.append((p, p.grad))
        p.grad = ng
    try:
        yield
    finally:
        for p, old in saved:
            p.grad = old

class TraversalAdamW(AdamW):
    traversal_default_config = dict(
        traversal_mode="none",
        traversal_dims=None,
        traversal_eps=1e-12,
        traversal_gain_clamp=None,
        moment1_source="trav",
        moment2_source="trav",
        keep_last_traversal=False,
    )

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        maximize: bool = False,
        foreach: Optional[bool] = None,
        capturable: bool = False,
        differentiable: bool = False,
        fused: Optional[bool] = None,
        traversal_mode: str = "none",
        traversal_dims: DimSpec = None,
        traversal_eps: float = 1e-12,
        traversal_gain_clamp: Optional[float] = None,
        moment1_source: str = "trav",
        moment2_source: str = "trav",
        keep_last_traversal: bool = False,
    ):
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=maximize,
            foreach=foreach,
            capturable=capturable,
            differentiable=differentiable,
            fused=fused,
        )

        self._traversal_defaults = self.traversal_default_config.copy()
        self._traversal_defaults.update(
            dict(
                traversal_mode=traversal_mode,
                traversal_dims=traversal_dims,
                traversal_eps=traversal_eps,
                traversal_gain_clamp=traversal_gain_clamp,
                moment1_source=moment1_source,
                moment2_source=moment2_source,
                keep_last_traversal=keep_last_traversal,
            )
        )
        for k, v in self._traversal_defaults.items():
            self.defaults.setdefault(k, v)
        self._ensure_group_config()

    def _ensure_group_config(self):
        for group in self.param_groups:
            for k, v in self._traversal_defaults.items():
                group.setdefault(k, v)

    @torch.no_grad()
    def step(self, closure=None):
        self._ensure_group_config()
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        all_groups_same_source = True
        all_groups_grad = True
        all_groups_trav = True
        for group in self.param_groups:
            m1 = group["moment1_source"]
            m2 = group["moment2_source"]
            same = (m1 == m2)
            all_groups_same_source &= same
            all_groups_grad &= (m1 == "grad" and m2 == "grad")
            all_groups_trav &= (m1 == "trav" and m2 == "trav")

        if all_groups_same_source and all_groups_grad:
            return super().step(closure=None)

        if all_groups_same_source and all_groups_trav:
            params_with_grad = []
            travs = []
            for group in self.param_groups:
                mode = group["traversal_mode"]
                dims = group["traversal_dims"]
                eps_t = group["traversal_eps"]
                clamp = group["traversal_gain_clamp"]
                keep_t = group["keep_last_traversal"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad.detach()
                    T, D = canonical_traversal(g, mode=mode, dims=dims, eps=eps_t, gain_clamp=clamp)
                    params_with_grad.append(p)
                    travs.append(T)
                    if keep_t:
                        st = self.state[p]
                        st["last_traversal"] = T.detach()
                        st["last_trav_norm2"] = D.detach()
            with _temporary_grad_swap(params_with_grad, travs):
                super().step(closure=None)
            return loss

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]
            amsgrad = group["amsgrad"]
            maximize = group["maximize"]
            eps_adam = group["eps"]

            mode = group["traversal_mode"]
            dims = group["traversal_dims"]
            eps_trav = group["traversal_eps"]
            clamp = group["traversal_gain_clamp"]

            m1_src = group["moment1_source"]
            m2_src = group["moment2_source"]
            keep_t = group["keep_last_traversal"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                if g.is_sparse:
                    raise RuntimeError("TraversalAdamW does not support sparse gradients.")

                g_eff = -g if maximize else g
                T, D = canonical_traversal(g_eff, mode=mode, dims=dims, eps=eps_trav, gain_clamp=clamp)

                state = self.state[p]
                if keep_t:
                    state["last_traversal"] = T.detach()
                    state["last_trav_norm2"] = D.detach()

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                step_t = state["step"]

                g_m = T if m1_src == "trav" else g_eff
                g_v = T if m2_src == "trav" else g_eff

                exp_avg.mul_(beta1).add_(g_m, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g_v, g_v, value=1.0 - beta2)

                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom_sq = max_exp_avg_sq
                else:
                    denom_sq = exp_avg_sq

                bias_correction1 = 1.0 - beta1 ** step_t
                bias_correction2 = 1.0 - beta2 ** step_t
                step_size = group["lr"] / bias_correction1
                denom = (denom_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps_adam)

                if wd != 0:
                    p.add_(p, alpha=-group["lr"] * wd)

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

class TraversalAdamWAll(TraversalAdamW):
    def __init__(self, params, **kwargs):
        kwargs.setdefault("moment1_source", "trav")
        kwargs.setdefault("moment2_source", "trav")
        super().__init__(params, **kwargs)