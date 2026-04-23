import importlib
import math
from collections.abc import Callable
from re import M
from typing import Optional

import torch
from torch import Tensor


def cross_entropy(
    logits: Tensor,
    targets: Tensor,
) -> Tensor:
    """
    Compute average cross-entropy over all batch-like dimensions.

    `logits` must have shape (..., vocab_size), and `targets` must have shape (...).
    """
    transformer_module = importlib.import_module("cs336_basics.3_transformer_lm")
    softmax_fn = getattr(transformer_module, "softmax")

    shifted_logits = logits - torch.amax(logits, dim=-1, keepdim=True)
    probs = softmax_fn(shifted_logits, dim=-1)

    max_indices = torch.argmax(shifted_logits, dim=-1, keepdim=True)
    max_probs = probs.gather(dim=-1, index=max_indices).squeeze(-1)
    logsumexp = -torch.log(max_probs)

    target_shifted_logits = shifted_logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    losses = logsumexp - target_shifted_logits
    return losses.mean()


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, beta1, beta2, lamb, epsilon):
        defaults = {"lr": lr, "beta1": beta1, "beta2": beta2, "lamb": lamb, "epsilon": epsilon}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            lamb = group["lamb"]
            epsilon = group["epsilon"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0) + 1
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))
                grad = p.grad.data

                alpha_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                p.data -= lr * lamb * p.data
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad * grad
                p.data -= alpha_t * m / (torch.sqrt(v) + epsilon)

                state["t"] = t
                state["m"] = m
                state["v"] = v
        return loss

