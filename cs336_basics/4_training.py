import importlib
import math
import os
from collections.abc import Callable
from re import M
from typing import IO, BinaryIO
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

def cosine_lr_wup(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate

    if it <= cosine_cycle_iters:
        phase = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        cosine_value = 0.5 * (1 + math.cos(math.pi * phase))
        return min_learning_rate + cosine_value * (max_learning_rate - min_learning_rate)

    return min_learning_rate


def gradient_clipping(parameters, max_l2_norm: float) -> None:
    grads = [p.grad for p in parameters if p.grad is not None]
    if len(grads) == 0:
        return

    total_norm_sq = sum(torch.sum(g * g) for g in grads)
    total_norm = torch.sqrt(total_norm_sq)

    if total_norm <= max_l2_norm:
        return

    scale = max_l2_norm / (total_norm + 1e-6)
    for g in grads:
        g.mul_(scale)

def get_batch(x, batch_size, context_length, device):
    data = torch.from_numpy(x)
    
    indx = torch.randint(0, len(data) - context_length, (batch_size,))
    
    ipt_seq = torch.stack([data[i : i + context_length] for i in indx])
    targets = torch.stack([data[i + 1 : i + context_length + 1] for i in indx])
    
    return ipt_seq.to(device), targets.to(device)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint = torch.load(src, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]

