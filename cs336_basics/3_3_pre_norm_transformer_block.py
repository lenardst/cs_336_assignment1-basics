import torch
from einops import einsum

import importlib

Linear = importlib.import_module("cs336_basics.3_1_linear").Linear

class RMSNorm(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        
        self.g = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms_a = torch.sqrt(einsum(x, x, 'batch sequence_len d_model, batch sequence_len d_model -> batch sequence_len') / self.d_model + + self.eps)
        rms_norm = einsum(x, 1.0 / rms_a, self.g, 'batch sequence_len d_model, batch sequence_len, d_model -> batch sequence_len d_model')
        return rms_norm.to(in_dtype)

def silu(x):
    return x * torch.sigmoid(x)

class PositionwiseFeedForward(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = round(d_model * 8/3  / 64) * 64
        self.device = device
        self.dtype = dtype

        self.w1 = Linear(self.d_ff, d_model, device=device,
        dtype=dtype)
        self.w2 = Linear(d_model, self.d_ff, device=device,
        dtype=dtype)
        self.w3 = Linear(self.d_ff, d_model, device=device,
        dtype=dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = einsum(
            silu(self.w1.forward(x)),
            self.w3.forward(x),
            '... d_ff, ... d_ff -> ... d_ff'
        )
        return self.w2.forward(gated)


class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        return null




