import torch
from torch.nn import Module, Parameter
from einops import einsum

class Linear(Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        torch_weights = torch.empty(out_features, in_features, device=device, dtype=dtype)
        std = 2 / (in_features + out_features)
        
        self.w = Parameter(
            torch.nn.init.trunc_normal_(
                torch_weights, mean=0.0, std=std, a= -3 * std, b = 3 * std)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(
            x,
            self.w,
            '... in_features, out_features in_features -> ... out_features')
