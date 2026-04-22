import torch
from torch.nn import Module, Parameter

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        embedding_matrix = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)

        self.embedding_matrix = Parameter(
            torch.nn.init.trunc_normal_(embedding_matrix, mean=0.0, std=1, a=-3, b=3)
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding_matrix[token_ids]
