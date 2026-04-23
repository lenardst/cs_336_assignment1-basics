import torch
from einops import einsum, rearrange
from torch.nn import Module, Parameter


class Linear(Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        torch_weights = torch.empty(out_features, in_features, device=device, dtype=dtype)
        std = 2 / (in_features + out_features)

        self.w = Parameter(
            torch.nn.init.trunc_normal_(
                torch_weights, mean=0.0, std=std, a=-3 * std, b=3 * std
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(
            x,
            self.w,
            "... in_features, out_features in_features -> ... out_features",
        )


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
        d_ff: int | None = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else round(d_model * 8 / 3 / 64) * 64
        self.device = device
        self.dtype = dtype

        # SwiGLU uses two projections from d_model -> d_ff, then projects back d_ff -> d_model.
        self.w1 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.w2 = Linear(self.d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
    
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

        positions = torch.arange(0, self.max_seq_len, device=device, dtype=torch.float32)[:, None]
        k_indices = torch.arange(0, self.d_k // 2, device=device, dtype=torch.float32)
        inv_freq = self.theta ** (-2.0 * k_indices / self.d_k)
        angles = positions * inv_freq[None, :]
        self.register_buffer("r_cos", torch.cos(angles), persistent=False)
        self.register_buffer("r_sin", torch.sin(angles), persistent=False)

        if self.d_k % 2 != 0:
            raise ValueError(f"d_k must be even for RoPE, got d_k={self.d_k}")

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.d_k:
            raise ValueError(
                f"x last dimension must match d_k={self.d_k}, got x.shape[-1]={x.shape[-1]}"
            )

        token_positions = token_positions.to(device=self.r_cos.device, dtype=torch.long)
        if token_positions.numel() > 0:
            if token_positions.min() < 0 or token_positions.max() >= self.max_seq_len:
                raise IndexError(
                    f"token_positions must be in [0, {self.max_seq_len}), "
                    f"got min={int(token_positions.min())}, max={int(token_positions.max())}"
                )

        trunc_r_cos = self.r_cos[token_positions].to(device=x.device, dtype=x.dtype)
        trunc_r_sin = self.r_sin[token_positions].to(device=x.device, dtype=x.dtype)
        x_pairs = x.reshape(x.shape[:-1] + (self.d_k // 2, 2))
        x_even = x_pairs[..., 0]
        x_odd = x_pairs[..., 1]

        while trunc_r_cos.ndim < x_even.ndim:
            trunc_r_cos = trunc_r_cos.unsqueeze(-3)
            trunc_r_sin = trunc_r_sin.unsqueeze(-3)

        rotated_even = x_even * trunc_r_cos - x_odd * trunc_r_sin
        rotated_odd = x_even * trunc_r_sin + x_odd * trunc_r_cos
        return torch.stack((rotated_even, rotated_odd), dim=-1).reshape_as(x)

def softmax(in_features: torch.Tensor, dim: int) -> torch.Tensor:
    shifted = in_features - torch.amax(in_features, dim=dim, keepdim=True)
    exp_shifted = torch.exp(shifted)
    return exp_shifted / torch.sum(exp_shifted, dim=dim, keepdim=True)

def run_scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask:torch.Tensor = None):
    denominator = q.shape[-1] ** 0.5
    pre_sm = einsum(
        q, k, 'batch_size ... n d_k, batch_size ... m d_k -> batch_size ... n m'
        )/denominator
    if mask is not None:
        pre_sm = pre_sm.masked_fill(mask == 0, float('-inf'))
    attention = einsum(
        softmax(pre_sm, dim=-1),
        v,
        'batch_size ... n m, batch_size ... m d_v -> batch_size ... n d_v')
    return attention


class CausalMultiHeadSelfAttention(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.device = device
        self.dtype = dtype

        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        d_k = d_model // num_heads
        d_v = d_k

        self.wq = Linear(num_heads * d_k, d_model, device=device, dtype=dtype)
        self.wk = Linear(num_heads * d_k, d_model, device=device, dtype=dtype)
        self.wv = Linear(num_heads * d_v, d_model, device=device, dtype=dtype)
        self.wo = Linear(d_model, num_heads * d_v, device=device, dtype=dtype)

        self.rope = RotaryPositionalEmbedding(1000, d_k, 4096, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        token_positions = torch.arange(seq_len, device=x.device)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        wq_forward = rearrange(
            self.wq.forward(x), 'b t (h d) -> b h t d', h=self.num_heads
            )
        wk_forward = rearrange(
            self.wk.forward(x), 'b t (h d) -> b h t d', h=self.num_heads
            )
        wv_forward = rearrange(
            self.wv.forward(x), 'b t (h d) -> b h t d', h=self.num_heads
            )
        wq_forward = self.rope.forward(wq_forward, token_positions)
        wk_forward = self.rope.forward(wk_forward, token_positions)

        head = run_scaled_dot_product_attention(wq_forward, wk_forward, wv_forward, causal_mask)

        head_concat = rearrange(head, 'b h t d -> b t (h d)', h=self.num_heads)

        wo_forward = self.wo.forward(head_concat)
        return wo_forward


class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int = 4096,
        theta: float = 1000.0,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.eps = eps
        self.device = device
        self.dtype = dtype

        self.attn = CausalMultiHeadSelfAttention(d_model, num_heads, device = device, dtype = dtype)
        self.attn.rope = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len, device = device)

        self.ln1 = RMSNorm(d_model, eps = eps, device = device, dtype = dtype)
        self.ln2 = RMSNorm(d_model, eps = eps, device = device, dtype = dtype)

        self.ffn = PositionwiseFeedForward(d_model, d_ff, device = device, dtype = dtype)

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        step_1_x = x + self.attn.forward(self.ln1.forward(x))
        return step_1_x + self.ffn.forward(self.ln2.forward(step_1_x))


class TransformerLM(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.device = device
        self.dtype = dtype

        self.token_embedding = Embedding(vocab_size, d_model, device = device, dtype = dtype)

        self.norm = RMSNorm(d_model, device = device, dtype = dtype)
        self.linear = Linear(d_model, vocab_size, device = device, dtype = dtype)

        self.transformers = torch.nn.ModuleList([TransformerBlock(
            d_model, num_heads, d_ff, context_length, rope_theta, device = device, dtype = dtype) for _ in range(num_layers)])

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        x_transformed = self.token_embedding.forward(in_indices)
        for transformer in self.transformers:
            x_transformed = transformer.forward(x_transformed)
        logits = self.linear.forward((self.norm.forward(x_transformed)))
        return logits
