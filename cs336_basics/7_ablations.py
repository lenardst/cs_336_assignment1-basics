import importlib

import torch
from einops import rearrange


transformer_module = importlib.import_module("cs336_basics.3_transformer_lm")

Linear = getattr(transformer_module, "Linear")
Embedding = getattr(transformer_module, "Embedding")
RMSNorm = getattr(transformer_module, "RMSNorm")
PositionwiseFeedForward = getattr(transformer_module, "PositionwiseFeedForward")
CausalMultiHeadSelfAttention = getattr(transformer_module, "CausalMultiHeadSelfAttention")
run_scaled_dot_product_attention = getattr(transformer_module, "run_scaled_dot_product_attention")
silu = getattr(transformer_module, "silu")


class SiLUFeedForward(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None) -> None:
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2.forward(silu(self.w1.forward(x)))


class NoPosEmbCausalMultiHeadSelfAttention(torch.nn.Module):
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

        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        d_k = d_model // num_heads
        d_v = d_k

        self.wq = Linear(num_heads * d_k, d_model, device=device, dtype=dtype)
        self.wk = Linear(num_heads * d_k, d_model, device=device, dtype=dtype)
        self.wv = Linear(num_heads * d_v, d_model, device=device, dtype=dtype)
        self.wo = Linear(d_model, num_heads * d_v, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        q = rearrange(self.wq.forward(x), "b t (h d) -> b h t d", h=self.num_heads)
        k = rearrange(self.wk.forward(x), "b t (h d) -> b h t d", h=self.num_heads)
        v = rearrange(self.wv.forward(x), "b t (h d) -> b h t d", h=self.num_heads)
        head = run_scaled_dot_product_attention(q, k, v, causal_mask)
        head_concat = rearrange(head, "b h t d -> b t (h d)", h=self.num_heads)
        return self.wo.forward(head_concat)


class LayerNormAblationBlock(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int = 4096,
        theta: float = 1000.0,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.attn = CausalMultiHeadSelfAttention(
            d_model,
            num_heads,
            rope_theta=theta,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
        )
        self.ffn = PositionwiseFeedForward(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn.forward(x)
        return x + self.ffn.forward(x)


class PostNormBlock(torch.nn.Module):
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
        self.attn = CausalMultiHeadSelfAttention(
            d_model,
            num_heads,
            rope_theta=theta,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
        )
        self.ffn = PositionwiseFeedForward(d_model, d_ff, device=device, dtype=dtype)
        self.ln1 = RMSNorm(d_model, eps=eps, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, eps=eps, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln1.forward(x + self.attn.forward(x))
        return self.ln2.forward(x + self.ffn.forward(x))


class NoPosEmbBlock(torch.nn.Module):
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
        self.attn = NoPosEmbCausalMultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            device=device,
            dtype=dtype,
        )
        self.ffn = PositionwiseFeedForward(d_model, d_ff, device=device, dtype=dtype)
        self.ln1 = RMSNorm(d_model, eps=eps, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, eps=eps, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn.forward(self.ln1.forward(x))
        return x + self.ffn.forward(self.ln2.forward(x))


class SiLUAblationBlock(torch.nn.Module):
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
        self.attn = CausalMultiHeadSelfAttention(
            d_model,
            num_heads,
            rope_theta=theta,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
        )
        self.ln1 = RMSNorm(d_model, eps=eps, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, eps=eps, device=device, dtype=dtype)

        silu_d_ff = max(64, round((3 * d_ff / 2) / 64) * 64)
        self.ffn = SiLUFeedForward(d_model=d_model, d_ff=silu_d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn.forward(self.ln1.forward(x))
        return x + self.ffn.forward(self.ln2.forward(x))


class _BaseTransformerLM(torch.nn.Module):
    block_cls = None
    final_norm = True

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
        if self.block_cls is None:
            raise ValueError("block_cls must be set by subclasses")

        self.context_length = context_length
        self.token_embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.norm = RMSNorm(d_model, device=device, dtype=dtype) if self.final_norm else torch.nn.Identity()
        self.linear = Linear(d_model, vocab_size, device=device, dtype=dtype)
        self.transformers = torch.nn.ModuleList(
            [
                self.block_cls(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding.forward(in_indices)
        for transformer in self.transformers:
            x = transformer.forward(x)
        return self.linear.forward(self.norm.forward(x))


class LayerNormAblationTransformerModel(_BaseTransformerLM):
    block_cls = LayerNormAblationBlock
    final_norm = False


class PreNormAblationTransformerModel(_BaseTransformerLM):
    block_cls = PostNormBlock


class NoPosEmbTransformerModel(_BaseTransformerLM):
    block_cls = NoPosEmbBlock


class SwiGLUAblationTransformerModel(_BaseTransformerLM):
    block_cls = SiLUAblationBlock


layer_norm_ablation_transformer_model = LayerNormAblationTransformerModel
pre_norm_ablation_transformer_model = PreNormAblationTransformerModel
no_pos_emb_transformer_model = NoPosEmbTransformerModel
swiglu_ablation_transformer_model = SwiGLUAblationTransformerModel
