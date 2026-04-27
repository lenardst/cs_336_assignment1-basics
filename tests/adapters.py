from __future__ import annotations

import importlib
import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

_TM = importlib.import_module("cs336_basics.3_transformer_lm")
_TR = importlib.import_module("cs336_basics.4_training")
_T0 = importlib.import_module("cs336_basics.2_0_bpe_tokenizer")
_T4 = importlib.import_module("cs336_basics.2_4_tokenizer")


def _set(layer: torch.nn.Module, w: Tensor) -> None:
    for name in ("w", "weight", "g", "embedding_matrix"):
        if hasattr(layer, name):
            getattr(layer, name).data = w
            return


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    linear = _TM.Linear(d_in, d_out, device=weights.device, dtype=weights.dtype)
    linear.load_state_dict({"w": weights})
    return linear(in_features)


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    emb = _TM.Embedding(vocab_size, d_model, device=weights.device, dtype=weights.dtype)
    emb.load_state_dict({"embedding_matrix": weights})
    return emb(token_ids)


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    ffn = _TM.PositionwiseFeedForward(d_model, d_ff, device=w1_weight.device, dtype=w1_weight.dtype)
    _set(ffn.w1, w1_weight)
    _set(ffn.w2, w2_weight)
    _set(ffn.w3, w3_weight)
    return ffn(in_features)


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    return _TM.run_scaled_dot_product_attention(Q, K, V, mask)


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
) -> Float[Tensor, " ... sequence_length d_model"]:
    attn = _TM.CausalMultiHeadSelfAttention(
        d_model, num_heads, device=in_features.device, dtype=in_features.dtype
    )

    class _NoRoPE(torch.nn.Module):
        def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:  # noqa: ARG002
            return x

    attn.rope = _NoRoPE()
    _set(attn.wq, q_proj_weight)
    _set(attn.wk, k_proj_weight)
    _set(attn.wv, v_proj_weight)
    _set(attn.wo, o_proj_weight)
    return attn(in_features)


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_model"]:
    attn = _TM.CausalMultiHeadSelfAttention(
        d_model, num_heads, device=in_features.device, dtype=in_features.dtype
    )
    rope = _TM.RotaryPositionalEmbedding(
        theta, d_model // num_heads, max_seq_len, device=in_features.device
    )
    if token_positions is None:
        attn.rope = rope
    else:
        pos = token_positions.to(device=in_features.device, dtype=torch.long)
        if pos.ndim > 1 and pos.shape[0] == 1:
            pos = pos.squeeze(0)

        class _FixedRoPE(torch.nn.Module):
            def __init__(self, base: torch.nn.Module, positions: Tensor):
                super().__init__()
                self.base = base
                self.register_buffer("positions", positions, persistent=False)

            def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:  # noqa: ARG002
                return self.base(x, self.positions)

        attn.rope = _FixedRoPE(rope, pos)

    _set(attn.wq, q_proj_weight)
    _set(attn.wk, k_proj_weight)
    _set(attn.wv, v_proj_weight)
    _set(attn.wo, o_proj_weight)
    return attn(in_features)


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    rope = _TM.RotaryPositionalEmbedding(theta, d_k, max_seq_len, device=in_query_or_key.device)
    return rope(in_query_or_key, token_positions)


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    block = _TM.TransformerBlock(
        d_model, num_heads, d_ff, max_seq_len, theta, device=in_features.device, dtype=in_features.dtype
    )
    block.attn.rope = _TM.RotaryPositionalEmbedding(
        theta, d_model // num_heads, max_seq_len, device=in_features.device
    )
    _set(block.attn.wq, weights["attn.q_proj.weight"])
    _set(block.attn.wk, weights["attn.k_proj.weight"])
    _set(block.attn.wv, weights["attn.v_proj.weight"])
    _set(block.attn.wo, weights["attn.output_proj.weight"])
    _set(block.ln1, weights["ln1.weight"])
    _set(block.ffn.w1, weights["ffn.w1.weight"])
    _set(block.ffn.w2, weights["ffn.w2.weight"])
    _set(block.ffn.w3, weights["ffn.w3.weight"])
    _set(block.ln2, weights["ln2.weight"])
    return block(in_features)


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    lm = _TM.TransformerLM(
        vocab_size,
        context_length,
        d_model,
        num_layers,
        num_heads,
        d_ff,
        rope_theta,
        device=in_indices.device,
        dtype=weights["token_embeddings.weight"].dtype,
    )
    _set(lm.token_embedding, weights["token_embeddings.weight"])
    for i in range(num_layers):
        p = f"layers.{i}."
        layer = lm.transformers[i]
        _set(layer.attn.wq, weights[f"{p}attn.q_proj.weight"])
        _set(layer.attn.wk, weights[f"{p}attn.k_proj.weight"])
        _set(layer.attn.wv, weights[f"{p}attn.v_proj.weight"])
        _set(layer.attn.wo, weights[f"{p}attn.output_proj.weight"])
        _set(layer.ln1, weights[f"{p}ln1.weight"])
        _set(layer.ffn.w1, weights[f"{p}ffn.w1.weight"])
        _set(layer.ffn.w2, weights[f"{p}ffn.w2.weight"])
        _set(layer.ffn.w3, weights[f"{p}ffn.w3.weight"])
        _set(layer.ln2, weights[f"{p}ln2.weight"])
    _set(lm.norm, weights["ln_final.weight"])
    _set(lm.linear, weights["lm_head.weight"])
    return lm(in_indices)


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    m = _TM.RMSNorm(d_model, eps=eps, device=weights.device, dtype=weights.dtype)
    m.load_state_dict({"g": weights})
    return m(in_features)


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    return _TM.silu(in_features)


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    return _TR.get_batch(dataset, batch_size, context_length, device)


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    return _TM.softmax(in_features, dim)


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    return _TR.cross_entropy(inputs, targets)


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    _TR.gradient_clipping(parameters, max_l2_norm)


def get_adamw_cls() -> Any:
    base = _TR.AdamW

    class AdamWAdapter(base):
        def __init__(
            self,
            params,
            lr: float = 1e-3,
            weight_decay: float = 0.0,
            betas: tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-8,
        ):
            super().__init__(
                params=params,
                lr=lr,
                beta1=betas[0],
                beta2=betas[1],
                lamb=weight_decay,
                epsilon=eps,
            )

    return AdamWAdapter


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    return _TR.cosine_lr_wup(
        it=it,
        max_learning_rate=max_learning_rate,
        min_learning_rate=min_learning_rate,
        warmup_iters=warmup_iters,
        cosine_cycle_iters=cosine_cycle_iters,
    )


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    _TR.save_checkpoint(model, optimizer, iteration, out)


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    return _TR.load_checkpoint(src, model, optimizer)


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    return _T4.Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    return _T0.train_bpe(str(input_path), vocab_size, special_tokens)
