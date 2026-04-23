from __future__ import annotations

from dataclasses import dataclass


def matmul_flops(m: int, n: int, p: int) -> int:
    """FLOPs for (m x n) @ (n x p), using 2mnp convention."""
    return 2 * m * n * p


def default_d_ff(d_model: int) -> int:
    """Match the d_ff default used in 3_transformer_lm.py."""
    return round(d_model * 8 / 3 / 64) * 64


@dataclass(frozen=True)
class ModelConfig:
    name: str
    vocab_size: int
    context_length: int
    num_layers: int
    d_model: int
    num_heads: int
    d_ff: int | None = None
    batch_size: int = 1


def transformer_lm_matmul_flops(config: ModelConfig) -> dict[str, float]:
    """
    Returns matrix-multiplication FLOPs only (forward pass).
    Token embedding is excluded because it is an index lookup.
    """
    b = config.batch_size
    t = config.context_length
    d = config.d_model
    h = config.num_heads
    d_k = d // h
    d_ff = config.d_ff if config.d_ff is not None else default_d_ff(d)
    v = config.vocab_size
    l = config.num_layers

    # Per-block components.
    q_proj = matmul_flops(b * t, d, d)
    k_proj = matmul_flops(b * t, d, d)
    v_proj = matmul_flops(b * t, d, d)
    o_proj = matmul_flops(b * t, d, d)

    attn_scores = 2 * b * h * t * t * d_k
    attn_value = 2 * b * h * t * t * d_k

    ffn_w1 = matmul_flops(b * t, d, d_ff)
    ffn_w3 = matmul_flops(b * t, d, d_ff)
    ffn_w2 = matmul_flops(b * t, d_ff, d)

    per_block = {
        "qkv_projections": q_proj + k_proj + v_proj,
        "attention_scores": attn_scores,
        "attention_value": attn_value,
        "attention_output_projection": o_proj,
        "ffn_w1_w3": ffn_w1 + ffn_w3,
        "ffn_w2": ffn_w2,
    }
    per_block_total = sum(per_block.values())

    final_output_projection = matmul_flops(b * t, d, v)

    totals = {k: v_ * l for k, v_ in per_block.items()}
    totals["final_output_projection"] = final_output_projection
    totals["total"] = sum(totals.values())

    out: dict[str, float] = {
        "name": config.name,
        "vocab_size": v,
        "context_length": t,
        "num_layers": l,
        "d_model": d,
        "num_heads": h,
        "d_ff": d_ff,
        "per_block_total": per_block_total,
        **totals,
    }

    total = out["total"]
    for key in [
        "qkv_projections",
        "attention_scores",
        "attention_value",
        "attention_output_projection",
        "ffn_w1_w3",
        "ffn_w2",
        "final_output_projection",
    ]:
        out[f"{key}_pct"] = out[key] / total
    return out


def _pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def _billions(x: float) -> str:
    return f"{x / 1e9:.2f}B"


def make_markdown_table(results: list[dict[str, float]]) -> str:
    headers = [
        "Model",
        "Total FLOPs",
        "QKV proj",
        "Attn scores",
        "Attn value",
        "Attn out proj",
        "FFN W1/W3",
        "FFN W2",
        "Final proj",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in results:
        lines.append(
            "| "
            + " | ".join(
                [
                    r["name"],
                    _billions(r["total"]),
                    _pct(r["qkv_projections_pct"]),
                    _pct(r["attention_scores_pct"]),
                    _pct(r["attention_value_pct"]),
                    _pct(r["attention_output_projection_pct"]),
                    _pct(r["ffn_w1_w3_pct"]),
                    _pct(r["ffn_w2_pct"]),
                    _pct(r["final_output_projection_pct"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    vocab_size = 50_257
    context_length = 1_024

    configs = [
        ModelConfig(
            name="Validation (48L, d=1600)",
            vocab_size=vocab_size,
            context_length=context_length,
            num_layers=48,
            d_model=1_600,
            num_heads=25,
            d_ff=4_288,
        ),
        ModelConfig(
            name="GPT-2 small",
            vocab_size=vocab_size,
            context_length=context_length,
            num_layers=12,
            d_model=768,
            num_heads=12,
        ),
        ModelConfig(
            name="GPT-2 medium",
            vocab_size=vocab_size,
            context_length=context_length,
            num_layers=24,
            d_model=1_024,
            num_heads=16,
        ),
        ModelConfig(
            name="GPT-2 large",
            vocab_size=vocab_size,
            context_length=context_length,
            num_layers=36,
            d_model=1_280,
            num_heads=20,
        ),
    ]

    results = [transformer_lm_matmul_flops(cfg) for cfg in configs]
    print("Matrix-multiplication FLOPs only (forward pass; batch_size=1).")
    print("d_ff uses model value if provided, else round((8/3)*d_model/64)*64.")
    print()
    print(make_markdown_table(results))
