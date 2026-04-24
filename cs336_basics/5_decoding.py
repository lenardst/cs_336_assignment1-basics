import importlib
import torch


transformer_module = importlib.import_module("cs336_basics.3_transformer_lm")
TransformerLM = getattr(transformer_module, "TransformerLM")
base_softmax = getattr(transformer_module, "softmax")


def softmax_with_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    return base_softmax(logits / temperature, dim=-1)


def top_p_filter(probabilities: torch.Tensor, top_p: float = 1.0) -> torch.Tensor:
    if not 0 < top_p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    if top_p == 1.0:
        return probabilities

    sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    keep_mask = cumulative <= top_p
    keep_mask[0] = True

    filtered_sorted = sorted_probs * keep_mask.to(sorted_probs.dtype)
    filtered = torch.zeros_like(probabilities)
    filtered.scatter_(0, sorted_indices, filtered_sorted)
    return filtered / filtered.sum()


@torch.no_grad()
def sample_next_token(
    model: TransformerLM,
    tokens: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> int:
    if tokens.ndim != 1:
        raise ValueError(f"tokens must be shape (seq_len,), got {tuple(tokens.shape)}")

    device = next(model.parameters()).device
    input_tokens = tokens[-model.context_length :].to(device=device, dtype=torch.long).unsqueeze(0)

    logits = model(input_tokens)[0, -1]
    probs = softmax_with_temperature(logits, temperature=temperature)
    probs = top_p_filter(probs, top_p=top_p)
    next_token = torch.multinomial(probs, num_samples=1)
    return int(next_token.item())


@torch.no_grad()
def decode(
    model: TransformerLM,
    prompt_tokens: list[int] | torch.Tensor,
    eos_token_id: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> list[int]:
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be >= 0, got {max_new_tokens}")

    if isinstance(prompt_tokens, torch.Tensor):
        generated = prompt_tokens.to(dtype=torch.long).flatten().tolist()
    else:
        generated = [int(token) for token in prompt_tokens]

    if len(generated) == 0:
        raise ValueError("prompt_tokens must contain at least one token")

    model.eval()
    for _ in range(max_new_tokens):
        token_tensor = torch.tensor(generated, dtype=torch.long)
        next_token = sample_next_token(
            model=model,
            tokens=token_tensor,
            temperature=temperature,
            top_p=top_p,
        )
        generated.append(next_token)
        if next_token == eos_token_id:
            break

    return generated
