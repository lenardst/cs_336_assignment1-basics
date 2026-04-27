import importlib

import torch

M = importlib.import_module("cs336_basics.3_transformer_lm")
TransformerLM = M.TransformerLM


def softmax_with_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    return M.softmax(logits / temperature, dim=-1)


def top_p_filter(probabilities: torch.Tensor, top_p: float = 1.0) -> torch.Tensor:
    if top_p >= 1.0:
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
    dev = next(model.parameters()).device
    x = tokens[-model.context_length :].to(device=dev, dtype=torch.long).unsqueeze(0)
    logits = model(x)[0, -1]
    probs = softmax_with_temperature(logits, temperature=temperature)
    probs = top_p_filter(probs, top_p=top_p)
    return int(torch.multinomial(probs, num_samples=1).item())


@torch.no_grad()
def decode(
    model: TransformerLM,
    prompt_tokens: list[int] | torch.Tensor,
    eos_token_id: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> list[int]:
    if isinstance(prompt_tokens, torch.Tensor):
        generated = prompt_tokens.to(dtype=torch.long).flatten().tolist()
    else:
        generated = [int(t) for t in prompt_tokens]

    model.eval()
    for _ in range(max_new_tokens):
        nxt = sample_next_token(model, torch.tensor(generated, dtype=torch.long), temperature, top_p)
        generated.append(nxt)
        if nxt == eos_token_id:
            break
    return generated
