from importlib import import_module

import torch

SGD = import_module("cs336_basics.4_training").SGD


def run_learning_rate_tuning() -> None:
    torch.manual_seed(0)
    for lr in (1e1, 1e2, 1e3):
        w = torch.nn.Parameter(5 * torch.randn((10, 10)))
        opt = SGD([w], lr=lr)
        print(f"lr={lr}")
        for _ in range(10):
            opt.zero_grad()
            loss = (w**2).mean()
            print(loss.item())
            loss.backward()
            opt.step()


if __name__ == "__main__":
    run_learning_rate_tuning()
