import importlib

import torch


def run_learning_rate_tuning() -> None:
    training_module = importlib.import_module("cs336_basics.4_training")
    sgd_cls = getattr(training_module, "SGD")

    torch.manual_seed(0)
    for lr in (1e1, 1e2, 1e3):
        weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
        opt = sgd_cls([weights], lr=lr)
        print(f"lr={lr}")
        for _ in range(10):
            opt.zero_grad()
            loss = (weights**2).mean()
            print(loss.item())
            loss.backward()
            opt.step()


if __name__ == "__main__":
    run_learning_rate_tuning()
