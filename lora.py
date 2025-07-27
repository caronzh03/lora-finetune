import torch
import math


class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.A = torch.nn.Parameter(torch.empty(in_dim, rank))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))  # similar to standard weight initialization
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha
        self.rank = rank
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x

    def __repr__(self):
        return (f"{self.__class__.__name__}(in_dim={self.in_dim}, out_dim={self.out_dim}, "
                f"rank={self.rank}, alpha={self.alpha})")


class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)

