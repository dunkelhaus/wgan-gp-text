"""CUDA-only vector models for Wasserstein GAN training."""

import math

import torch
from torch import nn


def require_cuda(device: str | torch.device = "cuda") -> torch.device:
    """Reject CPU selection and unavailable CUDA runtimes."""
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("Latent WGAN-GP requires a CUDA device; CPU is unsupported")
    if not torch.cuda.is_available():
        raise RuntimeError("Latent WGAN-GP requires an available CUDA runtime")
    return device


def positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def vector_batch(value: torch.Tensor, width: int | None = None) -> None:
    if value.device.type != "cuda":
        raise ValueError("Vector tensors must be on CUDA; CPU is unsupported")
    if not value.is_floating_point():
        raise ValueError("Vector tensors must have a floating-point dtype")
    if value.ndim != 2 or value.shape[0] < 1 or value.shape[1] < 1:
        raise ValueError("Expected a nonempty vector batch with shape [batch, features]")
    if width is not None and value.shape[1] != width:
        raise ValueError(f"Expected vector width {width}, got {value.shape[1]}")


def _mlp(input_dim, output_dim, hidden_dim, num_layers, negative_slope, device):
    for name, value in (("input_dim", input_dim), ("output_dim", output_dim),
                        ("hidden_dim", hidden_dim), ("num_layers", num_layers)):
        positive_int(value, name)
    if num_layers < 2:
        raise ValueError("num_layers must include at least one hidden and one output layer")
    if not math.isfinite(negative_slope) or negative_slope < 0:
        raise ValueError("negative_slope must be finite and nonnegative")
    device = require_cuda(device)
    layers = []
    for index in range(num_layers):
        last = index == num_layers - 1
        layers.append(nn.Linear(input_dim if index == 0 else hidden_dim,
                                output_dim if last else hidden_dim, device=device))
        if not last:
            layers.append(nn.LeakyReLU(negative_slope))
    return nn.Sequential(*layers)


class Generator(nn.Module):
    """Map Gaussian noise to unrestricted sentence-code vectors."""

    def __init__(self, noise_dim=100, code_dim=100, hidden_dim=512,
                 num_layers=4, negative_slope=0.2, device="cuda"):
        super().__init__()
        self.network = _mlp(noise_dim, code_dim, hidden_dim, num_layers,
                            negative_slope, device)
        self.noise_dim = noise_dim
        self.latent_dim = noise_dim
        self.code_dim = code_dim

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        vector_batch(noise, self.noise_dim)
        return self.network(noise)

    def sample_latent(self, num_samples: int, *, rng=None) -> torch.Tensor:
        positive_int(num_samples, "num_samples")
        parameter = next(self.parameters())
        require_cuda(parameter.device)
        return torch.randn(num_samples, self.noise_dim, device=parameter.device,
                           dtype=parameter.dtype, generator=rng)


class Critic(nn.Module):
    """Return one unbounded score per code, with shape [batch, 1]."""

    def __init__(self, code_dim=100, hidden_dim=512, num_layers=4,
                 negative_slope=0.2, device="cuda"):
        super().__init__()
        self.network = _mlp(code_dim, 1, hidden_dim, num_layers,
                            negative_slope, device)
        self.code_dim = code_dim

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        vector_batch(codes, self.code_dim)
        return self.network(codes)


Discriminator = Critic
