"""Fork-owned CUDA latent WGAN-GP public API; no image dependencies."""

from .models import Critic, Discriminator, Generator, require_cuda
from .training import (
    Trainer,
    critic_loss,
    critic_step,
    frozen_parameters,
    generator_loss,
    generator_step,
    gradient_penalty,
)

__all__ = [
    "Critic", "Discriminator", "Generator", "Trainer", "critic_loss", "critic_step",
    "frozen_parameters", "generator_loss", "generator_step", "gradient_penalty", "require_cuda",
]
