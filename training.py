"""Compatibility imports for the fork's supported latent training API."""

from wgan_gp.training import (
    Trainer, critic_loss, critic_step, frozen_parameters,
    generator_loss, generator_step, gradient_penalty,
)

__all__ = [
    "Trainer", "critic_loss", "critic_step", "frozen_parameters",
    "generator_loss", "generator_step", "gradient_penalty",
]
