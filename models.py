"""Compatibility imports for the fork's supported latent models.

The old image constructor signatures are no longer supported.
"""

from wgan_gp.models import Critic, Discriminator, Generator

__all__ = ["Critic", "Discriminator", "Generator"]
