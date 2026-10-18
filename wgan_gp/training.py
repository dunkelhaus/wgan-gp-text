"""Vector gradient penalty and isolated CUDA optimization steps."""

from contextlib import contextmanager
import math

import torch
from torch import nn

from .models import Generator, positive_int, vector_batch


@contextmanager
def frozen_parameters(module: nn.Module):
    """Freeze weights, retain input gradients, and restore flags on exit."""
    parameters = tuple(module.parameters())
    flags = tuple(parameter.requires_grad for parameter in parameters)
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield module
    finally:
        for parameter, flag in zip(parameters, flags):
            parameter.requires_grad_(flag)


def _finite(value: torch.Tensor, name: str) -> None:
    if value.device.type != "cuda":
        raise ValueError(f"{name} must be on CUDA")
    if not torch.isfinite(value).all().item():
        raise FloatingPointError(f"Non-finite {name}")


def _scores(scores: torch.Tensor) -> None:
    if scores.ndim not in (1, 2) or scores.shape[0] < 1:
        raise ValueError("Expected one critic score per example")
    if scores.ndim == 2 and scores.shape[1] != 1:
        raise ValueError("Expected critic scores with shape [batch, 1]")
    _finite(scores, "critic scores")


def critic_loss(real_scores, fake_scores, penalty, gp_weight=10.0):
    """Return mean fake minus mean real plus weighted gradient penalty."""
    if not math.isfinite(gp_weight) or gp_weight <= 0:
        raise ValueError("gp_weight must be finite and positive")
    _scores(real_scores)
    _scores(fake_scores)
    _finite(penalty, "gradient penalty")
    if penalty.ndim != 0:
        raise ValueError("Gradient penalty must be scalar")
    return fake_scores.mean() - real_scores.mean() + gp_weight * penalty


def generator_loss(fake_scores):
    """Return negative mean fake score."""
    _scores(fake_scores)
    return -fake_scores.mean()


def gradient_penalty(critic, real_codes, fake_codes, *, alpha=None, rng=None):
    """Return unweighted GP with a differentiable critic-gradient graph."""
    vector_batch(real_codes)
    vector_batch(fake_codes)
    if (real_codes.shape != fake_codes.shape or real_codes.device != fake_codes.device
            or real_codes.dtype != fake_codes.dtype):
        raise ValueError("Real and fake codes must share shape, CUDA device, and dtype")
    _finite(real_codes, "real codes")
    _finite(fake_codes, "fake codes")
    if alpha is None:
        alpha = torch.rand(real_codes.shape[0], 1, device=real_codes.device,
                           dtype=real_codes.dtype, generator=rng)
    if (alpha.shape != (real_codes.shape[0], 1) or alpha.device != real_codes.device
            or alpha.dtype != real_codes.dtype):
        raise ValueError("alpha must have shape [batch, 1] and match code device/dtype")
    _finite(alpha, "interpolation coefficients")
    if ((alpha < 0) | (alpha > 1)).any().item():
        raise ValueError("alpha must be in [0, 1]")
    interpolated = (alpha.detach() * real_codes.detach()
                    + (1 - alpha.detach()) * fake_codes.detach()).requires_grad_(True)
    scores = critic(interpolated)
    _scores(scores)
    gradients, = torch.autograd.grad(scores, interpolated,
                                    grad_outputs=torch.ones_like(scores), create_graph=True)
    penalty = (gradients.norm(2, dim=1) - 1).square().mean()
    _finite(penalty, "gradient penalty")
    return penalty


def _validate_update(generator, critic, optimizer, owner):
    generator_ids = {id(p) for p in generator.parameters()}
    critic_ids = {id(p) for p in critic.parameters()}
    if generator_ids & critic_ids:
        raise ValueError("Generator and critic cannot share parameters")
    expected = {id(p) for p in owner.parameters() if p.requires_grad}
    actual = {id(p) for group in optimizer.param_groups for p in group["params"]}
    if not expected or actual != expected:
        raise ValueError("Optimizer must own exactly the trainable phase parameters")
    for module in (generator, critic):
        for parameter in module.parameters():
            _finite(parameter, "model parameters")


def _step(module, optimizer):
    gradients = [p.grad for p in module.parameters() if p.grad is not None]
    if not gradients:
        raise RuntimeError("Update produced no parameter gradients")
    for gradient in gradients:
        _finite(gradient, "parameter gradients")
    norm = torch.stack([gradient.norm() for gradient in gradients]).norm()
    _finite(norm, "gradient norm")
    optimizer.step()
    for parameter in module.parameters():
        _finite(parameter, "updated parameters")
    return norm.item()


def critic_step(generator: Generator, critic, optimizer, real_codes, *,
                gp_weight=10.0, noise=None, rng=None):
    """Update only the critic; detach encoder codes and generated codes."""
    _validate_update(generator, critic, optimizer, critic)
    vector_batch(real_codes, generator.code_dim)
    generator.zero_grad(set_to_none=True)
    critic.zero_grad(set_to_none=True)
    optimizer.zero_grad(set_to_none=True)
    with frozen_parameters(generator), torch.no_grad():
        fake_codes = generator(generator.sample_latent(real_codes.shape[0], rng=rng)
                               if noise is None else noise.detach())
    real = real_codes.detach()
    fake = fake_codes.detach()
    real_scores, fake_scores = critic(real), critic(fake)
    penalty = gradient_penalty(critic, real, fake, rng=rng)
    loss = critic_loss(real_scores, fake_scores, penalty, gp_weight)
    _finite(loss, "critic loss")
    loss.backward()
    norm = _step(critic, optimizer)
    return {"critic_loss": loss.item(), "gradient_penalty": penalty.item(),
            "real_score": real_scores.mean().item(), "fake_score": fake_scores.mean().item(),
            "gradient_norm": norm}


def generator_step(generator: Generator, critic, optimizer, batch_size, *, noise=None, rng=None):
    """Update only the generator while retaining gradients through the critic."""
    positive_int(batch_size, "batch_size")
    _validate_update(generator, critic, optimizer, generator)
    generator.zero_grad(set_to_none=True)
    critic.zero_grad(set_to_none=True)
    optimizer.zero_grad(set_to_none=True)
    with frozen_parameters(critic):
        fake = generator(generator.sample_latent(batch_size, rng=rng)
                         if noise is None else noise.detach())
        if fake.shape[0] != batch_size:
            raise ValueError("Noise batch size must match batch_size")
        loss = generator_loss(critic(fake))
        _finite(loss, "generator loss")
        loss.backward()
        norm = _step(generator, optimizer)
    return {"generator_loss": loss.item(), "gradient_norm": norm}


class Trainer:
    """Run explicit critic and generator phases without sentence-model logic."""

    def __init__(self, generator, critic, gen_optimizer, critic_optimizer, *,
                 gp_weight=10.0, critic_iterations=5, generator_iterations=1):
        positive_int(critic_iterations, "critic_iterations")
        positive_int(generator_iterations, "generator_iterations")
        if not math.isfinite(gp_weight) or gp_weight <= 0:
            raise ValueError("gp_weight must be finite and positive")
        self.generator, self.critic = generator, critic
        self.gen_optimizer, self.critic_optimizer = gen_optimizer, critic_optimizer
        self.gp_weight = gp_weight
        self.critic_iterations = critic_iterations
        self.generator_iterations = generator_iterations
        self.critic_updates = self.generator_updates = 0

    def critic_step(self, real_codes, *, noise=None, rng=None):
        result = critic_step(self.generator, self.critic, self.critic_optimizer,
                             real_codes, gp_weight=self.gp_weight, noise=noise, rng=rng)
        self.critic_updates += 1
        return result

    def generator_step(self, batch_size, *, noise=None, rng=None):
        result = generator_step(self.generator, self.critic, self.gen_optimizer,
                                batch_size, noise=noise, rng=rng)
        self.generator_updates += 1
        return result

    def train_cycle(self, real_codes, *, rng=None):
        """Reuse one detached real batch; draw fresh noise for every update."""
        return {
            "critic": [self.critic_step(real_codes, rng=rng)
                       for _ in range(self.critic_iterations)],
            "generator": [self.generator_step(real_codes.shape[0], rng=rng)
                          for _ in range(self.generator_iterations)],
        }
