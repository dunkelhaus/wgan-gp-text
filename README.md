# Latent Wasserstein GAN with gradient penalty

This fork provides the CUDA-only `wgan_gp` package for sentence-code training.
It retains the objective from [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028).
The implementation stays in this repository. POG imports it; POG does not copy it.
The existing [license](LICENSE) applies.

## Supported models

- `Generator(noise_dim=100, code_dim=100, hidden_dim=512, num_layers=4,
  negative_slope=0.2, device="cuda")` maps `[batch, noise_dim]` to `[batch, code_dim]`.
- `Critic(code_dim=100, hidden_dim=512, num_layers=4,
  negative_slope=0.2, device="cuda")` returns scores of shape `[batch, 1]`.
- `Discriminator` is an alias for `Critic`.
- `Generator.sample_latent(batch_size, rng=None)` draws standard Gaussian noise
  on the generator's CUDA device, with its parameter dtype. Pass `rng` by keyword.

The default has four affine layers, three 512-wide hidden layers, and LeakyReLU
between affine layers. Both outputs are unrestricted. The critic has no sigmoid.
These dimensions and the 100d Gaussian input are documented interpretations of
the sentence-generation paper, not settings specified in full by that paper.
Invalid dimensions and CPU selection raise errors. Parameters are initialized
directly on CUDA. Noise and interpolation tensors use the selected CUDA device;
there is no fixed GPU index and no CPU fallback.

## Loss and update API

`gradient_penalty(critic, real_codes, fake_codes, alpha=None, rng=None)` returns
the **unweighted** mean squared deviation of the input-gradient norm from one.
Pass optional arguments by keyword. Both code batches must have shape
`[batch, code_dim]`, with the same CUDA device and dtype. Interpolation coefficients
have shape `[batch, 1]`. The function detaches both code inputs and uses
`torch.autograd.grad(..., create_graph=True)`, so the penalty remains differentiable
with respect to critic parameters. An optional `alpha` tensor supports analytic checks.

- `critic_loss(real_scores, fake_scores, penalty, gp_weight=10)` returns
  `mean(fake_scores) - mean(real_scores) + gp_weight * penalty`.
- `generator_loss(fake_scores)` returns `-mean(fake_scores)`.
- `critic_step(generator, critic, optimizer, real_codes, gp_weight=10,
  noise=None, rng=None)` updates only the critic. Pass arguments after `real_codes`
  by keyword. The function detaches real encoder codes and generates fake codes
  without a generator graph. Generator gradients are cleared.
- `generator_step(generator, critic, optimizer, batch_size, noise=None, rng=None)`
  updates only the generator. Pass optional arguments by keyword. Critic gradients
  are cleared and critic weights are frozen, but critic input gradients remain active.
- `frozen_parameters(module)` restores each original gradient flag, including
  when the guarded block raises an exception.

Each update clears stale gradients in both GAN models. Its optimizer must own
exactly the trainable parameters of the model for that phase. Models must not
share parameters. Losses, gradients, and parameters are checked for finite values;
the function raises an error rather than proceeding with invalid values. A failed
post-step check does not roll back optimizer state; discard or restore the failed run.
Diagnostics are Python floats: critic loss, unweighted penalty, real/fake mean
scores, and parameter-gradient norm for a critic update; generator loss and
parameter-gradient norm for a generator update.

`Trainer(generator, critic, gen_optimizer, critic_optimizer, gp_weight=10,
critic_iterations=5, generator_iterations=1)` exposes the same `critic_step`
and `generator_step` methods and a `train_cycle(real_codes, rng=None)` method.
Pass settings and `rng` by keyword. A cycle reuses one real batch, draws fresh noise
for each update, and returns lists under `critic` and `generator`. Its
`critic_updates` and `generator_updates` counters count completed steps. Call
individual steps with separate real batches if data exposure must differ.

The caller owns encoder training, optimizers, data loading, entropy losses,
checkpoint state, and update schedules. Save both model and optimizer state dicts
and the caller's CUDA RNG state. Save trainer counters separately if used.
Do not combine a POG entropy step with the Wasserstein generator step.
For Adam, use `capturable=True` to keep its scalar step-state tensors on CUDA too.

## Tiny standalone CUDA example

Install this fork through the parent project's local editable dependency setup.
Then run this example in that environment. It uses small, non-paper dimensions.
It does not download data or import image packages.

```python
import torch
from wgan_gp import Generator, Critic, Trainer

generator = Generator(noise_dim=3, code_dim=2, hidden_dim=8, device="cuda")
critic = Critic(code_dim=2, hidden_dim=8, device="cuda")
g_opt = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.0, 0.9), capturable=True)
d_opt = torch.optim.Adam(critic.parameters(), lr=1e-4, betas=(0.0, 0.9), capturable=True)
trainer = Trainer(generator, critic, g_opt, d_opt, critic_iterations=2)
real_codes = torch.randn(4, 2, device=next(generator.parameters()).device)
diagnostics = trainer.train_cycle(real_codes)
print(diagnostics)
```

From the parent project, run the CUDA acceptance tests explicitly:

```sh
snip uv run --locked pytest tests/reproduction/test_latent_gan.py
```

The tests require CUDA and fail if it is absent. They cover model dimensions,
unbounded scores, analytic linear-critic GP and its parameter derivative, loss
signs, encoder/generator/critic isolation, freeze-flag restoration, finite
diagnostics, and configurable update counts.

## Legacy files and import boundary

Root `models.py` and `training.py` reexport the latent API for old import paths.
They do not preserve obsolete image constructor or trainer signatures. The
supported package does not import `main.py`, `dataloaders.py`, image data,
`torchvision`, `imageio`, or visualization code. Existing images, GIFs, and image
entrypoints remain as historical files; those entrypoints are not supported by
the latent API. Image convolutions, image-shaped interpolation, sigmoid scores,
and automatic GIF generation are removed from the supported path.

## Sources and inspiration

- https://github.com/caogang/wgan-gp
- https://github.com/kuc2477/pytorch-wgan-gp
