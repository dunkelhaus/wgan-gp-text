import torch
import imageio
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import make_grid
from torch.autograd import grad as torch_grad


class Trainer():
    """
    Main Trainer class; provides dataset-wide train function,
    as well as train functions per epoch, and one for each
    the Generator and the Discriminator.

    :ivar G: Generator nn.Module object.
    :ivar D: Discriminator nn.Module object.
    :ivar G_opt: Generator optimizer object.
    :ivar D_opt: Discriminator optimizer object.
    :ivar losses: Losses dictionary.
    :ivar num_steps: Current iteration step counter.
    :ivar use_cuda: Boolean to check whether to use cuda or gpu.
    :ivar gp_weight: Gradient penalty weight hyperparameter.
    :ivar critic_iterations: Number of Discriminator iterations to
                             let pass, before updating Generator.
    :ivar print_every: Print outputs every n iterations.
    """
    def __init__(
            self,
            generator: nn.Module,
            discriminator: nn.Module,
            gen_optimizer: torch.optim,
            dis_optimizer: torch.optim,
            gp_weight: int = 10,
            critic_iterations: int = 5,
            print_every: int = 50,
            use_cuda: bool = False
    ):
        self.G = generator
        self.G_opt = gen_optimizer
        self.D = discriminator
        self.D_opt = dis_optimizer
        self.losses = {
            'G': [],
            'D': [],
            'GP': [],
            'gradient_norm': []
        }
        self.num_steps = 0
        self.use_cuda = use_cuda
        self.gp_weight = gp_weight
        self.critic_iterations = critic_iterations
        self.print_every = print_every

        if self.use_cuda:
            self.G.cuda()
            self.D.cuda()

    # XXX: Stays unchanged.
    def _critic_train_iteration(
            self,
            data: torch.Tensor
    ) -> None:
        """
        Run a single batch through the discriminator.
        Computes losses, applies GP, propagates loss backward.

        :param data: One batch of data from the dataloader.
        """
        # Get generated data
        batch_size = data.size()[0]
        generated_data = self.sample_generator(batch_size)

        # Calculate probabilities on real and generated data
        data = Variable(data)
        if self.use_cuda:
            data = data.cuda()
        d_real = self.D(data)
        d_generated = self.D(generated_data)

        # Get gradient penalty
        gradient_penalty = self._gradient_penalty(data, generated_data)
        self.losses['GP'].append(gradient_penalty.data.item())

        # Create total loss and optimize
        self.D_opt.zero_grad()
        d_loss = d_generated.mean() - d_real.mean() + gradient_penalty
        d_loss.backward()

        self.D_opt.step()

        # Record loss
        self.losses['D'].append(d_loss.data.item())

    # XXX: Stays unchanged.
    def _generator_train_iteration(
            self,
            data: torch.Tensor
    ) -> None:
        """
        Run a single batch through the generator.
        Computes losses using discriminator, propagates loss
        backward.

        :param data: One batch of data from the dataloader.
        """
        self.G_opt.zero_grad()

        # Get generated data
        batch_size = data.size()[0]
        generated_data = self.sample_generator(batch_size)

        # Calculate loss and optimize
        d_generated = self.D(generated_data)
        g_loss = - d_generated.mean()
        g_loss.backward()
        self.G_opt.step()

        # Record loss
        self.losses['G'].append(g_loss.data.item())

    # XXX: Changes made.
    # NOTE: Changed to gradnorm from gradients_norm since it looks
    # like mentioned problem may be an issue with old pytorch.
    # Current pytorch norm works well; if it fails,
    # will revert to manual calculation of norm.
    def _gradient_penalty(
            self,
            real_data: torch.Tensor,
            generated_data: torch.Tensor
    ) -> float:
        """
        Computes gradient penalty for given pair of real and
        generated data.
        Applies gradient penalty weight as well.

        :param real_data: Example from training data.
        :param generated_data: Output from Generator.
        """
        batch_size = real_data.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real_data)

        if self.use_cuda:
            alpha = alpha.cuda()

        interpolated = (alpha * real_data.data +
                        (1 - alpha) * generated_data.data)
        interpolated = Variable(interpolated, requires_grad=True)

        if self.use_cuda:
            interpolated = interpolated.cuda()

        # Calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(
            outputs=prob_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones(prob_interpolated.size()).cuda()
            if self.use_cuda
            else torch.ones(prob_interpolated.size()),
            create_graph=True,
            retain_graph=True
        )[0]

        # Gradients have shape:
        # (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)
        gradnorm = gradients.norm(2, dim=1)
        self.losses['gradient_norm'].append(
            gradnorm.mean().data.item()
        )
        # print(f"Saved gradnorm:
        #        f"{self.gp_weight * ((gradnorm - 1) ** 2).mean()}")

        # --- Former issue ---
        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        # gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        # print(f"Used gradnorm:
        #        f"{self.gp_weight * ((gradients_norm - 1) ** 2).mean()}")
        # --- xxx ---

        # Return gradient penalty
        return self.gp_weight * ((gradnorm - 1) ** 2).mean()

    def _train_epoch(
            self,
            data_loader: torch.utils.data.DataLoader
    ) -> None:
        """
        Train a single epoch of the WGAN.

        :param data_loader: PyTorch training dataloader.
        """
        for i, data in enumerate(data_loader):
            self.num_steps += 1
            self._critic_train_iteration(data[0])
            # Only update generator every |critic_iterations|
            # iterations
            if self.num_steps % self.critic_iterations == 0:
                self._generator_train_iteration(data[0])

            if i % self.print_every == 0:
                print(f"Iteration {i + 1}")
                print(f"D: {self.losses['D'][-1]}")
                print(f"GP: {self.losses['GP'][-1]}")
                print("Gradient norm: "
                      f"{self.losses['gradient_norm'][-1]}")

                if self.num_steps > self.critic_iterations:
                    print(f"G: {self.losses['G'][-1]}")

    def train(
            self,
            data_loader: torch.utils.data.DataLoader,
            epochs: int,
            save_training_gif: bool = True
    ) -> None:
        """
        Topmost train function.

        :param data_loader: PyTorch DataLoader for training examples.
        :param epochs: Total number of epochs to train for.
        :param save_training_gif: Whether to save intermediary
                                  images during training to be
                                  combined into one gif later.
        """
        if save_training_gif:
            # Fix latents to see how image generation improves during training
            fixed_latents = Variable(self.G.sample_latent(64))

            if self.use_cuda:
                fixed_latents = fixed_latents.cuda()
            training_progress_images = []

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}")
            self._train_epoch(data_loader)

            if save_training_gif:
                # Generate batch of images and convert to grid
                img_grid = make_grid(self.G(fixed_latents).cpu().data)
                # Convert to numpy and transpose axes to fit imageio convention
                # i.e. (width, height, channels)
                img_grid = np.transpose(img_grid.numpy(), (1, 2, 0))
                # Add image grid to training progress
                training_progress_images.append(img_grid)

        if save_training_gif:
            imageio.mimsave(f"./training_{epochs}_epochs.gif",
                            training_progress_images)

    def sample_generator(self, num_samples: int) -> torch.Tensor:
        """
        Sample noise vector of shape (num_samples, latent_dim),
        define as PyTorch Variable, and pass through the Generator.
        Return generated data.

        :param num_samples: Essentially the batch size.
        :return generated_data: Output from Generator for sampled noise vector.
        """
        latent_samples = Variable(self.G.sample_latent(num_samples))

        if self.use_cuda:
            latent_samples = latent_samples.cuda()
        generated_data = self.G(latent_samples)

        return generated_data

    # XXX: Unused function.
    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Gets results from Trainer.sample_generator, and removes
        the color channel before returning.
        Note: Moves return data to cpu for numpy comp, does not
              move it back to cuda.

        :param num_samples: Essentially the batch size.
        :return generated_data: Output from Generator for sampled
                                noise vector, with the color
                                channel removed.
        """
        generated_data = self.sample_generator(num_samples)
        # Remove color channel

        return generated_data.data.cpu().numpy()[:, 0, :, :]
