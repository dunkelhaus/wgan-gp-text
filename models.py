import torch
import torch.nn as nn
from typing import Tuple


class Generator(nn.Module):
    """
    Main Generator class.

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
            img_size: Tuple[int],
            latent_dim: int,
            dim: int
    ):
        """
        :param img_size: Input image shape in form:
                        Tuple(int, int, int)
                        Height and width must be powers of 2,
                        e.g. (32, 32, 1) or(64, 128, 3).
                        Last number indicates number of channels,
                        e.g. 1 for grayscale or 3 for RGB.
        :param latent_dim: Dimension for the noise vector.
        :param dim: Base factor of dimensions for convolutions.
        """
        super(Generator, self).__init__()

        self.dim = dim
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.feature_sizes = (self.img_size[0] / 16,
                              self.img_size[1] / 16)

        self.latent_to_features = nn.Sequential(
            nn.Linear(
                latent_dim,
                int(8 * dim * self.feature_sizes[0] *
                    self.feature_sizes[1])
            ),
            nn.ReLU()
        )

        self.features_to_image = nn.Sequential(
            nn.ConvTranspose2d(8 * dim, 4 * dim, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(4 * dim),
            nn.ConvTranspose2d(4 * dim, 2 * dim, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(2 * dim),
            nn.ConvTranspose2d(2 * dim, dim, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(dim),
            nn.ConvTranspose2d(dim, self.img_size[2], 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Single forward pass.

        :param input_data: Batch of noise data.
        :return: Generator output.
        """
        # Map latent into appropriate size for transposed convolutions
        x = self.latent_to_features(input_data)
        # Reshape
        x = x.view(-1, int(8 * self.dim),
                   int(self.feature_sizes[1]),
                   int(self.feature_sizes[0]))

        return self.features_to_image(x)

    def sample_latent(self, num_samples: int) -> torch.Tensor:
        """
        Sample noise data, and return torch.Tensor of the same
        shape as the training batch, using num_samples.

        :param num_samples: Number of samples in batch.
        :return torch.Tensor: Tensor of shape (num_samples,
                              self.latent_dim) containing data
                              sampled from normal distribution.
        """
        return torch.randn((num_samples, self.latent_dim))


class Discriminator(nn.Module):
    """
    Main Discriminator class.

    :ivar img_size: Input image shape in form: Tuple(int, int, int)
                     Height and width must be powers of 2,
                     e.g. (32, 32, 1) or(64, 128, 3).
                     Last number indicates number of channels,
                     e.g. 1 for grayscale or 3 for RGB.
    :ivar image_to_features: nn.Sequential block to propagate image
                             into network.
    :ivar features_to_prob: nn.Sequential block to convert outputs
                            of last layer into single
                            classification probability.
    """
    def __init__(self, img_size: Tuple[int], dim: int):
        """
        :param img_size: See class docs.
        :param dim: Base factor of dimensions for convolutions.
        """
        super(Discriminator, self).__init__()

        self.img_size = img_size

        self.image_to_features = nn.Sequential(
            nn.Conv2d(self.img_size[2], dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim, 2 * dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(2 * dim, 4 * dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(4 * dim, 8 * dim, 4, 2, 1),
            nn.Sigmoid()
        )

        # 4 convolutions of stride 2, i.e. halving of size everytime
        # So output size will be 8 * (img_size / 2 ^ 4) * (img_size / 2 ^ 4)
        output_size = int(8 * dim * (img_size[0] / 16) *
                          (img_size[1] / 16))
        self.features_to_prob = nn.Sequential(
            nn.Linear(output_size, 1),
            nn.Sigmoid()
        )

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Single forward pass.

        :param input_data: Batch of data from DataLoader.
        :return: Classification probability output.
        """
        batch_size = input_data.size()[0]
        x = self.image_to_features(input_data)
        x = x.view(batch_size, -1)

        return self.features_to_prob(x)
