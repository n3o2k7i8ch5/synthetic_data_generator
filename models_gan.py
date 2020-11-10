import torch
from torch import nn

from sample_norm import sample_norm


class Generator(nn.Module):
    def __init__(self, latent_size: int, in_size: int):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_size, 16),
            nn.Tanh(),
            nn.Linear(16, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, in_size),
        )

    def forward(self, z):
        out = self.model(z)
        return out


class Discriminator(nn.Module):
    def __init__(self, in_size: int):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(in_size, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        validity = self.model(x)
        return validity
