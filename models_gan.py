from torch import nn


class Generator(nn.Module):
    def __init__(self, latent_size: int, in_size: int):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_size, 16),
            nn.Tanh(),
            nn.Linear(16, 64),
            nn.Tanh(),
            nn.Linear(64, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
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
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        validity = self.model(x)
        return validity
