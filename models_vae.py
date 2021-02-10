import torch
from torch import nn

from sample_norm import sample_norm


class AutoencoderIn(nn.Module):

    def __init__(self, in_size: int, latent_size: int, device):
        self.device = device
        super(AutoencoderIn, self).__init__()

        # PRE_LAT_SIZE = 64

        self.mean = nn.Sequential(
            nn.Linear(in_size, 64, bias=True),
            nn.Tanh(),
            nn.Linear(64, 128, bias=True),
            nn.Tanh(),
            nn.Linear(128, 256, bias=True),
            nn.Tanh(),
            nn.Linear(256, 256, bias=True),
            nn.Tanh(),
            nn.Linear(256, latent_size, bias=True),
            nn.Tanh()
        ).to(device=device)

        self.logvar = nn.Sequential(
            nn.Linear(in_size, 64, bias=True),
            nn.Tanh(),
            nn.Linear(64, 128, bias=True),
            nn.Tanh(),
            nn.Linear(128, 256, bias=True),
            nn.Tanh(),
            nn.Linear(256, 256, bias=True),
            nn.Tanh(),
            nn.Linear(256, latent_size, bias=True),
            nn.Tanh()
        ).to(device=device)

    def forward(self, x: torch.Tensor):
        lat_mean = self.mean(x)
        lat_logvar = self.logvar(x)

        return lat_mean, lat_logvar


class AutoencoderOut(nn.Module):

    def __init__(self, latent_size: int, in_size: int, device):
        self.device = device
        super(AutoencoderOut, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_size, 64, bias=True),
            nn.Tanh(),
            nn.Linear(64, 128, bias=True),
            nn.Tanh(),
            nn.Linear(128, 256, bias=True),
            nn.Tanh(),
            nn.Linear(256, 256, bias=True),
            nn.Tanh(),
            nn.Linear(256, in_size, bias=True),
        ).to(device=device)

    def forward(self, x: torch.Tensor):
        return self.net(x)


class Autoencoder(nn.Module):
    def __init__(self, auto_in: AutoencoderIn, auto_out: AutoencoderOut):
        super(Autoencoder, self).__init__()

        self.auto_in = auto_in
        self.auto_out = auto_out

    def forward(self, x: torch.Tensor):
        lat_mean, lat_logvar = self.auto_in(x)

        lat_vec = sample_norm(mean=lat_mean, logvar=lat_logvar)

        out = self.auto_out(lat_vec)
        return out, lat_mean, lat_logvar, lat_vec

    @staticmethod
    def create(in_size: int, latent: int, device):
        auto_in = AutoencoderIn(in_size=in_size, latent_size=latent, device=device)
        auto_out = AutoencoderOut(in_size=in_size, latent_size=latent, device=device)

        return auto_in, auto_out
