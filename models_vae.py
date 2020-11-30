import torch
from torch import nn

from sample_norm import sample_norm


class AutoencoderIn(nn.Module):

    def __init__(self, in_size: int, latent_size: int, device):
        self.device = device
        super(AutoencoderIn, self).__init__()

        PRE_LAT_SIZE = 16

        self.net = nn.Sequential(
            nn.Linear(in_size, 32, bias=True),
            nn.Tanh(),
            nn.Linear(32, 64, bias=True),
            nn.Tanh(),
            nn.Linear(64, PRE_LAT_SIZE, bias=True),
            nn.Tanh()
        ).to(device=device)

        self.mean = nn.Sequential(
            nn.Linear(PRE_LAT_SIZE, 32, bias=True),
            nn.Tanh(),
            nn.Linear(32, 64, bias=True),
            nn.Tanh(),
            nn.Linear(64, latent_size, bias=True)

        ).to(device=device)

        self.logvar = nn.Sequential(
            nn.Linear(PRE_LAT_SIZE, 32, bias=True),
            nn.Tanh(),
            nn.Linear(32, 64, bias=True),
            nn.Tanh(),
            nn.Linear(64, latent_size, bias=True),

        ).to(device=device)

    def forward(self, x: torch.Tensor):
        out = self.net(x)

        lat_mean = self.mean(out)
        lat_logvar = self.logvar(out)

        return lat_mean, lat_logvar


class AutoencoderOut(nn.Module):

    def __init__(self, latent_size: int, in_size: int, device):
        self.device = device
        super(AutoencoderOut, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_size, 32, bias=True),
            nn.Tanh(),
            nn.Linear(32, 64, bias=True),
            nn.Tanh(),
            nn.Linear(64, in_size, bias=True),

            # nn.Linear(latent_size, 64, bias=True),
            # nn.Tanh(),
            # nn.Linear(64, 128, bias=True),
            # nn.Tanh(),
            # nn.Linear(128, in_size, bias=True),
            # #nn.Tanh(),

        ).to(device=device)

    def forward(self, x: torch.Tensor):
        x = self.net(x)

        return x


class Autoencoder(nn.Module):
    def __init__(self, auto_in: AutoencoderIn, auto_out: AutoencoderOut):
        super(Autoencoder, self).__init__()

        self.auto_in = auto_in
        self.auto_out = auto_out

    def forward(self, x: torch.Tensor):
        lat_mean, lat_logvar = self.auto_in(x)

        lat_vec = sample_norm(mean=lat_mean, logvar=lat_logvar)

        out = self.auto_out(lat_vec)
        # out = self.auto_out(lat_vec)
        return out, lat_mean, lat_logvar

    @staticmethod
    def create(in_size: int, latent: int, device):
        auto_in = AutoencoderIn(in_size=in_size, latent_size=latent, device=device)
        auto_out = AutoencoderOut(in_size=in_size, latent_size=latent, device=device)

        return auto_in, auto_out