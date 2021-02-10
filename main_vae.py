import os

import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import MSELoss
from torch.utils.data import DataLoader

from consts import BATCH_SIZE, parent_path
from load_data import load_data
from models_vae import Autoencoder
from show_quality import show_quality, show_lat_histograms

'''
LINKS:
https://github.com/pytorch/examples/blob/master/vae/main.py
'''


class Trainer2:
    LATENT_SPACE_SIZE = 6

    AUTOENC_MODEL_PATH = parent_path() + 'data/single_prtc_autoenc_model'


    SHOW_FEAT_RANGE = None#(-6, -4)

    def __init__(self):
        self.device = torch.device("cpu")

    def load_data(self) -> (torch.Tensor, int):
        data, attr_cnt = load_data(100_000)
        tensor = torch.tensor(data, device=self.device)
        return tensor, attr_cnt

    def prep_data(self, data: torch.Tensor, batch_size: int, valid=0.1, shuffle=True) -> (DataLoader, DataLoader):

        valid_cnt = int(len(data) * valid)

        train_vals = data[valid_cnt:, :]
        valid_vals = data[:valid_cnt, :]

        train_data_loader = DataLoader(train_vals, batch_size=batch_size, shuffle=shuffle)
        valid_data_loader = DataLoader(valid_vals, batch_size=batch_size, shuffle=shuffle)

        return train_data_loader, valid_data_loader

    def _loss(self, input, output, lat_mean: torch.Tensor, lat_logvar: torch.Tensor) -> (Variable, Variable, Variable):
        mse_loss = MSELoss()(input, output)
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + lat_logvar - lat_mean.pow(2) - lat_logvar.exp(), dim=1), dim=0)

        return mse_loss + kld_loss*10, mse_loss, kld_loss

    def create_autoenc(self) -> Autoencoder:
        _autoenc_in, _autoenc_out = Autoencoder.create(in_size=3, latent=Trainer2.LATENT_SPACE_SIZE, device=self.device)
        autoenc = Autoencoder(_autoenc_in, _autoenc_out)
        return autoenc

    def gen_autoenc_data(self, sample_cnt, autoenc):
        np_input = np.random.normal(loc=0, scale=1, size=(sample_cnt, Trainer2.LATENT_SPACE_SIZE))
        rand_input = torch.from_numpy(np_input).float().to(self.device)
        generated_data = autoenc.auto_out.forward(rand_input).detach()
        return generated_data

    def show_real_gen_data_comparison(self, autoenc, load_model: bool = False, save: bool = False):

        if load_model:
            autoenc.load_state_dict(torch.load(Trainer2.AUTOENC_MODEL_PATH))

        gen_data = self.gen_autoenc_data(2000, autoenc)
        real_data, _ = self.load_data()
        real_data = real_data[:2000]
        show_quality(real_data, gen_data, feature_range=Trainer2.SHOW_FEAT_RANGE, save=save,
                     title='Generation comparison')

    def train_autoenc(self):

        EPOCHS = 300

        _data, _ = self.load_data()
        data_train, data_valid = self.prep_data(_data, batch_size=BATCH_SIZE)

        autoenc = self.create_autoenc()
        print(autoenc)

        autoenc_optimizer = torch.optim.Adam(autoenc.parameters(), lr=0.002)

        for epoch in range(EPOCHS):

            loss: torch.Tensor = torch.Tensor()

            for n_batch, batch in enumerate(data_train):

                real_data = batch.float()

                autoenc_optimizer.zero_grad()

                out_data, lat_mean, lat_logvar, lat_vec = autoenc(real_data)
                # gen_data = generate_data(mu=lat_mu, var=lat_var)

                loss, mse_loss, kld_loss = self._loss(
                    input=real_data,
                    output=out_data,
                    lat_mean=lat_mean,
                    lat_logvar=lat_logvar,
                )
                loss.backward()
                autoenc_optimizer.step()

                if epoch % 10 == 0 and n_batch % 500 == 0:
                    show_lat_histograms(lat_mean, lat_logvar)
                    show_quality(
                        real=real_data,
                        gen=out_data,
                        feature_range=Trainer2.SHOW_FEAT_RANGE,
                        save=True, loss=loss.item(), mse_loss=mse_loss.item(), kld_loss=kld_loss.item())
                    self.show_real_gen_data_comparison(autoenc, save=True)
                    print(
                        f'Epoch: {str(epoch)}/{EPOCHS} :: '
                        f'Batch: {str(n_batch)}/{str(len(data_train))} :: '
                        f'train loss: {str(loss.item())} :: '
                        f'kld loss: {str(kld_loss.item())} :: '
                        f'mse loss: {str(mse_loss.item())} :: ')
                    # f'valid loss: {str(valid_loss)}')

            # show_quality(emb_data, gen_data)

        path = os.path.join(parent_path(), 'data')
        if not os.path.isdir(path):
            os.mkdir(path)

        print('saving model...')
        torch.save(autoenc.state_dict(), Trainer2.AUTOENC_MODEL_PATH)


Trainer2().train_autoenc()

'''
SHOW_FEAT_RANGE = (-6, -4)

device = torch.device("cpu")

data, attr_cnt = load_data(100_000)
data = torch.tensor(data, device=device)

data_train = DataLoader(
    data,
    batch_size=BATCH_SIZE,
    shuffle=True
)

LATENT_SPACE_SIZE = 10


def create_autoenc() -> Autoencoder:
    _autoenc_in, _autoenc_out = Autoencoder.create(in_size=3, latent=LATENT_SPACE_SIZE, device=device)
    autoenc = Autoencoder(_autoenc_in, _autoenc_out)
    return autoenc


autoenc = create_autoenc()

print(autoenc)

autoenc_optimizer = torch.optim.Adam(autoenc.parameters(), lr=0.001)


def _loss(input, output, lat_mean: torch.Tensor, lat_logvar: torch.Tensor) -> (Variable, Variable, Variable):
    mse_loss = MSELoss()(input, output)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld_loss = torch.mean(-0.5 * torch.sum(1 + lat_logvar - lat_mean.pow(2) - lat_logvar.exp(), dim=1), dim=0)

    return mse_loss + kld_loss * 0.1, mse_loss, kld_loss


def train_autoenc():

    EPOCHS = 100

    for epoch in range(EPOCHS):

        loss: torch.Tensor = torch.Tensor()

        for n_batch, batch in enumerate(data_train):

            real_data = batch.float()

            autoenc_optimizer.zero_grad()

            out_data, lat_mean, lat_logvar, lat_vec = autoenc(real_data)
            # gen_data = generate_data(mu=lat_mu, var=lat_var)

            loss, mse_loss, kld_loss = _loss(
                input=real_data,
                output=out_data,
                lat_mean=lat_mean,
                lat_logvar=lat_logvar,
            )
            loss.backward()
            autoenc_optimizer.step()

            if n_batch % 500 == 0:
                show_lat_histograms(lat_mean, lat_logvar)
                show_quality(
                    real=real_data,
                    gen=out_data,
                    feature_range=SHOW_FEAT_RANGE,
                    save=True, loss=loss.item(), mse_loss=mse_loss.item(), kld_loss=kld_loss.item())
                show_real_gen_data_comparison(autoenc, save=True)
                print(
                    f'Epoch: {str(epoch)}/{EPOCHS} :: '
                    f'Batch: {str(n_batch)}/{str(len(data_train))} :: '
                    f'train loss: {str(loss.item())} :: '
                    f'kld loss: {str(kld_loss.item())} :: '
                    f'mse loss: {str(mse_loss.item())} :: ')
                # f'valid loss: {str(valid_loss)}')

        # show_quality(emb_data, gen_data)

    path = os.path.join(parent_path(), 'data')
    if not os.path.isdir(path):
        os.mkdir(path)

    print('saving model...')
    torch.save(autoenc.state_dict(), AUTOENC_MODEL_PATH)


def gen_autoenc_data(sample_cnt, autoenc):
    np_input = np.random.normal(loc=0, scale=1, size=(sample_cnt, LATENT_SPACE_SIZE))
    rand_input = torch.from_numpy(np_input).float().to(device=device)
    generated_data = autoenc.auto_out.forward(rand_input).detach()
    return generated_data


def show_real_gen_data_comparison(autoenc, load_model: bool = False, save: bool = False):

    if load_model:
        autoenc.load_state_dict(torch.load(AUTOENC_MODEL_PATH))

    gen_data = gen_autoenc_data(1000, autoenc)
    real_data = data[:1000]
    show_quality(real_data, gen_data, feature_range=SHOW_FEAT_RANGE, save=save, title='Generation comparison')

train_autoenc()

show_real_gen_data_comparison(autoenc)
'''
