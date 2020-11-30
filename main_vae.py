import os

import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import MSELoss
from torch.utils.data import DataLoader

from consts import BATCH_SIZE, parent_path
from load_data import load_data
from models_vae import Autoencoder
from show_quality import show_quality

'''
LINKS:

https://github.com/pytorch/examples/blob/master/vae/main.py

'''

AUTOENC_MODEL_PATH = parent_path() + 'data/single_prtc_autoenc_model'

device = torch.device("cpu")

data = load_data(100_000)

data_train = DataLoader(
    torch.tensor(data),
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

autoenc_optimizer = torch.optim.SGD(autoenc.parameters(), lr=0.1, momentum=0.01)


def _loss(input, output, lat_mean: torch.Tensor, lat_logvar: torch.Tensor, show=False) -> (Variable, Variable, Variable):
    mse_loss = MSELoss()(input, output)

    # kl_loss = 0.5 * torch.sum(lat_var.exp() + lat_mu.pow(2) - 1.0 - lat_var)# / len(lat_var.)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld_loss = torch.mean(-0.5 * torch.sum(1 + lat_logvar - lat_mean.pow(2) - lat_logvar.exp(), dim=1), dim=0)

    if show:
        print('mse_loss: ' + str(mse_loss.item()))
        print('kld_loss: ' + str(kld_loss.item()))

    return mse_loss + kld_loss * 0.02, mse_loss, kld_loss


def train_autoenc():
    for epoch in range(500):

        loss: torch.Tensor = torch.Tensor()

        for n_batch, batch in enumerate(data_train):

            real_data = batch.float()

            autoenc_optimizer.zero_grad()

            out_data, lat_mean, lat_logvar = autoenc(real_data)
            # gen_data = generate_data(mu=lat_mu, var=lat_var)

            loss, mse_loss, kld_loss = _loss(
                input=real_data,
                output=out_data,
                lat_mean=lat_mean,
                lat_logvar=lat_logvar,
                show=epoch % 50 == 49 and n_batch == 0
            )
            loss.backward()
            autoenc_optimizer.step()

            if epoch % 50 == 49 and n_batch == 0:
                print('Epoch: ' + str(epoch) + ', batch:' + str(n_batch) + '/' + str(len(data_train)))
                show_quality(
                    real=real_data,
                    gen=out_data,
                    save=True, loss=loss.item(), mse_loss=mse_loss.item(), kld_loss=kld_loss.item())

                '''
                plt.figure('Lattent mu')
                plt.hist(lat_mu.cpu().detach().flatten(), 100)

                plt.figure('Lattent var')
                plt.hist(lat_var.cpu().detach().flatten(), 100)
                plt.show()

                plt.pause(0.001)
                '''

        print('loss: ' + str(loss.item()))
        # show_quality(emb_data, gen_data)

    path = os.path.join(parent_path(),'data')
    if not os.path.isdir(path):
        os.mkdir(path)

    print('saving model...')
    torch.save(autoenc.state_dict(), AUTOENC_MODEL_PATH)


def gen_autoenc_data(autoenc, sample_cnt, load_model: False):
    if load_model:
        autoenc = create_autoenc()
        autoenc.load_state_dict(torch.load(AUTOENC_MODEL_PATH))

    autoenc_out = autoenc.auto_out

    np_input = np.random.normal(loc=0, scale=1, size=(sample_cnt, LATENT_SPACE_SIZE))
    rand_input = torch.from_numpy(np_input).float().to(device=device)
    generated_data = autoenc_out.forward(rand_input).detach()
    return generated_data


train_autoenc()

gen_data = gen_autoenc_data(autoenc, 1000, load_model=True)
real_data = torch.tensor(data[:1000])
show_quality(real=real_data, gen=gen_data)
