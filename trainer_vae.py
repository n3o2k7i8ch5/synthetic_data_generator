import torch
from torch.autograd import Variable
from torch.nn import MSELoss
from torch.utils.data import DataLoader
import numpy as np

from consts import parent_path, BATCH_SIZE
from load_data import load_data
from models_vae import Autoencoder
from show_quality import show_quality, show_lat_histograms


class Trainer:
    LATENT_SPACE_SIZE = 6

    AUTOENC_SAVE_PATH = parent_path() + 'data/single_prtc_autoenc.model'
    PDG_DEEMBED_SAVE_PATH = parent_path() + 'data/pdg_deembed.model'

    SHOW_FEAT_RANGE = (-3, 0)  # (-6, -5)

    def __init__(self):
        self.device = torch.device("cuda")

    def load_data(self) -> (torch.Tensor, int):
        data, attr_cnt = load_data(size=100_000)
        data = torch.tensor(data, device=self.device)
        return data, attr_cnt

    def prep_data(self, data: torch.Tensor, batch_size: int, valid=0.1, shuffle=True) -> (DataLoader, DataLoader):

        valid_cnt = int(len(data) * valid)

        train_vals = data[valid_cnt:, :]
        valid_vals = data[:valid_cnt, :]

        train_data_loader = DataLoader(train_vals, batch_size=batch_size, shuffle=shuffle)
        valid_data_loader = DataLoader(valid_vals, batch_size=batch_size, shuffle=shuffle)

        return train_data_loader, valid_data_loader

    def loss(self, input_x, output_x, lat_mean: torch.Tensor, lat_logvar: torch.Tensor) -> (Variable, Variable, Variable):
        mse_loss = MSELoss()(input_x, output_x)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + lat_logvar - lat_mean.pow(2) - lat_logvar.exp(), dim=1), dim=0)

        return mse_loss + kld_loss * 1e-4, mse_loss, kld_loss

    def create_autoenc(self, attr_cnt) -> Autoencoder:
        # DAN
        # return AutoencPrtcl(emb_features=EMB_FEATURES, latent_size=PRTCL_LATENT_SPACE_SIZE, device=self.device)
        '''
        return Autoencoder(
            in_size=3,
            latent=LATENT_SPACE_SIZE,
            device=self.device
        )
        '''

        _autoenc_in, _autoenc_out = Autoencoder.create(in_size=attr_cnt, latent=Trainer.LATENT_SPACE_SIZE,
                                                       device=self.device)
        autoenc = Autoencoder(_autoenc_in, _autoenc_out)

        #autoenc.apply(init_weights)
        return autoenc

    def train(self, epochs):
        print('TRAINING MODEL:'
              ' BATCH_SIZE = ' + str(BATCH_SIZE) +
              ', EPOCHS: ' + str(epochs) +
              ', LATENT_SPACE_SIZE: ' + str(Trainer.LATENT_SPACE_SIZE)
              )

        err: torch.Tensor = torch.Tensor()
        real_data: torch.Tensor = None
        emb_data: torch.Tensor = torch.Tensor()
        gen_data: torch.Tensor = torch.Tensor()

        data, attr_cnt = self.load_data()
        data_train, data_valid = self.prep_data(data, batch_size=BATCH_SIZE, valid=0.1)

        autoenc = self.create_autoenc(attr_cnt)
        autoenc_optimizer = torch.optim.Adam(autoenc.parameters(), lr=0.001)

        print('AUTOENCODER')
        print(autoenc)

        for epoch in range(epochs):

            for n_batch, batch in enumerate(data_train):
                autoenc_optimizer.zero_grad()

                # DAN
                # real_data: torch.Tensor = batch.to(device=device).detach()
                real_data: torch.Tensor = batch.to(self.device)

                # DAN
                # emb_data = self.embed_data(embeder, real_data)

                # DAN
                # gen_data, lat_mean, lat_logvar, lat_vec = autoenc(emb_data)
                gen_data, lat_mean, lat_logvar, lat_vec = autoenc(real_data)

                err, mse_loss, kld_loss = self.loss(
                    # DAN
                    # input_x=emb_data,
                    input_x=real_data,
                    output_x=gen_data,
                    lat_mean=lat_mean,
                    lat_logvar=lat_logvar)

                err.backward()
                autoenc_optimizer.step()

                if epoch%10 == 0 and n_batch % 500 == 0:
                    show_lat_histograms(lat_mean=lat_mean, lat_logvar=lat_logvar)
                    #valid_loss = self._valid_loss(autoenc, data_valid)
                    print(
                        f'Epoch: {str(epoch)}/{epochs} :: '
                        f'Batch: {str(n_batch)}/{str(len(data_train))} :: '
                        f'train loss: {str(err.item())} :: '
                        f'kld loss: {str(kld_loss.item())} :: '
                        f'mse loss: {str(mse_loss.item())} :: ')
                        #f'valid loss: {str(valid_loss)}')

                    # DAN
                    # show_quality(emb_data, gen_data, feature_range=Trainer.SHOW_FEAT_RANGE, save=True)
                    show_quality(
                        real_data,
                        gen_data,
                        #feature_range=Trainer.SHOW_FEAT_RANGE,
                        save=True
                    )
                    self.show_real_gen_data_comparison(autoenc, save=True)

            # DAN
            # show_quality(emb_data, gen_data, feature_range=Trainer.SHOW_FEAT_RANGE)
            if real_data is not None:
                show_quality(
                    real_data,
                    gen_data,
                    feature_range=Trainer.SHOW_FEAT_RANGE
                )

        print('Saving autoencoder model')
        torch.save(autoenc.state_dict(), Trainer.AUTOENC_SAVE_PATH)

        return autoenc

    def _valid_loss(self, autoenc, valid_data_loader) -> float:
        loss = 0
        criterion = MSELoss()

        for batch_data in valid_data_loader:
            # DAN
            # emb_data = self.embed_data(embeder, batch_data.to(self.device))
            batch_data = batch_data.to(self.device)
            # DAN
            # out_prtcl_emb, lat_mean, lat_vec, lat_vec = autoenc(emb_data)
            out_prtcl_emb, lat_mean, lat_vec, lat_vec = autoenc(batch_data)
            # DAN
            # train_loss = criterion(out_prtcl_emb, emb_data)
            train_loss = criterion(out_prtcl_emb, batch_data)

            loss += train_loss.item()

        loss /= len(valid_data_loader)

        return loss

    def gen_autoenc_data(self, sample_cnt, autoenc):
        np_input = np.random.normal(loc=0, scale=1, size=(sample_cnt, Trainer.LATENT_SPACE_SIZE))
        rand_input = torch.from_numpy(np_input).float().to(device=self.device)
        generated_data = autoenc.auto_out(rand_input).detach()
        return generated_data

    def show_real_gen_data_comparison(self, autoenc, load_model: bool = False, save: bool = False):

        if load_model:
            autoenc.load_state_dict(torch.load(Trainer.AUTOENC_SAVE_PATH))

        gen_data = self.gen_autoenc_data(1000, autoenc)
        real_data, attr_cnt = self.load_data()[:1000]
        # DAN
        # emb_data = self.embed_data(embeder, real_data)
        # DAN
        # show_quality(emb_data, gen_data, feature_range=Trainer.SHOW_FEAT_RANGE, save=save, title='Generation comparison')
        show_quality(real_data, gen_data, feature_range=Trainer.SHOW_FEAT_RANGE, save=save, title='Generation comparison')
