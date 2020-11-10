# Optimizers
import numpy as np
import torch
from torch import Tensor
from torch.autograd import Variable
from torch.utils.data import DataLoader

from consts import parent_path, BATCH_SIZE
from load_data import load_data
from models_gan import Generator, Discriminator
from show_quality import show_quality

AUTOENC_MODEL_PATH = parent_path() + 'data/single_prtc_gan_model'

device = torch.device("cpu")

data = load_data(100_000)

data_train = DataLoader(
    torch.tensor(data),
    batch_size=BATCH_SIZE,
    shuffle=True
)

LATENT_SPACE_SIZE = 3
IN_SIZE = 3

# Initialize generator and discriminator
generator = Generator(latent_size=LATENT_SPACE_SIZE, in_size=IN_SIZE)
discriminator = Discriminator(in_size=IN_SIZE)

adversarial_loss = torch.nn.MSELoss()

print(generator)
print(discriminator)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.001)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.001)

# ----------
#  Training
# ----------

for epoch in range(200):
    for n_batch, batch in enumerate(data_train):

        # Adversarial ground truths
        valid = Variable(Tensor(len(batch), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(len(batch), 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_data = batch

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = torch.normal(mean=0, std=1, size=(len(batch), LATENT_SPACE_SIZE))

        # Generate a batch of images
        gen_data = generator(z)

        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_data)

        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_data), valid)
        fake_loss = adversarial_loss(discriminator(gen_data.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        if epoch % 1 == 0 and n_batch == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, epoch, n_batch, len(data_train), d_loss.item(), g_loss.item())
            )

            import matplotlib.pyplot as plt
            plt.close('all')
            show_quality(
                real=real_data,
                gen=gen_data,
            )

        batches_done = epoch * len(data_train) + n_batch
