import os

import matplotlib.pyplot as plt

last_saved_cnt = 0


def show_quality(real, gen, save=False, loss=None, mse_loss=None, kld_loss=None):
    global last_saved_cnt
    for i in range(0, real.shape[1]):
        plt.figure('Feature ' + str(i))
        real_attr = real[:, i].detach().flatten()
        gen_attr = gen[:, i].detach().flatten()

        plt.hist([real_attr, gen_attr],
                 range=(-5, 5),
                 stacked=False,
                 bins=100,
                 histtype='stepfilled',
                 label=['real', 'fake'],
                 color=['blue', 'red'],
                 alpha=0.5
                 )
        plt.legend()
        if save:
            folder = 'vae_plots'
            if not os.path.isdir(folder):
                os.mkdir(folder)
            loss_str = ''

            "{:.3f}".format(round(mse_loss, 3))

            if mse_loss is not None:
                loss_str += f'loss: {"{:.3f}".format(round(loss, 3))}'
            if mse_loss is not None:
                loss_str += f' mse_loss: {"{:.3f}".format(round(mse_loss, 3))}'
            if kld_loss is not None:
                loss_str += f' kld_loss: {"{:.3f}".format(round(kld_loss, 3))}'

            plt.savefig(f'{folder}/plot_{i}_{last_saved_cnt}_{loss_str}.png')

        plt.show()
        plt.ion()

    last_saved_cnt += 1
    plt.pause(0.001)
