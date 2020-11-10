import matplotlib.pyplot as plt


def show_quality(real, gen):
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
        plt.show()
        plt.ion()

    plt.pause(0.001)