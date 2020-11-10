import numpy as np


def load_data(size: int):
    data = []

    for i in range(size):

        sample_1 = -1
        sample_2 = np.random.normal(3, 1)

        if i % 2 == 0:
            sample_3 = np.random.normal(-1, 0.2)
        else:
            sample_3 = np.random.normal(1, 0.2)

        # data.append([sample_1, sample_1, sample_1, sample_2, sample_2, sample_2])
        data.append([sample_1, sample_2, sample_3])

    return data
