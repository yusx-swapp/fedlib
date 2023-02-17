import numpy as np


def random_sampler(n_clients, sample_rate):
        # select clients
    arr = np.arange(n_clients)
    np.random.shuffle(arr)
    selected = arr[:int(n_clients * sample_rate)]

    return selected
