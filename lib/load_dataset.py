import os
import numpy as np


def load_st_dataset(dataset):
    # output B, N, D
    if dataset == 'miniapp1':
        data_path = os.path.join('./data/miniapp1/data.npz')
        data = np.load(data_path)['data'][:, :, 0]  # onley the first dimension, traffic flow data
    elif dataset == 'miniapp2':
        data_path = os.path.join('./data/miniapp2/data.npz')
        data = np.load(data_path)['data'][:, :, 0]  # onley the first dimension, traffic flow data
    elif dataset == 'bintree':
        data_path = os.path.join('./data/bintree/data.npz')
        data = np.load(data_path)['data'][:, :, 0]  # onley the first dimension, traffic flow data
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data
