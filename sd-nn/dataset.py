from sklearn.datasets import make_regression, load_boston
import numpy as np


# create random dataset for regression
def create_dataset(typ='poly', length=100, normalize=False, reshape_target=False):
    if typ == 'poly':
        x = np.linspace(-1, 1, length).reshape(-1, 1)  # x data (tensor), shape=(100, 1)
        y = np.power(x, 2) + 0.2 * np.random.rand()  # noisy y data (tensor), shape=(100, 1)
        print(x.shape, y.shape)

    elif typ == 'sin':
        x = np.linspace(-10, 10, length).reshape(-1, 1)  # x data (tensor), shape=(100, 1)
        y = np.sin(x)  # noisy y data (tensor), shape=(100, 1)
        print(x.shape, y.shape)

    else:
        data = make_regression(n_samples=length, n_features=1, noise=5, random_state=3, bias=50)
        x, y = data

    if reshape_target:
        y = y.reshape(-1, 1)

    if normalize:
        x = (x - np.mean(x)) / np.std(x)
    # torch.manual_seed(1)  # reproducible

    return x, y
