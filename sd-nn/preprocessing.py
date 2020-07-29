from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


def preprocess_data(x, y, split=True, keras=True):

    if not keras:
        if not split:
            x = torch.from_numpy(x.astype(np.float32))
            y = torch.from_numpy(y.astype(np.float32))
            return x, y

        else:
            y = y.reshape(-1, 1)
            X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)
            X_train_torch = torch.from_numpy(X_train.astype(np.float32))
            y_train_torch = torch.from_numpy(y_train.astype(np.float32))
            X_test_torch = torch.from_numpy(X_test.astype(np.float32))
            y_test_torch = torch.from_numpy(y_test.astype(np.float32))
        #     print(X_train_torch.shape, y_train_torch.shape)
            return X_train_torch, y_train_torch, X_test_torch, y_test_torch

    else:

        if not split:
            return x, y

        else:
            X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)
            return X_train, y_train, X_test, y_test


def create_loaders(x_train, y_train, *test_data_args):

    if not test_data_args:
        trainset = TensorDataset(x_train, y_train)
        train_loader = DataLoader(trainset, batch_size=100, shuffle=True)
        return train_loader

    else:
        x_test, y_test = test_data_args
        trainset = TensorDataset(x_train, y_train)
        testset = TensorDataset(x_test, y_test)

        # create the loader of the dataset
        train_loader = DataLoader(trainset, batch_size=32, shuffle=True)
        test_loader = DataLoader(testset, batch_size=32, shuffle=False)
        return train_loader, test_loader