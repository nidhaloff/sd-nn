import torch.nn as nn
from torch_lr_finder import LRFinder
import torch
import numpy as np
from sklearn.metrics import r2_score
from helpers.logs import create_logger
from tqdm import trange

train_logger = create_logger(__name__)


# training function => will perform a single training step given a batch
def training_func(model, criterion, optimizer):
    def train_step(sample, target):
        model.train()
        optimizer.zero_grad()
        out = model(sample)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        return loss.item()
    return train_step


def find_best_learning_rate(model, criterion, optimizer, loader, mode='linear', method='fast'):
    """
    plot the learning rate with respect to loss change and find out the best learning rate to use for the Model
    :param model: model to perform the method on
    :param criterion: loss function of the model
    :param optimizer: optimizer used for the model
    :param loader: data loader
    :param mode: linear or exponential
    :param method: based on which paper. fast is based on the fastai method, leslie is based on leslie's paper on the subject
    :return: None
    """

    lr_finder = LRFinder(model=model,
                         optimizer=optimizer,
                         criterion=criterion)
    if method == 'fast':
        lr_finder.range_test(train_loader=loader, end_lr=100, num_iter=100)
        lr_finder.plot()

    elif method == 'leslie':
        lr_finder.range_test(train_loader=loader, end_lr=1, num_iter=100, step_mode=mode)
        lr_finder.plot(log_lr=False)


# fit function will train the given Model over defined epochs
def fit(model, loss_func, optimizer, learning_rate, train_loader, test_loader, epochs):

    criterion = loss_func
    optimizer = optimizer(model.parameters(), lr=learning_rate)

    train_losses, test_losses, accuracy_list = ([] for _ in range(3))
    avg_train_losses, avg_test_losses, avg_accuracy_list = ([] for _ in range(3))
    train = training_func(model=model, criterion=criterion, optimizer=optimizer)
    count = 0
    for e in trange(epochs):

        for i, (x, y) in enumerate(train_loader):
            loss_val = train(x, y)
            count += 1
            train_losses.append(loss_val)

        else:
            with torch.no_grad():
                model.eval()
                for x_test, y_test in test_loader:
                    pred = model(x_test)
                    loss = criterion(pred, y_test)

                    test_losses.append(loss.item())
                    accuracy = np.sqrt(loss.item())  # r2_score(y_test.numpy(), pred.numpy()) | or rmse
                    accuracy_list.append(accuracy)

        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_test_loss = sum(test_losses) / len(test_losses)
        avg_acc = sum(accuracy_list) / len(accuracy_list)

        avg_train_losses.append(avg_train_loss)
        avg_test_losses.append(avg_test_loss)
        avg_accuracy_list.append(avg_acc)

        if count % 80 == 0:

            print(f"Epoch: {e} => average training loss= {avg_train_loss} average test losses= {avg_test_loss} and accuracy= {avg_acc}")
            print(f"Epoch: {e} => training loss={train_losses[-1]} | test loss={test_losses[-1]} | accuracy={accuracy_list[-1]}")

    return avg_train_losses, avg_test_losses, avg_accuracy_list
