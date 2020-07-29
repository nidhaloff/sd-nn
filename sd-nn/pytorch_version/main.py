from pytorch_version.models.coarse import CoarseModel
from pytorch_version.models.difference import DifferenceModel
from dataset import create_dataset
from preprocessing import preprocess_data, create_loaders
from pytorch_version.train import fit, find_best_learning_rate
from pytorch_version.plots import plot_train_results, plot_test_results
import numpy as np
import pandas as pd
import torch
from torch import nn
from matplotlib import pyplot as plt
from torch import optim
from sklearn.metrics import r2_score
from helpers.logs import create_logger
from torch_lr_finder import LRFinder
import sys

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

logger = create_logger(__name__)


class PytorchSDNN:
    """
      Pytorch version of the source differential neural network architecture
      the Implementation was made based on the Paper: A new source difference ANN for enhanced positioning accuracy
      """

    def __init__(self, coarse_hidden, diff_hidden):
        """

        :param coarse_hidden: number of hidden neurons in the coarse neural network
        :param diff_hidden: number of hidden neurons in the difference neural network
        """
        x, y = create_dataset(typ='sin', length=1000, normalize=False, reshape_target=True)
        self.x, self.y = x, y
        self.x_train, self.y_train, self.x_test, self.y_test = preprocess_data(x=x,
                                                                               y=y,
                                                                               split=True,
                                                                               keras=False)

        self.train_loader, self.test_loader = create_loaders(self.x_train, self.y_train, self.x_test, self.y_test)
        self.m, self.n = x.shape
        self.coarse_model = self._create_model(coarse_hidden, name='coarse')
        self.difference_model = self._create_model(diff_hidden, name='difference')
        self.coarse_trained = False
        self.diff_loader_created = False
        self.diff_trained = False

    def get_models_by_name(self, model_name='all'):
        """
        get a neural network model by name
        :param model_name: 'coarse' for getting the coarse model | 'difference' for getting the difference model
        :return: a specific neural network model
        """
        if model_name == 'coarse':
            return self.coarse_model

        elif model_name == 'difference':
            return self.difference_model

        elif model_name == 'all':
            return self.coarse_model, self.difference_model

        else:
            logger.warning(f"{model_name} is not valid. choose between (coarse, difference or all)")

    def _create_model(self, n_hidden, name='coarse'):
        """
        create and return a neural network Model
        :param n_hidden: number of hidden neurons
        :param name: name of the model
        :return: neural network model
        """
        if name == 'coarse':
            model = CoarseModel(n_features=self.n, n_hidden=n_hidden, n_out=1)
            return model
        else:
            model = DifferenceModel(n_features=self.n, n_hidden=n_hidden, n_out=1)
            return model

    def find_best_learning_rate(self, model_name, loader, start_lr, criterion=nn.MSELoss(), opt=optim.SGD):
        """
        find a suitable learning rate value by plotting losses over learning rate curve
        :param model_name: name of the model
        :param loader: loader of the data
        :param start_lr: learning rate to start with
        :param criterion: loss function to use on the model
        :param opt: optimizer passed as a reference to the function or class Optimizer
        :return: None
        """

        model = self.coarse_model if model_name == 'coarse' else self.difference_model
        optimizer = opt(model.parameters(), lr=start_lr)

        lr_finder = LRFinder(model=model,
                             optimizer=optimizer,
                             criterion=criterion)

        lr_finder.range_test(loader, end_lr=1, num_iter=100, step_mode='exp')
        lr_finder.plot(log_lr=True)

    def train_coarse_model(self, loss_func, optimizer, learning_rate, epochs, plot_results=False, plot_best_lr=False):
        """
        train the coarse Model
        :param loss_func: loss function used for training
        :param optimizer: optimizer used for training
        :param learning_rate: learning rate used for training
        :param epochs: number of epochs used for training
        :param plot_results: whether to plot the change of train and test losses during training
        :param plot_best_lr: when set to True, plot the losses over learning rate
        :return: the trained model and state dictionary of the model
        """
        if plot_best_lr:
            self.find_best_learning_rate(model_name='coarse',
                                         loader=self.train_loader,
                                         start_lr=learning_rate,
                                         criterion=loss_func,
                                         opt=optimizer)

        train_losses, test_losses, accs = fit(model=self.coarse_model,
                                              loss_func=loss_func,
                                              optimizer=optimizer,
                                              learning_rate=learning_rate,
                                              train_loader=self.train_loader,
                                              test_loader=self.test_loader,
                                              epochs=epochs)
        if plot_results:
            plot_train_results(train_losses=train_losses, test_losses=test_losses, accs=accs)

        self.coarse_trained = True
        torch.save(self.coarse_model, 'results/models/coarse_model.pth')
        torch.save(self.coarse_model.state_dict(), 'results/state_dics/coarse_model.pth')
        return self.coarse_model, self.coarse_model.state_dict()

    def train_difference_model(self, loss_func, optimizer, learning_rate, epochs, plot_results=False,
                               plot_best_lr=False):
        """
          train the difference Model. Notice that the coarse Model should be trained first!
          :param loss_func: loss function used for training
          :param optimizer: optimizer used for training
          :param learning_rate: learning rate used for training
          :param epochs: number of epochs used for training
          :param plot_results: whether to plot the change of train and test losses during training
          :param plot_best_lr: when set to True, plot the losses over learning rate
          :return: the trained model and state dictionary of the model
          """

        if not self.coarse_trained:
            logger.error("cannot train difference Model without training the Coarse Model First!!!")
            raise Exception("You should train the coarse Model First.")

        diff_train_loader, diff_test_loader = self.__prepare_difference_model_data()

        if plot_best_lr:
            self.find_best_learning_rate(model_name='difference',
                                         start_lr=learning_rate,
                                         criterion=loss_func,
                                         opt=optimizer,
                                         loader=self.diff_train_loader)

        diff_train_losses, diff_test_losses, diff_accs = fit(model=self.difference_model,
                                                             loss_func=loss_func,
                                                             optimizer=optimizer,
                                                             learning_rate=learning_rate,
                                                             train_loader=diff_train_loader,
                                                             test_loader=diff_test_loader,
                                                             epochs=epochs)
        if plot_results:
            plot_train_results(diff_train_losses, diff_test_losses, diff_accs)

        torch.save(self.difference_model, 'results/models/difference_model.pth')
        torch.save(self.difference_model.state_dict(), 'results/state_dics/difference_model.pth')
        self.diff_trained = True
        return self.difference_model, self.difference_model.state_dict()

    def __prepare_difference_model_data(self):
        """
        prepare the data that will be used to train the difference model by taking the difference
        between the target values and the coarse predictions
        :return: train and test loader for the new data to train the difference model
        """
        if not self.coarse_trained:
            logger.error("cannot prepare difference Model data without training the coarse Model first!!!")
            raise Exception("You should train the coarse Model First.")
        # get coarse model predictions
        coarse_train_predictions = self.coarse_model(self.x_train).detach().numpy()
        coarse_test_predictions = self.coarse_model(self.x_test).detach().numpy()
        # get the difference/deviation of the coarse model predictions from the real target data
        d_diff_train = self.y_train.numpy() - coarse_train_predictions
        d_diff_test = self.y_test.numpy() - coarse_test_predictions
        # create a new trainset and testset for the difference model
        diff_trainset = np.append(self.x_train.numpy(), d_diff_train, axis=1)
        diff_testset = np.append(self.x_test.numpy(), d_diff_test, axis=1)
        # append the two datasets to create a new datasets to train the difference model on it
        diff_dataset = np.append(diff_trainset, diff_testset, axis=0)
        # split the dataset into inputs and target
        x_diff = diff_dataset[:, :-1]
        y_diff = diff_dataset[:, -1].reshape(-1, 1)

        # split the dataset into traindata and validationdata
        x_diff_train, y_diff_train, x_diff_test, y_diff_test = preprocess_data(x_diff, y_diff, split=True, keras=False)
        self.diff_train_loader, self.diff_test_loader = create_loaders(x_diff_train, y_diff_train, x_diff_test,
                                                                       y_diff_test)
        self.diff_loader_created = True
        return self.diff_train_loader, self.diff_test_loader

    def coarse_predict(self, x):
        """
        get the coarse model predictions on data x
        :param x: data to predict on
        :return: predictions of the coarse model as a tensor
        """
        self.coarse_model.eval()
        return self.coarse_model(x)

    def difference_predict(self, x):
        """
        get the difference model predictions on data x that will be used later to define the final predictions of the sd-nn
        :param x: data to predict on
        :return: predictions of the difference model as a tensor
        """
        self.difference_model.eval()
        return self.difference_model(x)

    def sdnn_predict(self, x):
        """
        get the predictions of the sd-nn model on some data x
        :param x: data to predict on
        :return: predictions of the sd-nn model as a tensor
        """
        self.coarse_model.eval()
        self.difference_model.eval()

        coarse_preds = self.coarse_predict(x)
        diff_preds = self.difference_predict(x)
        y_final_preds = coarse_preds + diff_preds
        return y_final_preds

    def compare_coarse_sdnn(self, on_test_data=True, plot_results=False):
        """
        compare the coarse and the final sd-nn models
        :param on_test_data: whether to compare according to test data or train data
        :param plot_results: if set to True plot the results of the comparison
        :return: None
        """
        train_data = self.x_test if on_test_data else self.x_train
        target = self.y_test if on_test_data else self.y_train
        with torch.no_grad():
            self.coarse_model.eval()
            self.difference_model.eval()
            x_sort, idx = torch.sort(train_data, 0)

            criterion = torch.nn.MSELoss()
            coarse_preds = self.coarse_predict(train_data)
            y_final = self.sdnn_predict(train_data)
            torch.save(y_final, 'results/models/y_final.pth')
            # predictions = np.append(coarse_preds.numpy(), y_final.numpy(), axis=1)
            # np.savetxt(fname='./results/predictions/predictions.csv', X=predictions, delimiter=',')
            results = {"coarse_predictions": coarse_preds.numpy().flatten(),
                       "y_final": y_final.numpy().flatten(),
                       "y_true": target.numpy().flatten()
                       }
            pd.DataFrame(data=results).to_csv('results/predictions/predictions.csv', index=False)

            coarse_loss = criterion(coarse_preds, target)
            y_final_loss = criterion(y_final, target)
            logger.info(f"Coarse loss = {coarse_loss.item()} | y_final loss = {y_final_loss.item()}")

            if plot_results:
                plt.scatter(train_data.numpy(), target.numpy(), c="r", marker="o", label='trueTarget')
                plt.plot(x_sort.numpy(), coarse_preds[idx[:, 0]].numpy(), color="g", label='coarsePredictions')
                plt.plot(x_sort.numpy(), y_final[idx[:, 0]].numpy(), color="k", label='sdnnPredictions')
                plt.legend()
                plt.title('comparison of coarse and sd-nn predictions')
                plt.show()

    def evaluate_coarse_model(self, with_test_data=True):
        """
        evaluate the coarse Model
        :param with_test_data: if True, evaluate using the test data else evaluate using the train data
        :return: the loss and the score of the coarse model
        """
        train_data = self.x_test if with_test_data else self.x_train
        target = self.y_test if with_test_data else self.y_train

        with torch.no_grad():
            self.coarse_model.eval()
            criterion = torch.nn.MSELoss()
            coarse_preds = self.coarse_predict(train_data)
            coarse_loss = criterion(coarse_preds, target)
            coarse_score = r2_score(target.numpy(), coarse_preds.numpy())
            logger.info(f"Coarse loss = {coarse_loss.item()} | Coarse R2 Score= {coarse_score}")
            return coarse_loss, coarse_score

    def evaluate_sdnn_model(self, with_test_data=True):
        """
        evaluate the sd-nn Model
        :param with_test_data: if True, evaluate using the test data else evaluate using the train data
        :return: the loss and the score of the final sd-nn model
        """

        train_data = self.x_test if with_test_data else self.x_train
        target = self.y_test if with_test_data else self.y_train
        #
        with torch.no_grad():
            criterion = torch.nn.MSELoss()
            y_final = self.sdnn_predict(train_data)
            y_finalloss = criterion(y_final, target)
            y_final_score = r2_score(target.numpy(), y_final.numpy())
            logger.info(f"Y_final loss = {y_finalloss.item()} | Y_final Score = {y_final_score.item()}")
            return y_finalloss, y_final_score

    def plot_fitting(self, model_name='all', on_test_data=True):
        """
        plot the best fit line and see how it fits the real data
        :param model_name: indicate which model to use for prediction
        :param on_test_data: whether to use test data or train data to evaluate the model
        :return: None
        """
        x, y = (self.x_train, self.y_train) if not on_test_data else (self.x_test, self.y_test)
        x_sort, idx = torch.sort(x, 0)

        if model_name == 'all':
            y_coarse = self.coarse_predict(x_sort)
            y_final = self.sdnn_predict(x_sort)
            plt.scatter(x=x.numpy(), y=y.numpy(), c='r', marker='o', label='target')
            plt.plot(x_sort.numpy(), y_coarse.detach().numpy(), 'g', label='coarsePredictions')
            plt.plot(x_sort.numpy(), y_final.detach().numpy(), 'k', label='sdnnPredictions')
            plt.xlabel('x data')
            plt.ylabel('predictions')
            plt.legend()
            plt.show()
            return

        else:

            y_pred = self.coarse_predict(x_sort) if model_name == 'coarse' else self.sdnn_predict(x_sort)
            plt.scatter(x=x.numpy(), y=y.numpy(), c='r', marker='o', label='target')
            plt.plot(x_sort.numpy(), y_pred.detach().numpy(), 'g', label='predictions')
            plt.xlabel('x data')
            plt.ylabel('predictions')
            plt.legend()
            plt.show()

if __name__ == '__main__':

    sdnn = SDNN(coarse_hidden=1000, diff_hidden=1000)
    coarse_model, coarse_res = sdnn.train_coarse_model(loss_func=nn.MSELoss(),
                                                       learning_rate=1e-3,
                                                       optimizer=optim.Adam,
                                                       epochs=100,
                                                       plot_results=False,
                                                       plot_best_lr=False)

    diff_model, diff_res = sdnn.train_difference_model(loss_func=nn.MSELoss(),
                                                       learning_rate=1e-4,
                                                       optimizer=optim.Adam,
                                                       epochs=100,
                                                       plot_results=False,
                                                       plot_best_lr=False)
    #
    coarse_loss, coarse_score = sdnn.evaluate_coarse_model()
    sdnn_loss, sdnn_score = sdnn.evaluate_sdnn_model()
    sdnn.compare_coarse_sdnn(on_test_data=True, plot_results=True)
    sdnn.plot_fitting()
    sdnn.plot_fitting(model_name='sd-nn')
