import keras
from keras.layers import Dense
from keras import optimizers
from dataset import create_dataset
from preprocessing import preprocess_data
from keras import metrics
import keras.backend as K
import matplotlib.pyplot as plt
from helpers.logs import create_logger
import numpy as np
import pandas as pd

import warnings

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

logger = create_logger(__name__)


class KerasSDNN:
    """
    Keras version of the source differential neural network architecture
    the Implementation was made based on the Paper: A new source difference ANN for enhanced positioning accuracy
    """

    def __init__(self, coarse_hidden, coarse_lr, diff_hidden, diff_lr):
        """

        :param coarse_hidden: int => Number of hidden neurons in the coarse model
        :param diff_hidden: int => Number of hidden neurons in the difference model
        """
        self.default_coarse_optimizer = optimizers.SGD(learning_rate=coarse_lr)
        self.default_diff_optimizer = optimizers.SGD(learning_rate=diff_lr)

        x, y = create_dataset(polynominal_data=False,
                              length=1000,
                              normalize=False,
                              reshape_target=True)

        self.x, self.y = x, y
        self.x_train, self.y_train, self.x_test, self.y_test = preprocess_data(x=x, y=y, split=True, keras=True)
        self.m, self.n = x.shape
        self.coarse_model = self._create_model(name='coarse',
                                               n_hidden=coarse_hidden,
                                               optimizer=self.default_coarse_optimizer)

        self.difference_model = self._create_model(name='difference',
                                                   n_hidden=diff_hidden,
                                                   optimizer=self.default_diff_optimizer)

        self.coarse_trained = False
        self.diff_trained = False
        self.coarse_weights_saved = False
        self.diff_weights_saved = False

        print(f"x shape: {x.shape} | y shape: {y.shape}")

    def rmse(self, y_true, y_pred):
        """
            custom root mean squared error function to evaluate the model
            :param y_true: Tensor => Target value
            :param y_pred: Tensor => predictions of the model
        """
        return K.sqrt(metrics.mse(y_true, y_pred))

    def custom_score(self, y_true, y_pred):

        error = y_true - y_pred
        squared_err = error ** 2
        mse = np.mean(squared_err)
        rmse = np.sqrt(mse)

        return mse, rmse

    def _create_model(self, name, n_hidden, optimizer, activation='relu', loss_func='mean_squared_error'):
        """
        create and return a compiled keras model
        :param n_hidden: number of hidden neurons of the model
        :param activation: activation function of the model
        :param optimizer: optimizer that will be used through the learning process
        :param loss_func: loss function that will be used for back propagation and for evaluating the model
        :return: instance of a compiled keras model
        """
        model = keras.Sequential()
        model.add(Dense(units=n_hidden, activation=activation, input_dim=self.n, kernel_initializer='normal'))
        model.add(Dense(units=1, activation='linear'))
        weights_path = f'keras_version/compile/{name}.h5'

        if name == 'coarse':
            model.save_weights(filepath=weights_path)
            self.coarse_weights_saved = True

        elif name == 'difference':
            model.save_weights(filepath=weights_path)
            self.diff_weights_saved = True

        model.compile(optimizer=optimizer, loss=loss_func, metrics=[self.rmse])
        return model

    def get_models_by_name(self, model_name='all'):

        if model_name == 'coarse':
            return self.coarse_model

        elif model_name == 'difference':
            return self.difference_model

        else:
            return self.coarse_model, self.difference_model

    def recompile_model(self, model_name, new_optimizer, new_loss, new_metric):

        if model_name == 'coarse' and not self.coarse_weights_saved:
            logger.warning("you need to save the initial coarse weights models to recompile it later")

        elif model_name == 'difference' and not self.diff_weights_saved:
            logger.warning("you need to save the initial difference weights models to recompile it later")

        model = self.get_models_by_name(model_name=model_name)

        model.compile(optimizer=new_optimizer,
                      loss=new_loss,
                      metrics=[new_metric])

    def plot_accuracy(self, history):

        plt.plot(history.history['rmse'])
        plt.plot(history.history['val_rmse'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def plot_losses(self, history):

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def train_coarse_model(self, batch_size=None, validation_split=None, epochs=10, plot_losses=True, plot_accuracy=False):
        """
        train the coarse model
        :param batch_size: int => batch_size used for the training
        :param validation_split: float => validation size from the dataset
        :param epochs: int => training epochs
        :param plot_losses: bool => whether to plot the losses over time or not
        :param plot_accuracy: bool => whether to plot the accuracy change over time or not
        :return: loss of the trained model
        """
        history = self.coarse_model.fit(x=self.x_train,
                                        y=self.y_train,
                                        batch_size=batch_size,
                                        validation_split=validation_split,
                                        epochs=epochs)
        self.coarse_trained = True

        if plot_accuracy:
            self.plot_accuracy(history)

        if plot_losses:
            self.plot_losses(history)

        return history.history

    def train_difference_model(self, batch_size=None, validation_split=None, epochs=10, plot_losses=True, plot_accuracy=False):
        """
        train the difference model
        :param batch_size: int => batch_size used for the training
        :param validation_split: float => validation size from the dataset
        :param epochs: int => training epochs
        :param plot_losses: bool => whether to plot the losses over time or not
        :param plot_accuracy: bool => whether to plot the accuracy change over time or not
        :return: loss of the trained model
        """
        if not self.coarse_trained:
            logger.error("cannot prepare difference Model data without training the coarse Model first!!!")
            raise Exception("You should train the coarse Model First.")

        x_diff_train, y_diff_train, x_diff_test, y_diff_test = self.__prepare_difference_model_data()
        # print(x_diff_train.shape, y_diff_train.shape, y_diff_test.shape)
        # exit()
        history = self.difference_model.fit(x=x_diff_train,
                                            y=y_diff_train,
                                            batch_size=batch_size,
                                            validation_split=validation_split,
                                            epochs=epochs)

        self.diff_trained = True

        if plot_accuracy:
            self.plot_accuracy(history)

        if plot_losses:
            self.plot_losses(history)
        return history.history

    def coarse_predict(self, x):
        """
        return the predictions of the coarse Model
        :param x: torch or numpy array => data to predict on
        """
        if not self.coarse_trained:
            logger.warning("You didn't train the coarse Model yet!")

        return self.coarse_model.predict(x)

    def diff_predict(self, x):
        """
        return the predictions of the difference Model
        :param x: torch or numpy array => data to predict on
        """
        if not self.diff_trained:
            logger.warning("you didn't trained the difference Model yet!")

        return self.difference_model.predict(x)

    def __prepare_difference_model_data(self, split=True):
        if not self.coarse_trained:
            logger.error("cannot prepare difference Model data without training the coarse Model first!!!")
            raise Exception("You should train the coarse Model First.")

        # get the coarse Model Prediction:
        coarse_preds = self.coarse_predict(x=self.x)
        diff_target = self.y - coarse_preds
        diff_dataset = np.append(coarse_preds, diff_target, axis=1)

        x_diff = diff_dataset[:, :-1]
        y_diff = diff_dataset[:, -1].reshape(-1, 1)
        # print(f"dataset shape= {diff_dataset.shape} | diff_target shape= {diff_target.shape} | x_diff={x_diff.shape}"
        #       f" | y_diff={y_diff.shape} | self.y shape= {self.y.shape} | coarse_preds shape: {coarse_preds.shape}")

        return preprocess_data(x_diff, y_diff, split=split, keras=True)

    def sdnn_predict(self, x):
        if not self.coarse_trained or not self.diff_trained:
            logger.warning("you didn't trained the coarse or the difference Model yet!"
                           " this can lead to unwanted/wierd results")

        coarse_predictions = self.coarse_predict(x)
        diff_predictions = self.diff_predict(x)
        y_final = coarse_predictions + diff_predictions
        return y_final

    def evaluate_coarse_model(self, on_test_data=True, plot_predictions=False):

        x, y = (self.x_train, self.y_train) if not on_test_data else (self.x_test, self.y_test)

        loss, score = self.coarse_model.evaluate(x=x, y=y)

        if plot_predictions:
            x_sort = np.sort(x, axis=0)
            y_pred = self.coarse_predict(x_sort)
            plt.scatter(x=x, y=y, c='r', marker='o')
            plt.plot(x_sort, y_pred, color='g')
            plt.show()

        return loss, score

    def evaluate_sdnn_model(self, on_test_data=True, plot_predictions=False):

        x, y = (self.x_train, self.y_train) if not on_test_data else (self.x_test, self.y_test)
        y_final = self.sdnn_predict(x)

        sdnn_mse, sdnn_rmse = self.custom_score(y, y_final)

        if plot_predictions:
            x_sort = np.sort(x, axis=0)
            y_final = self.sdnn_predict(x_sort)
            plt.scatter(x=x, y=y, c='r', marker='o')
            plt.plot(x_sort, y_final, color='g')
            plt.show()

        return sdnn_mse, sdnn_rmse

    def save_predictions_as_csv(self, on_test_data=True):
        x, y = (self.x_train, self.y_train) if not on_test_data else (self.x_test, self.y_test)
        y_coarse = self.coarse_predict(x)
        y_final = self.sdnn_predict(x)

        results = {"coarse_predictions": y_coarse.flatten(),
                   "y_final": y_final.flatten(),
                   "y_true": y.flatten()
                   }
        pd.DataFrame(data=results).to_csv('keras_version/results/predictions/predictions.csv', index=False)

    def reset(self):
        self.coarse_trained = False
        self.diff_trained = False
        self.coarse_weights_saved = False
        self.diff_weights_saved = False


if __name__ == '__main__':

    sdnn = KerasSDNN(coarse_hidden=10,
                     diff_hidden=30,
                     coarse_lr=1e-4,
                     diff_lr=1e-4)

    coarse_his = sdnn.train_coarse_model(epochs=100, validation_split=0.01, plot_losses=True)

    diff_his = sdnn.train_difference_model(epochs=100, validation_split=0.01, plot_losses=True)

    coarse_loss, coarse_score = sdnn.evaluate_coarse_model(plot_predictions=True)
    sdnn_loss, sdnn_score = sdnn.evaluate_sdnn_model(plot_predictions=True)

    logger.info(f"coarse loss= {coarse_loss} | coarse_score= {coarse_score}")
    logger.info(f"sdnn_loss= {sdnn_loss} | sdnn_score= {sdnn_score}")

    sdnn.save_predictions_as_csv()
