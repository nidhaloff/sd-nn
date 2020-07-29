import matplotlib.pyplot as plt


# plot the results of the training and testing loss values over the iterations
def plot_train_results(train_losses, test_losses, accs):
    plt.plot(train_losses, color="b", label='train losses')
    plt.title("train losses over epochs")
    plt.xlabel('iterations')
    plt.ylabel('train losses')
    plt.legend()
    plt.show()

    plt.plot(test_losses, color="b", label='test losses')
    plt.title("test losses over epochs")
    plt.xlabel('iterations')
    plt.ylabel('test losses')
    plt.legend()
    plt.show()

    plt.plot(accs, color="b", label='score/accuracy')
    plt.title("accuracy over epochs")
    plt.xlabel('iterations')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()


def plot_test_results(in_data, target, coarse_predictions, y_final_predictions):

    plt.scatter(in_data, target, c="r", marker="o", label='trueTarget')
    plt.plot(in_data, coarse_predictions, color="g", label='coarsePredictions')
    plt.plot(in_data, y_final_predictions, color="k", label='sdnnPredictions')
    plt.legend()
    plt.title('comparison of coarse and sd-nn predictions')
    plt.show()

    # plt.scatter(in_data, target, c="r", marker="o")
    # plt.scatter(in_data, coarse_predictions, c="g", marker="+")
    # plt.scatter(in_data, y_final_predictions, c="k", marker="*")
    # plt.show()

