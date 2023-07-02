from l2_distance import l2_distance
from utils import *

import matplotlib.pyplot as plt
import numpy as np


def knn(k, train_data, train_labels: np.array, valid_data):
    """ Uses the supplied training inputs and labels to make
    predictions for validation data using the K-nearest neighbours
    algorithm.

    Note: N_TRAIN is the number of training examples,
          N_VALID is the number of validation examples,
          M is the number of features per example.

    :param k: The number of neighbours to use for classification
    of a validation example.
    :param train_data: N_TRAIN x M array of training data.
    :param train_labels: N_TRAIN x 1 vector of training labels
    corresponding to the examples in train_data (must be binary).
    :param valid_data: N_VALID x M array of data to
    predict classes for validation data.
    :return: N_VALID x 1 vector of predicted labels for
    the validation data.
    """
    dist = l2_distance(valid_data.T, train_data.T)
    nearest = np.argsort(dist, axis=1)[:, :k]

    train_labels = train_labels.reshape(-1)
    valid_labels = train_labels[nearest]

    # Note this only works for binary labels:
    valid_labels = (np.mean(valid_labels, axis=1) >= 0.5).astype(np.int32)
    valid_labels = valid_labels.reshape(-1, 1)

    return valid_labels


def compute_classification_rate(predictions: np.array, targets: np.array):
    return np.equal(predictions, targets).sum() / predictions.size


def run_knn():

    def get_class_accuracy(k: int, inputs: np.array, targets: np.array):
        pred = knn(k, train_inputs, train_targets, inputs)
        return compute_classification_rate(pred, targets)

    train_inputs, train_targets = load_train()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()

    #####################################################################
    # DONE                                                              #
    # Implement a function that runs kNN for different values of k,     #
    # plots the classification rate on the validation set, and etc.     #
    #####################################################################
    k_range = np.arange(1, 10, 2)
    classification_rates = np.zeros(k_range.size)

    for i in range(k_range.size):
        classification_rates[i] = get_class_accuracy(k_range[i], valid_inputs, valid_targets)

    # plotting code

    plt.figure(figsize=(8, 6))  # creating a figure with desired size

    # Scatter plot
    plt.scatter(k_range, classification_rates, marker='x', color='purple', label='Validation Accuracy')
    plt.plot(k_range, classification_rates, color='grey')

    # Adding title and labels
    plt.title('KNN Classification Accuracy on Validation Set')
    plt.xlabel('k (number of neighbors)')
    plt.ylabel('Classification Accuracy')

    # Adding grid
    plt.grid(True)

    # Adding legend
    plt.legend()

    # Displaying the plot
    plt.savefig('../../figures/knn_classification_accuracy.png')

    # now with a chosen value of k
    k_star = 3

    print(f"Classification accuracies for k_star={k_star}")
    print(f"\tValidation accuracy: {get_class_accuracy(k_star, valid_inputs, valid_targets)}")
    print(f"\tTest accuracy: {get_class_accuracy(k_star, test_inputs, test_targets)}")

    print(f"Classification accuracies for (k_star + 2)={k_star+2}")
    print(f"\tValidation accuracy: {get_class_accuracy(k_star+2, valid_inputs, valid_targets)}")
    print(f"\tTest accuracy: {get_class_accuracy(k_star+2, test_inputs, test_targets)}")

    print(f"Classification accuracies for (k_star - 2)={k_star - 2}")
    print(f"\tValidation accuracy: {get_class_accuracy(k_star - 2, valid_inputs, valid_targets)}")
    print(f"\tTest accuracy: {get_class_accuracy(k_star - 2, test_inputs, test_targets)}")

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    run_knn()
