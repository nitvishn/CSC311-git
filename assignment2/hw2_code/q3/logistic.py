from utils import sigmoid

import numpy as np


def logistic_predict(weights, data: np.array):
    """ Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples
          M is the number of features per example

    :param weights: A vector of weights with dimension (M + 1) x 1, where
    the last element corresponds to the bias (intercept).
    :param data: A matrix with dimension N x M, where each row corresponds to
    one data point.
    :return: A vector of probabilities with dimension N x 1, which is the output
    to the classifier.
    """
    #####################################################################
    # TODO: CHECK                                                                #
    # Given the weights and bias, compute the probabilities predicted   #
    # by the logistic classifier.                                       #
    #####################################################################
    (N, M) = data.shape()
    data_aug = np.hstack((data, np.ones((N, 1))))
    y = sigmoid(data_aug @ weights)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return y


def evaluate(targets, y):
    """ Compute evaluation metrics.

    Note: N is the number of examples
          M is the number of features per example

    :param targets: A vector of targets with dimension N x 1.
    :param y: A vector of probabilities with dimension N x 1.
    :return: A tuple (ce, frac_correct)
        WHERE
        ce: (float) Averaged cross entropy
        frac_correct: (float) Fraction of inputs classified correctly
    """
    #####################################################################
    # TODO: CHECK                                                           #
    # Given targets and probabilities predicted by the classifier,      #
    # return cross entropy and the fraction of inputs classified        #
    # correctly.                                                        #
    #####################################################################
    ce = np.mean(-targets * np.log2(y) - (1 - targets) * np.log2(1 - y))
    frac_correct = np.round(y).equals(targets).sum()/targets.size
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return ce, frac_correct


def logistic(weights, data, targets, hyperparameters):
    """ Calculate the cost and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples
          M is the number of features per example

    :param weights: A vector of weights with dimension (M + 1) x 1, where
    the last element corresponds to the bias (intercept).
    :param data: A matrix with dimension N x M, where each row corresponds to
    one data point.
    :param targets: A vector of targets with dimension N x 1.
    :param hyperparameters: The hyperparameter dictionary.
    :returns: A tuple (f, df, y)
        WHERE
        f: The average of the loss over all data points.
           This is the objective that we want to minimize.
        df: (M + 1) x 1 vector of derivative of f w.r.t. weights.
        y: N x 1 vector of probabilities.
    """
    y = logistic_predict(weights, data)

    #####################################################################
    # TODO: CHECK                                                          #
    # Given weights and data, return the averaged loss over all data    #
    # points, gradient of parameters, and the probabilities given by    #
    # logistic regression.                                              #
    #####################################################################
    (N, M) = data.shape
    one = np.ones((N, 1))

    data_aug = np.hstack((data, one))

    f = np.mean(- targets * np.log2(y) - (one - targets) * np.log2(one - y))
    df = data_aug.transpose() @ (y - targets) / N
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return f, df, y
