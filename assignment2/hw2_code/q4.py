# -*- coding: utf-8 -*-
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506, 1)), x), axis=1)  # add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))


# helper function
def l2(A, B):
    """
    Input: A is a Nxd matrix
           B is a Mxd matrix
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    """
    A_norm = (A ** 2).sum(axis=1).reshape(A.shape[0], 1)
    B_norm = (B ** 2).sum(axis=1).reshape(1, B.shape[0])
    dist = A_norm + B_norm - 2 * A.dot(B.transpose())
    return dist


# to implement
def LRLS(test_datum, x_train, y_train, tau, lam=1e-5):
    """
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    """

    assert test_datum.shape == (d, 1)
    assert x_train.shape[0] == y_train.shape[0]
    assert x_train.shape[1] == d

    N_train = x_train.shape[0]

    # Compute distance-based weights for each training example
    distances = l2(x_train, test_datum.transpose())  # N_train x 1 vector containing distances from test_datum to rows
    # of x_train
    exp_distances = np.exp(- distances / (2 * tau ** 2)).reshape(N_train)
    A = np.diag(exp_distances / exp_distances.sum())

    # Compute the optimal weights
    w = LA.inv((x_train.transpose() @ A @ x_train + lam * np.eye(d))) @ x_train.transpose() @ A @ y_train

    # Return the prediction
    return test_datum.transpose() @ w


def run_validation(x, y, taus, val_frac):
    """
    Input: x is the N x d design matrix
           y is the N x 1 targets vector
           taus is a vector of tau values to evaluate
           val_frac is the fraction of examples to use as validation data
    output is
           a vector of training losses, one for each tau value
           a vector of validation losses, one for each tau value
    """

    x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=val_frac)

    tau_range = np.arange(10, 1000)
    train_losses = np.zeros(tau_range.size)
    valid_losses = np.zeros(tau_range.size)

    for i, tau in enumerate(tau_range):

        # Compute train loss
        train_preds = np.array([LRLS(datum.reshape((d, 1)), x_train, y_train, tau) for datum in x_train])
        train_losses[i] = np.square(train_preds - y_train).mean()

        # Compute validation loss
        valid_preds = np.array([LRLS(datum.reshape((d, 1)), x_train, y_train, tau) for datum in x_validation])
        valid_losses[i] = np.square(valid_preds - y_validation).mean()

    return train_losses, valid_losses



if __name__ == "__main__":
    # In this exercise we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as
    # well if you wish
    taus = np.logspace(1.0, 3, 200)
    train_losses, test_losses = run_validation(x, y, taus, val_frac=0.3)
    plt.semilogx(train_losses)
    plt.semilogx(test_losses)
