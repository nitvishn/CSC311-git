import numpy as np
import os
import gzip
import struct
import array
import matplotlib.pyplot as plt
import matplotlib.image
from urllib.request import urlretrieve


def download(url, filename):
    if not os.path.exists('data'):
        os.makedirs('data')
    out_file = os.path.join('data', filename)
    if not os.path.isfile(out_file):
        urlretrieve(url, out_file)


def mnist():
    base_url = 'http://yann.lecun.com/exdb/mnist/'

    def parse_labels(filename):
        with gzip.open(filename, 'rb') as fh:
            magic, num_data = struct.unpack(">II", fh.read(8))
            return np.array(array.array("B", fh.read()), dtype=np.uint8)

    def parse_images(filename):
        with gzip.open(filename, 'rb') as fh:
            magic, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(num_data, rows, cols)

    for filename in ['train-images-idx3-ubyte.gz',
                     'train-labels-idx1-ubyte.gz',
                     't10k-images-idx3-ubyte.gz',
                     't10k-labels-idx1-ubyte.gz']:
        download(base_url + filename, filename)

    train_images = parse_images('data/train-images-idx3-ubyte.gz')
    train_labels = parse_labels('data/train-labels-idx1-ubyte.gz')
    test_images = parse_images('data/t10k-images-idx3-ubyte.gz')
    test_labels = parse_labels('data/t10k-labels-idx1-ubyte.gz')

    return train_images, train_labels, test_images[:1000], test_labels[:1000]


def load_mnist():
    partial_flatten = lambda x: np.reshape(x, (x.shape[0], np.prod(x.shape[1:])))
    one_hot = lambda x, k: np.array(x[:, None] == np.arange(k)[None, :], dtype=int)
    train_images, train_labels, test_images, test_labels = mnist()
    train_images = (partial_flatten(train_images) / 255.0 > .5).astype(float)
    test_images = (partial_flatten(test_images) / 255.0 > .5).astype(float)
    train_labels = one_hot(train_labels, 10)
    test_labels = one_hot(test_labels, 10)
    N_data = train_images.shape[0]

    return N_data, train_images, train_labels, test_images, test_labels


def plot_images(images, ax, ims_per_row=5, padding=5, digit_dimensions=(28, 28),
                cmap=matplotlib.cm.binary, vmin=None, vmax=None):
    """Images should be a (N_images x pixels) matrix."""
    N_images = images.shape[0]
    N_rows = np.int32(np.ceil(float(N_images) / ims_per_row))
    pad_value = np.min(images.ravel())
    concat_images = np.full(((digit_dimensions[0] + padding) * N_rows + padding,
                             (digit_dimensions[1] + padding) * ims_per_row + padding), pad_value)
    for i in range(N_images):
        cur_image = np.reshape(images[i, :], digit_dimensions)
        row_ix = i // ims_per_row
        col_ix = i % ims_per_row
        row_start = padding + (padding + digit_dimensions[0]) * row_ix
        col_start = padding + (padding + digit_dimensions[1]) * col_ix
        concat_images[row_start: row_start + digit_dimensions[0],
        col_start: col_start + digit_dimensions[1]] = cur_image
        cax = ax.matshow(concat_images, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    return cax


def save_images(images, filename, **kwargs):
    fig = plt.figure(1)
    fig.clf()
    ax = fig.add_subplot(111)
    plot_images(images, ax, **kwargs)
    fig.patch.set_visible(False)
    ax.patch.set_visible(False)
    plt.savefig(filename)


def train_mle_estimator(train_images, train_labels):
    """ Inputs: train_images, train_labels
        Returns the MLE estimators theta_mle and pi_mle"""

    # Shape of train_images: (N_data, 784)
    # Shape of train_labels: (N_data, 10)

    # theta_mle is a matrix of size 784 x 10, where theta_mle[j, c] = p(x_j = 1 | c)
    # pi_mle is a vector of size 10, where pi_mle[c] = p(c)

    N, D = train_images.shape
    N_classes = train_labels.shape[1]

    pi_mle = np.zeros(N_classes)
    num_class = np.sum(train_labels, axis=0)
    theta_mle = train_images.T @ train_labels / num_class

    print("Number of zeroes in MLE:", np.sum(theta_mle == 0))
    print("Number of ones in MLE:", np.sum(theta_mle == 1))

    for c in range(N_classes):
        # pi_mle[c] = #{class = c} / #{all data}
        pi_mle[c] = np.sum(train_labels[:, c]) / N

    return theta_mle, pi_mle


def train_map_estimator(train_images, train_labels):
    """ Inputs: train_images, train_labels
        Returns the MAP estimators theta_map and pi_map"""

    N, D = train_images.shape
    N_classes = train_labels.shape[1]

    alpha, beta = 3, 3

    pi_map = np.zeros(N_classes)  # uniform prior, so same as MLE
    num_class = np.sum(train_labels, axis=0)
    theta_map = ((train_images.T @ train_labels) + alpha - 1) / (num_class + alpha + beta - 2)  # beta prior

    print("Number of zeroes in MAP:", np.sum(theta_map == 0))
    print("Number of ones in MAP:", np.sum(theta_map == 1))

    for c in range(N_classes):
        # pi_map[c] = #{class = c} / #{all data}
        pi_map[c] = (np.sum(train_labels[:, c])) / N

    return theta_map, pi_map


def log_likelihood(images, theta, pi):
    """ Inputs: images, theta, pi
        Returns the matrix 'log_like' of loglikehoods over the input images where
    log_like[i,c] = log p (c |x^(i), theta, pi) using the estimators theta and pi.
    log_like is a matrix of num of images x num of classes
    Note that log likelihood is not only for c^(i), it is for all possible c's."""

    # YOU NEED TO WRITE THIS PART

    # Shape of images: (N_data, 784)
    # Shape of theta: (784, 10)
    # Shape of pi: (10,)

    N_data, D = images.shape
    N_classes = pi.shape[0]

    log_like = np.zeros((N_data, N_classes))

    for i in range(N_data):

        # calculate p(x | theta, pi)

        # prob_image = np.sum([
        #     pi[c] * np.prod([
        #         theta[d, c] ** images[i, d] * (1 - theta[d, c]) ** (1 - images[i, d])
        #         for d in range(D)
        #     ])
        #     for c in range(N_classes)
        # ])
        # log_prob_image = np.log(prob_image)
        # if prob_image == 0:
        #     print("Oops!")

        for c in range(N_classes):
            if np.sum(theta[:, c] == 0) > 0 or np.sum(theta[:, c] == 1) > 0:
                log_like[i, c] = -np.inf
            else:
                log_like[i, c] = np.log(pi[c]) + np.sum(
                    images[i, :] * np.log(theta[:, c]) + (1 - images[i, :]) * np.log(1 - theta[:, c]))

    return log_like


def predict(log_like):
    """ Inputs: matrix of log likelihoods
    Returns the predictions based on log likelihood values"""

    # YOU NEED TO WRITE THIS PART
    # log_like is a matrix of num of images x num of classes

    predictions = np.argmax(log_like, axis=1)

    return predictions  # vector of size num of images, where each element is a class label


def accuracy(log_like, labels):
    """ Inputs: matrix of log likelihoods and 1-of-K labels
    Returns the accuracy based on predictions from log likelihood values"""

    predictions = predict(log_like)
    acc = np.sum(predictions == np.argmax(labels, axis=1)) / labels.shape[0]

    return acc


def main():
    N_data, train_images, train_labels, test_images, test_labels = load_mnist()

    # Fit MLE and MAP estimators
    theta_mle, pi_mle = train_mle_estimator(train_images, train_labels)
    theta_map, pi_map = train_map_estimator(train_images, train_labels)

    # Find the log likelihood of each data point
    loglike_train_mle = log_likelihood(train_images, theta_mle, pi_mle)
    loglike_train_map = log_likelihood(train_images, theta_map, pi_map)

    avg_loglike_mle = np.sum(loglike_train_mle * train_labels) / N_data
    avg_loglike_map = np.sum(loglike_train_map * train_labels) / N_data

    print("Average log-likelihood for MLE is ", avg_loglike_mle)
    print("Average log-likelihood for MAP is ", avg_loglike_map)

    train_accuracy_map = accuracy(loglike_train_map, train_labels)
    loglike_test_map = log_likelihood(test_images, theta_map, pi_map)
    test_accuracy_map = accuracy(loglike_test_map, test_labels)

    print("Training accuracy for MAP is ", train_accuracy_map)
    print("Test accuracy for MAP is ", test_accuracy_map)

    # Plot MLE and MAP estimators
    save_images(theta_mle.T, 'mle.png')
    save_images(theta_map.T, 'map.png')


def q2c():
    N_data, train_images, train_labels, test_images, test_labels = load_mnist()

    theta_mle, pi_mle = train_mle_estimator(train_images, train_labels)

    print(log_likelihood(train_images, theta_mle, pi_mle))

    save_images(theta_mle.T, 'mle.png')


def q2c_but_with_map():
    N_data, train_images, train_labels, test_images, test_labels = load_mnist()

    theta_map, pi_map = train_map_estimator(train_images, train_labels)

    save_images(theta_map.T, 'map.png')


if __name__ == '__main__':
    main()