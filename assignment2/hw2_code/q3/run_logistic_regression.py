from check_grad import check_grad
from utils import *
from logistic import *

import matplotlib.pyplot as plt
import numpy as np


def run_logistic_regression():
    # train_inputs, train_targets = load_train()
    train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()

    N, M = train_inputs.shape

    #####################################################################
    # TODO:                                                             #
    # Set the hyperparameters for the learning rate, the number         #
    # of iterations, and the way in which you initialize the weights.   #
    #####################################################################
    hyperparameters = {
        "learning_rate": 0.2,
        "weight_regularization": 1.0,
        "num_iterations": 500
    }
    weights = np.zeros((M + 1, 1))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyperparameters)

    # Begin learning with gradient descent
    #####################################################################
    # TODO:                                                             #
    # Modify this section to perform gradient descent, create plots,    #
    # and compute test error.                                           #
    #####################################################################
    iterations_range = range(hyperparameters["num_iterations"])
    validation_errors = []
    training_errors = []
    for t in iterations_range:
        # Gradient descent
        f, df, y = logistic(weights, train_inputs, train_targets, hyperparameters)

        weights = weights - hyperparameters["learning_rate"] * df

        # Evaluation
        y_validation = logistic_predict(weights, valid_inputs)
        ce_validation, frac_correct_validation = evaluate(valid_targets, y_validation)

        ce_train, frac_correct_train = evaluate(train_targets, y)

        validation_errors.append(ce_validation)
        training_errors.append(ce_train)

    test_inputs, test_targets = load_test()

    y = logistic_predict(weights, train_inputs)
    y_validation = logistic_predict(weights, valid_inputs)
    y_test = logistic_predict(weights, test_inputs)
    ce_train, frac_correct_train = evaluate(train_targets, y)
    ce_validation, frac_correct_validation = evaluate(valid_targets, y_validation)
    ce_test, frac_correct_test = evaluate(test_targets, y_test)
    print(f'Classification accuracy on training set: {frac_correct_train}')
    print(f'Classification accuracy on validation set: {frac_correct_validation}')
    print(f'Classification accuracy on test set: {frac_correct_test}')
    print(f"Cross entropy on training set: {ce_train}")
    print(f"Cross entropy on validation set: {ce_validation}")
    print(f"Cross entropy on test set: {ce_test}")


    # Plotting
    plt.figure(figsize=(10, 6))  # Define the figure size
    plt.plot(training_errors, label="Training Error")  # Plot training error
    plt.plot(validation_errors, label="Validation Error")  # Plot validation error
    plt.title('Training Curves')  # Set the title
    plt.xlabel("Iteration")  # Label x-axis
    plt.ylabel("Cross-Entropy Loss")  # Label y-axis
    plt.legend()  # Show legend
    plt.grid(True)  # Show grid
    plt.savefig('../../figures/mnist_train_small.png')  # Display the plot

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def run_check_grad(hyperparameters):
    """ Performs gradient check on logistic function.
    :return: None
    """
    # This creates small random data with 20 examples and
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10

    weights = np.random.randn(num_dimensions + 1, 1)
    data = np.random.randn(num_examples, num_dimensions)
    targets = np.random.rand(num_examples, 1)

    diff = check_grad(logistic,
                      weights,
                      0.001,
                      data,
                      targets,
                      hyperparameters)

    print("diff =", diff)


if __name__ == "__main__":
    run_logistic_regression()
