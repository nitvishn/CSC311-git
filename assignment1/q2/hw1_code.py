import graphviz as graphviz
import numpy as np
import sklearn.tree
from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

TRAIN_PROPORTION = 0.7  # proportion of data to use for training
TEST_TO_VALIDATE_RATIO = 0.5  # proportion of test:validation data


def load_data():
    """
    Loads the data, preprocesses it using the vectorizer, and splits it into train, validate, and test sets.
    Returns: (X_train, X_validate, X_test, y_train, y_validate, y_test, vectorizer)

    """
    # load clean data
    real_file = open('clean_real.txt', 'r')
    fake_file = open('clean_fake.txt', 'r')

    # build set of words, and store sentences as list of tokens
    real_sentences = [line for line in real_file]
    fake_sentences = [line for line in fake_file]
    corpus = real_sentences + fake_sentences

    # make labels
    labels = np.array(['real'] * len(real_sentences) + ['fake'] * len(fake_sentences))

    # split the data
    X_train, X_test, y_train, y_test = train_test_split(corpus, labels, train_size=TRAIN_PROPORTION)
    X_test, X_validate, y_test, y_validate = train_test_split(X_test, y_test,
                                                              train_size=TEST_TO_VALIDATE_RATIO)

    print(f"Training, validation, test split: ({len(X_train), len(X_test), len(X_validate)})")

    # vectorize the sentences
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_validate = vectorizer.transform(X_validate)
    X_test = vectorizer.transform(X_test)

    # return the train, validate, test data and the vectorizer

    return (X_train, X_validate, X_test, y_train, y_validate, y_test, vectorizer)


def measure_accuracy(test, predicted):
    assert len(test) == len(predicted)
    tot = 0
    correct = 0
    for i in range(len(test)):
        if test[i] == predicted[i]:
            correct += 1
        tot += 1
    return correct / tot


def select_model(x_train, x_validate, y_train, y_validate, plot_validation_accuracy=False):
    """
    - Trains the decision tree classifier using at least 5 different values of max_depth,
    as well as three different split criteria (information gain, log loss and Gini coefficient),
    - Evaluates the performance of each one on the validation set
    - Prints the resulting accuracies of each model.
    - Additionally, for the hyperparameters that achieve the highest validation accuracy,
    reports the corresponding test accuracy.
    Args:
        x_train:
        x_validate:
        y_train:
        y_validate:

    Returns:

    """
    depths = np.arange(50, 1000, 50)
    criteria = ['gini', 'entropy', 'log_loss']
    hyperparams = [(d, c) for c in criteria for d in depths]
    val_accuracies = np.zeros((len(criteria), len(depths)))

    for i, criterion in enumerate(criteria):
        for j, d in enumerate(depths):
            clf = tree.DecisionTreeClassifier(max_depth=d, criterion=criterion)
            clf = clf.fit(x_train, y_train)

            y_validation_prediction = clf.predict(x_validate)

            val_accuracies[i, j] = measure_accuracy(y_validate, y_validation_prediction)

            print(
                f"Depth {d:3} with {criterion:8} criterion had validation accuracy {measure_accuracy(y_validate, y_validation_prediction):0.5f} ")

    best_ind = np.argmax(val_accuracies)

    if plot_validation_accuracy:
        # scatter validation accuracy against max_depth, drawing lines for each criterion
        for i, criterion in enumerate(criteria):
            plt.scatter(depths, val_accuracies[i, :])
            plt.plot(depths, val_accuracies[i, :], label=criterion)


        plt.title('Validation accuracy vs. max_depth')

        plt.xlabel('max_depth')
        plt.ylabel('validation accuracy')
        plt.legend()
        plt.show()

    return hyperparams[best_ind]

def calc_entropy(prob_array: np.array):
    """
    Given a 1-D probability distribution, compute the entropy.
    Args:
        prob_array: A 1-D numpy array representing a probability distribution.

    Returns: The entropy of the probability distribution.
    """
    assert np.isclose(np.sum(prob_array), 1), "Probabilities must sum to 1."

    p_replaced = np.where(prob_array == 0, 1, prob_array)  # replace 0s with 1s to avoid log(0) errors.
    return -np.sum(prob_array * np.log2(p_replaced))


def calc_expectation(var_vals: np.array, var_probs: np.array):
    return sum(var_vals * var_probs)


def compute_information_gain(x_train: np.array, y_train: np.array, feature_ind: int, threshold: float):
    """
    Compute the information gain for a given feature and threshold.
    Args:
        x_train: The training data, where each row is a datapoint and each column is a feature.
        y_train: The training labels, where each row is a label.
        feature_ind: The index of the feature to compute the information gain for.
        threshold: The threshold to compute the information gain for.

    Returns: The information gain for the given feature and threshold.

    """
    above_t = np.transpose(
        x_train[:, feature_ind] >= threshold)  # contains True if the feature for a datapoint is above the threshold.
    below_t = np.logical_not(above_t)  # contains True if the feature for a datapoint is below the threshold.

    possible_x_conclusions = [below_t, above_t]
    possible_labels = np.unique(y_train)

    n_labels = len(possible_labels)
    label_count_table = np.zeros((2, n_labels))

    for i, conc in enumerate(possible_x_conclusions):
        for j, label in enumerate(possible_labels):
            label_count_table[i, j] = np.sum(np.logical_and(conc, y_train == label))

    probs = label_count_table / len(y_train)

    x_probs: np.array = probs.sum(axis=1)
    y_probs: np.array = probs.sum(axis=0)

    probs_y_given_x = probs / x_probs[:, None]

    entropy_y = calc_entropy(y_probs)

    conditional_entropies = [calc_entropy(probs_y_given_x[i]) for i in range(len(x_probs))]

    expected_conditional_entropy = calc_expectation(conditional_entropies, x_probs)

    inf_gain = entropy_y - expected_conditional_entropy

    return inf_gain


def compute_information_gain_for(X_train: np.array, y_train: np.array, feature: str, threshold,
                                 vectorizer: CountVectorizer, print_gain=False):
    feature_arr = vectorizer.get_feature_names_out()
    feature_ind = np.where(feature_arr == feature)[0][0]
    gain = compute_information_gain(X_train, y_train, feature_ind, threshold)
    if print_gain:
        print(f"Information gain for feature `{feature}` with threshold {threshold} is {gain}")


def export_tree(classifier: sklearn.tree.DecisionTreeClassifier):
    dot_data = tree.export_graphviz(classifier,
                                    out_file=None,
                                    feature_names=vectorizer.get_feature_names_out(),
                                    max_depth=2,
                                    filled=True,
                                    rounded=True)
    graph = graphviz.Source(dot_data)
    graph.render(filename="best_tree", directory="figures", format="png")


if __name__ == '__main__':
    # load the data
    X_train, X_validate, X_test, y_train, y_validate, y_test, vectorizer = load_data()
    depth, criterion = select_model(X_train, X_validate, y_train, y_validate, True)

    # train a model with the best hyperparameters
    clf = tree.DecisionTreeClassifier(max_depth=depth, criterion=criterion)

    clf.fit(X_train, y_train)

    # report its accuracy on the test dataset

    y_test_prediction = clf.predict(X_test)
    acc = measure_accuracy(y_test, y_test_prediction)

    X_train_arr = X_train.toarray()

    print(
        f"\nA model trained on the best hyperparameters (depth={depth}, criterion={criterion}) had test accuracy {acc}")

    # print some information gain values
    compute_information_gain_for(X_train_arr, y_train, "donald", 0.5, vectorizer, print_gain=True)
    compute_information_gain_for(X_train_arr, y_train, "trumps", 0.5, vectorizer, print_gain=True)
    compute_information_gain_for(X_train_arr, y_train, "the", 0.5, vectorizer, print_gain=True)
    compute_information_gain_for(X_train_arr, y_train, "hillary", 0.5, vectorizer, print_gain=True)
    compute_information_gain_for(X_train_arr, y_train, "and", 0.5, vectorizer, print_gain=True)

    export_tree(clf)
