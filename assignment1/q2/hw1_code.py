import graphviz as graphviz
import numpy as np
from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

TRAIN_PROPORTION = 0.7
TEST_TO_VALIDATE_RATIO = 0.1


def load_data():
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


def select_model(x_train, x_validate, X_test, y_train, y_validate, y_test, plot_results=False):
    depths = np.arange(50, 300, 50)
    criteria = ['gini', 'entropy', 'log_loss']
    hyperparams = [(d, c) for d in depths for c in criteria]
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

    return hyperparams[best_ind]

    # fig = plt.figure()
    # for criterion in criteria:
    #     y_acc = []
    #     plt.scatter()


def compute_information_gain(X_train, y_train, feature, threshold):

    # joint_pdf = np.array([
    #     X_train[]
    # ])
    #
    # entropy =

    pass


if __name__ == "__main__":
    X_train, X_validate, X_test, y_train, y_validate, y_test, vectorizer = load_data()
    depth, criterion = select_model(X_train, X_validate, X_test, y_train, y_validate, y_test)

    # train a model with the best hyperparameters
    clf = tree.DecisionTreeClassifier(max_depth=depth, criterion=criterion)

    clf.fit(X_train, y_train)

    # report its accuracy on the test dataset

    y_test_prediction = clf.predict(X_test)
    acc = measure_accuracy(y_test, y_test_prediction)

    print(f"\nA model trained on the best hyperparameters (depth={depth}, criterion={criterion}) had test accuracy {acc}")




    dot_data = tree.export_graphviz(clf,
                                    out_file=None,
                                    feature_names=vectorizer.get_feature_names_out(),
                                    max_depth=2,
                                    filled=True,
                                    rounded=True)
    graph = graphviz.Source(dot_data)
    graph.render(filename="iris", directory="figures", format="png")
