import pytest
import numpy as np
from hw1_code import compute_information_gain, calc_entropy

def generate_cloudy_raining_example() -> (np.array, np.array):
    x_train = []
    y_train = []

    for i in range(24):
        x_train.append([1])
        y_train.append("real")

    for i in range(1):
        x_train.append([1])
        y_train.append("fake")

    for i in range(25):
        x_train.append([0])
        y_train.append("real")

    for i in range(50):
        x_train.append([0])
        y_train.append("fake")

    return np.array(x_train), np.array(y_train)

def generate_orange_lemon_example() -> (np.array, np.array):
    x_train = []
    y_train = []

    x_train.append([0, 0])
    y_train.append("orange")

    x_train.append([0, 2])
    y_train.append("orange")

    x_train.append([1, 1])
    y_train.append("lemon")

    x_train.append([1, 2])
    y_train.append("orange")

    x_train.append([2, 0])
    y_train.append("lemon")

    x_train.append([2, 1])
    y_train.append("orange")

    x_train.append([2, 2])
    y_train.append("orange")

    return np.array(x_train), np.array(y_train)

def test_calc_entropy_cloudy_raining():
    x_train, y_train = generate_cloudy_raining_example()

    reals = y_train == "real"
    fakes = y_train == "fake"

    y_probs = np.array([fakes.sum(), reals.sum()]) / len(y_train)

    assert calc_entropy(y_probs) == pytest.approx(0.999711441753)


def test_compute_information_gain_cloudy_raining():
    x_train, y_train = generate_cloudy_raining_example()

    assert compute_information_gain(x_train, y_train, 0, 0.5) == pytest.approx(0.250416518941)


def test_compute_information_gain_orange_lemon():
    x_train, y_train = generate_orange_lemon_example()

    gain = compute_information_gain(x_train, y_train, 0, 1.5)
    assert gain == pytest.approx(0.006, abs=0.001)


def test_compute_information_gain_orange_lemon_2():
    x_train, y_train = generate_orange_lemon_example()

    gain = compute_information_gain(x_train, y_train, 0, 0.5)
    print(f"gain: {gain}")
    assert gain == pytest.approx(0.17, abs=0.001)