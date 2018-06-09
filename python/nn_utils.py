import numpy as np


def shuffle(xsets, ysets, seed=None):
    """Shuffle two datasets harmonically
    Args:
        x, y: datasets, both of them should have same length
    Return:
        (shuffled_x, shuffled_y): tuple including shuffled x and y
    """
    if len(xsets) != len(ysets):
        raise ValueError
    np.random.seed(seed=seed)
    shuffled_indexes = np.random.permutation(len(xsets))
    shuffled_x = xsets[shuffled_indexes]
    shuffled_y = ysets[shuffled_indexes]
    return (shuffled_x, shuffled_y)


def evaluate_nn(X, Y, model):
    ypred = model.forward(X)
    num_true = (Y.argmax(axis=1) == ypred.argmax(axis=1)).sum()
    num_samples = Y.shape[0]
    accuracy = num_true / float(num_samples)
    return accuracy
