import numpy as np
from numpy import linalg as LA


def euclidean_distances(X, Y):
    """Compute pairwise Euclidean distance between the rows of two matrices X (shape MxK)
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Euclidean distance between two rows.

    (Hint: You're free to implement this with numpy.linalg.norm)

    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Euclidean distances between rows of X and rows of Y.
    """
    a = np.zeros(shape=(np.shape(X)[0], np.shape(Y)[0]))
    for i in range (np.shape(X)[0]):
        for j in range (np.shape(Y)[0]):
            a[i][j] = LA.norm(X[i] - Y[j])
    return a
    raise NotImplementedError


def manhattan_distances(X, Y):
    """Compute pairwise Manhattan distance between the rows of two matrices X (shape MxK)
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Manhattan distance between two rows.

    (Hint: You're free to implement this with numpy.linalg.norm)

    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Manhattan distances between rows of X and rows of Y.
    """
    a = np.zeros(shape=(np.shape(X)[0], np.shape(Y)[0]))
    for i in range (np.shape(X)[0]):
        for j in range (np.shape(Y)[0]):
            new = X[i] - Y[j]
            sum = 0
            for e in new:
                sum += abs(e)
            a[i][j] = sum
    return a

    raise NotImplementedError


def cosine_distances(X, Y):
    """Compute pairwise Cosine distance between the rows of two matrices X (shape MxK)
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Cosine distance between two rows.

    (Hint: You're free to implement this with numpy.linalg.norm)

    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Cosine distances between rows of X and rows of Y.
    """
    a = np.zeros(shape=(np.shape(X)[0], np.shape(Y)[0]))
    for i in range (np.shape(X)[0]):
        for j in range (np.shape(Y)[0]):
            new = np.multiply(X[i], Y[j])
            upper = 0
            for e in new:
                upper = upper + e
            down = LA.norm(X[i]) * LA.norm(Y[j])
            S = upper/(down+1e-8)
            a[i][j] = 1-S
    return a
    raise NotImplementedError
