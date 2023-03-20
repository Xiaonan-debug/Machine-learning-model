import numpy as np


class PolynomialRegression():
    def __init__(self, degree):
        """
        Implement PolynomialRegression from scratch.
        
        The `degree` argument controls the complexity of the function.  For
        example, degree = 2 would specify a hypothesis space of all functions
        of the form:

            f(x) = ax^2 + bx + c

        You should implement the closed form solution of least squares:
            w = (X^T X)^{-1} X^T y
        
        Do not import or use these packages: scipy, sklearn, sys, importlib.
        Do not use these numpy or internal functions: polynomial, polyfit, polyval, getattr, globals

        Args:
            degree (int): Degree used to fit the data.
        """
        self.degree = degree
        self.w = np.zeros(self.degree + 1)

    
    def fit(self, features, targets):
        """
        Fit to the given data.

        Hints:
          - Remember to use `self.degree`
          - Remember to include an intercept (a column of all 1s) before you
            compute the least squares solution.

        Args:
            features (np.ndarray): an array of shape [N, 1] containing real-valued inputs.
            targets (np.ndarray): an array of shape [N, 1] containing real-valued targets.
        Returns:
            None (saves model internally)
        """
        sub_matrix = np.ones([features.shape[0], 1])
        for i in range(1, self.degree + 1):
            sub_matrix = np.append(sub_matrix, features**i, axis=1)

        left = np.linalg.inv(np.dot(sub_matrix.T, sub_matrix))
        right = np.dot(sub_matrix.T, targets)
        self.w = np.dot(left, right)
        

    def predict(self, features):
        """
        Given features, use the trained model to predict target estimates. Call
        this after calling fit.

        Args:
            features (np.ndarray): array of shape [N, 1] containing real-valued inputs.
        Returns:
            predictions (np.ndarray): array of shape [N, 1] containing real-valued predictions
        """
        sub_matrix = np.ones([features.shape[0], 1])
        for i in range(1, self.degree + 1):
            sub_matrix = np.append(sub_matrix, features**i, axis=1)

        y_predict = np.dot(sub_matrix, self.w)
        return y_predict
