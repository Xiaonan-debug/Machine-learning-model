import numpy as np 
np.random.seed(0)


def generate_regression_data(degree, N, amount_of_noise=1.0):
    """
    Generates data to test one-dimensional regression models. This function:

    1)  Generates explanatory variable x: an array of shape (N, 1) that contains 
        floats chosen at uniformly at random between -1 and 1.

    2)  Creates a Polynomial function f() of degree 'degree'. The function's 
        float coefficients are chosen uniformally at random between -10 and 10.

    3)  Generates response variable y: a shape (N, 1) array that contains f(x), 
        where the ith element of y is calculated by applying f() to the ith 
        element of x

    4)  Adds Gaussian noise to y, with mean 0 and standard deviation equal to
        `np.std(y) * amount_of_noise`. That is, the standard deviation of
        the noise should be proportional to the standard deviation of the y
        variable you calculated, without any noise.
        Hint: use np.random.normal to generate this noise


    Do not import or use these packages: scipy, sklearn, sys, importlib.
    Do not use these numpy or internal functions: polynomial, polyfit, polyval, getattr, globals

    Args:
        degree (int): degree of Polynomial that relates the output x and y
        N (int): number of points to generate
        amount_of_noise (float): amount of random noise to add to the relationship 
            between x and y.
    Returns:
        x (np.ndarray): explanatory variable of size N, ranges between -1 and 1.
        y (np.ndarray): response variable of size N, which responds to x as a
                        Polynomial of degree 'degree'.

    """
    x = np.random.uniform(-1, 1, (N,))
    coef = np.random.uniform(-10, 10, (degree + 1,))

    y = np.zeros((N))
    for i in range(coef.size):
        y += coef[i] * x**i

    noise = np.random.normal(0.0, np.std(y)*amount_of_noise, y.shape)
    y += noise
    x_new = x.reshape(-1,1)
    y_new = y.reshape(-1,1)

    return x_new, y_new
