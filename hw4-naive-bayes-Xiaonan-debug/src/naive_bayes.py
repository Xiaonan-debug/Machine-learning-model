import numpy as np
import warnings
from src.sparse_practice import sparse_to_numpy

from src.utils import softmax


class NaiveBayes:
    """
    A Naive Bayes classifier for binary data.
    """

    def __init__(self, smoothing=1):
        """
        Args:
            smoothing: controls the smoothing behavior when computing p(x|y).
                If the word "jackpot" appears `k` times across all documents with
                label y=1, we will instead record `k + self.smoothing`. Then
                `p("jackpot" | y=1) = (k + self.smoothing) / Z`, where Z is a
                normalization constant that accounts for adding smoothing to
                all words.
        """
        self.smoothing = smoothing

    def predict(self, X):
        """
        Return the most probable label for each row x of X.
        You should not need to edit this function.
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X):
        """
        Using self.alpha and self.beta, compute the probability p(y | X[i, :])
            for each row X[i, :] of X.  While you will have used log
            probabilities internally, the returned array should be
            probabilities, not log probabilities.

        See equation (9) in `naive_bayes.pdf` for a convenient way to compute
            this using your self.alpha and self.beta. However, note that
            (9) produces unnormalized log probabilities; you will need to use
            your src.utils.softmax function to transform those into probabilities
            that sum to 1 in each row.

        Args:
            X: a sparse matrix of shape `[n_documents, vocab_size]` on which to predict p(y | x)

        Returns 
            probs: an array of shape `[n_documents, n_labels]` where probs[i, j] contains
                the probability `p(y=j | X[i, :])`. Thus, for a given row of this array,
                np.sum(probs[i, :]) == 1.
        """
        n_docs, vocab_size = X.shape
        n_labels = 2

        assert hasattr(self, "alpha") and hasattr(self, "beta"), "Model not fit!"
        assert vocab_size == self.vocab_size, "Vocab size mismatch"
        alpha = self.alpha
        beta = self.beta
        res = np.zeros((n_docs, n_labels))
        for i in range(n_docs):
            for j in range(n_labels):
                res[i,j] = np.sum(X[i,:] * beta[:,j]) + alpha[j]
        return softmax(res)


    def fit(self, X, y):
        """
        Compute self.alpha and self.beta using the training data.
        You should store log probabilities to avoid underflow.
        This function *should not* use unlabeled data. Wherever y is NaN, that
        label and the corresponding row of X should be ignored.

        See equations (10) and (11) in `naive_bayes.pdf` for the math necessary
            to compute your alpha and beta.

        self.alpha should be set to contain the marginal probability of each class label.

        self.beta should be set to the conditional probability of each word
            given the class label: p(w_j | y_i). This should be an array of shape
            [n_vocab, n_labels].  Remember to use `self.smoothing` to smooth word counts!
            See __init__ for details. If we see M total words across all N documents with
            label y=1, have a vocabulary size of V words, and see the word "jackpot" `k`
            times, then: `p("jackpot" | y=1) = (k + self.smoothing) / (M + self.smoothing *
            V)` Note that `p("jackpot" | y=1) + p("jackpot" | y=0)` will not sum to 1;
            instead, `sum_j p(word_j | y=1)` will sum to 1.

        Args: X, a sparse matrix of word counts; Y, an array of labels
        Returns: None; sets self.alpha and self.beta
        """
        n_docs, vocab_size = X.shape
        self.vocab_size = vocab_size

        total = (y == 0).sum() + (y == 1).sum()
        X_1 = np.zeros((total, sparse_to_numpy(X).shape[1]))
        y_1 = np.zeros((total, 1))

        it = 0
        for i in range(X.shape[0]):
            if not np.isnan(y[i]):
                X_1[it] = sparse_to_numpy(X)[i]
                y_1[it] = y[i]
                it = it + 1
        n_docs, vocab_size = X_1.shape
        self.vocab_size = vocab_size

        alpha = np.zeros(2)
        alpha[0] = np.log((y == 0).sum()/total)
        alpha[1] = np.log((y == 1).sum()/total)
        self.alpha = alpha

        combine = np.zeros((total, 2))
        for i in range(n_docs):
            if y_1[i] == 0:
                combine[i,0] = 1
            else:
                combine[i,1] = 1
        upper = np.dot(np.transpose(X_1), combine)
        lower = np.sum(upper, axis=0).reshape((1,2))
        lower = np.repeat(lower, vocab_size, axis = 0).reshape((vocab_size, 2))
        beta = np.log((upper+self.smoothing) /(lower + vocab_size*self.smoothing))
        self.beta = beta



    def likelihood(self, X, y):
        r"""
        Using fit self.alpha and self.beta, compute the log likelihood of the data.
            You should use logs to avoid underflow.
            This function should not use unlabeled data. Wherever y is NaN,
            that label and the corresponding row of X should be ignored.

         Note: when self.smoothing = 0, some elements of your beta will be -inf.
            If `X_{i, j} = 0` and `\beta_{j, y_i} = -inf`, your code should
            compute `X_{i, j} \beta_{j, y_i} = 0` even though numpy will by
            default compute `0 * -inf` as `nan`.

            This behavior is important to pass both `test_smoothing` and
            `test_tiny_dataset_a` simultaneously.

            The easy way to do this is to leave `X` as a *sparse array*, which
            will solve the problem for you. You can also explicitly define the
            desired behavior, or use `np.nonzero(X)` to only consider nonzero
            elements of X.

        Equation (5) in `naive_bayes.pdf` contains the likelihood, which can be written:

            \sum_{i=1}^N \alpha_{y_i} + \sum_{i=1}^N \sum_{j=1}^V X_{i, j} \beta_{j, y_i}

            You can visualize this formula in http://latex2png.com

        Args: X, a sparse matrix of word counts; Y, an array of labels
        Returns: the log likelihood of the data
        """
        assert hasattr(self, "alpha") and hasattr(self, "beta"), "Model not fit!"

        alpha = self.alpha
        beta = self.beta
        total = (y == 0).sum() + (y == 1).sum()
        X_1 = np.zeros((total, sparse_to_numpy(X).shape[1]))
        y_1 = np.zeros((total, ))

        it = 0
        for i in range(X.shape[0]):
            if not np.isnan(y[i]):
                X_1[it] = sparse_to_numpy(X)[i]
                y_1[it] = y[i]
                it = it + 1
        n_docs, vocab_size = X_1.shape
        self.vocab_size = vocab_size

        likelihood = 0
        index = 0
        for i in range(n_docs):
            if y[i] == 1:
                index = 1
            likelihood += alpha[index]
            for j in range(vocab_size):
                if X_1[i,j] == 0:
                    likelihood += 0
                else:
                    likelihood += X_1[i,j]*beta[j,index]

        return likelihood
