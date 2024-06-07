import numpy as np
from collections import Counter
# from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin


class NaiveBayesClassificator(ClassifierMixin):

    def __init__(self, k: int = 3, a: int = 4) -> None:
        self._estiamtor_type = "classifier"
        self.k = k
        self.a = a

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:

        D = len(y)
        N = list(Counter(y).values())
        Classes = list(set(y))
        for i in range(len(Classes)):
            Classes[i] = int(Classes[i])

        ammount_features = X.shape[1]
        mu = np.zeros(shape=(len(Classes), ammount_features))
        sigma = np.zeros(shape=(len(Classes), ammount_features))

        for j in range(D):
            for i in range(ammount_features):
                mu[:, i] += X[j, i]
        for c in Classes:
            mu[c, :] = mu[c, :] / N[c]

        for j in range(D):
            for i in range(ammount_features):
                for c in Classes:
                    sigma[c, i] = np.power(X[j, i] - mu[c, i], 2)
        for c in Classes:
            sigma[c, :] = sigma[c, :] / (N[c] -1)
        sigma = np.sqrt(sigma)

        self.k = len(Classes)
        self.a = ammount_features
        self.mu = mu
        self.sigma = sigma

    def predict(self, X):

        P = np.zeros(shape=(self.k, self.a))

        for i in range(self.a):
            for c in range(self.k):

                frac = 1 / np.sqrt(2*np.pi*np.power(self.sigma[c, i], 2))
                exp = np.exp(-1 * np.power(X[i] - self.mu[c, i], 2) / (2 * np.power(self.sigma, 2)))
                P[c, i] = frac*exp
