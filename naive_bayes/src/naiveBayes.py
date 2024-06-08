import numpy as np
from collections import Counter
# from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin


SEED = 318407
np.random.seed(seed=SEED)


class NaiveBayesClassificator(ClassifierMixin):

    def __init__(self, k: int = 3, a: int = 4) -> None:
        self._estiamtor_type = "classifier"
        self.k = k
        self.a = a

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:

        D = len(y)
        N = list(Counter(y).values())
        Classes = list(set(y))
        P_classes = np.zeros(shape=(len(Classes), 1))

        for c in range(len(Classes)):
            Classes[c] = int(Classes[c])
            P_classes[c] = N[c] / D

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
            sigma[c, :] = sigma[c, :] / (N[c] - 1)
        sigma = np.sqrt(sigma)

        self.k = len(Classes)
        self.a = ammount_features
        self.mu = mu
        self.sigma = sigma
        self.P_classes = P_classes

    def single_predict(self, X: np.ndarray):

        P = np.zeros(shape=(self.k, self.a))

        for c in range(self.k):
            for i in range(self.a):

                variance = np.power(self.sigma[c, i], 2)
                fraction = 1 / np.sqrt(2 * np.pi * variance)

                nominator = np.power(X[i] - self.mu[c, i], 2)
                exp = np.exp(-1/2 * nominator/variance)

                P[c, i] = fraction*exp

        p = np.zeros(shape=(self.k, 1))
        for c in range(self.k):
            PI_pxy = 1
            for i in range(self.a):
                PI_pxy = PI_pxy*P[c, i]
            p[c] = self.P_classes[c] * PI_pxy

        # print(f"P:\n{P}\n")
        # print(f"p:\n{p}\n")

        return p.argmax()  # np.argmax(p)

    def predict(self, X: np.ndarray):
        if len(X.shape) == 1 or X.shape[0] == 1 or X.shape[1] == 1:
            predictions = self.single_predict(X)
        else:
            predictions = np.zeros(shape=(X.shape[0], 1))
            i = 0
            for x in X:
                print(i, x)
                predictions[i] = self.single_predict(x)
                i += 1
            predictions = np.reshape(predictions, (-1,))
        return predictions
