import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin

# from scipy.special import logsumexp

SEED = 318407
np.random.seed(seed=SEED)


class NaiveBayesClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, k: int = 3, a: int = 4) -> None:
        self._estiamtor_type = "classifier"
        self.k = k
        self.a = a
        self.pstwa = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:

        D = len(y)
        N = list(Counter(y).values())

        self.classes_ = list(set(y))
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        P_classes = np.zeros(shape=(n_classes, 1))

        for c in range(n_classes):
            self.classes_[c] = int(self.classes_[c])
            P_classes[c] = N[c] / D

        mu = np.zeros(shape=(n_classes, n_features))
        sigma = np.zeros(shape=(n_classes, n_features))

        """
            Mu
        """
        for j in range(D):
            for i in range(n_features):
                for c in self.classes_:
                    if y[j] == c:
                        mu[c, i] += X[j, i]
        for c in self.classes_:
            mu[c, :] = mu[c, :] / N[c]

        """
            Sigma
        """
        for j in range(D):
            for i in range(n_features):
                for c in self.classes_:
                    if y[j] == c:
                        sigma[c, i] = np.power(X[j, i] - mu[c, i], 2)
        for c in self.classes_:
            sigma[c, :] = sigma[c, :] / (N[c] - 1)
        sigma = np.sqrt(sigma)

        self.k = n_classes
        self.a = n_features
        self.mu = mu
        self.sigma = sigma
        self.theta_ = mu
        self.var_ = np.power(sigma, 2)
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
            PI_pxy = 0
            for i in range(self.a):
                if P[c, i] < 1e-300:
                    PI_pxy += -300
                else:
                    PI_pxy += np.log(P[c, i])
            p[c] = self.P_classes[c] * PI_pxy

        self.pstwa.append(p.T)

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
                # print(i, x)
                predictions[i] = self.single_predict(x)
                i += 1
            predictions = np.reshape(predictions, (-1,))
        return predictions

    # def predict(self, X: np.ndarray):
    #     if len(X.shape) == 1 or X.shape[0] == 1 or X.shape[1] == 1:
    #         joint_log_likelihood = self.single_predict(X)
    #     else:
    #         joint_log_likelihood = []
    #         for i in range(np.size(self.classes_)):
    #             jointi = 0  # np.log(self.class_prior_[i])
    #             n_ij = -0.5 * np.sum(np.log(2.0 * np.pi * self.var_[i, :]))
    #             n_ij -= 0.5 * np.sum(((X - self.theta_[i, :]) ** 2) / (self.var_[i, :]), 1)
    #             joint_log_likelihood.append(jointi + n_ij)
    #         jll = np.array(joint_log_likelihood).T
    #         log_prob_x = logsumexp(jll, axis=1)
    #         return jll - np.atleast_2d(log_prob_x).T
