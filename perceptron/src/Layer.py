import numpy as np
from scipy.special import expit


verbose = False


class ActivationFunction:
    def __init__(self) -> None:
        pass

    def each_forward(self, x):
        pass

    def forward(self, x):
        pass

    def backward(self, x):
        pass

    def derivative(self, x):
        pass


class ReLU(ActivationFunction):
    def __init__(self) -> None:
        super().__init__()

    def each_forward(self, x: np.ndarray):
        for i in range(x.shape[0]):
            x[i] = max(x[i], 0.0)
        return x

    def forward(self, x: np.ndarray):
        for i in range(x.shape[1]):
            column = x[:, i]
            x[:, i] = self.each_forward(column)
        return x

    def derivative(self, x: np.ndarray):
        d = np.zeros(shape=x.shape, dtype=float)
        for c in range(x.shape[1]):
            for r in range(x.shape[0]):
                if x[r][c] >= 0.0:
                    d[r][c] = 1.0
        return d


class Sigmoid(ActivationFunction):
    def __init__(self) -> None:
        super().__init__()

    def each_forward(self, x: np.ndarray):
        return expit(x)

    def forward(self, x: np.ndarray):
        for i in range(x.shape[1]):
            column = x[:, i]
            x[:, i] = self.each_forward(column)
        return x

    def derivative(self, x: np.ndarray):
        d = np.zeros(shape=x.shape, dtype=float)
        for i in range(x.shape[1]):
            column = x[:, i]
            expit = self.each_forward(column)
            d[:, i] = expit * (1 - expit)
        return d


class Layer:
    def __init__(
        self,
        lr: float,
        inputSize: int,
        outputSize: int,
        actFunction: ActivationFunction,
        batch_size: int = 1,
        labels: np.ndarray = None,
    ) -> None:
        self.learningRate_ = lr
        self.inputSize = inputSize
        self.labels = labels
        self.outputSize = outputSize
        self.function = actFunction
        self.batch_size = batch_size
        self.weights = np.random.randn(self.outputSize, self.inputSize + 1)

    def learning_rate(self) -> float:
        return self.learningRate_

    def set_learning_rate(self, lr: float):
        self.learningRate_ = lr

    def extend_input_by_1(self, X):
        if X.shape[0] == self.inputSize:
            current_batch_size = X.shape[1]
        else:
            current_batch_size = X.shape[0]
        input = np.append(X, np.ones((1, current_batch_size)), axis=0)
        return input

    def forward(self, X: np.ndarray):

        if X.shape[0] == self.inputSize:
            current_batch_size = X.shape[1]
        else:
            current_batch_size = X.shape[0]

        # stworzenie wektora pionowego
        X = X.reshape(self.inputSize, current_batch_size)
        self.input = X

        # rozszerzenie wektora o 1 (do biasu)
        X = np.append(X, np.ones((1, current_batch_size)), axis=0)
        W = self.weights

        net = np.dot(W, X)
        net = np.clip(net, -1e2, 1e2)
        self.net = net
        self.output = self.function.forward(net)

        return self.output

    def sgd(self, weights_derivative: np.ndarray):
        beta = self.learning_rate()
        self.weights = self.weights - beta * weights_derivative

    def backward(self, next_layer_derivative: np.ndarray,):

        net_values = self.net

        activation_gradient = self.function.derivative(net_values)
        output_gradient = next_layer_derivative * activation_gradient

        weights_gradient = np.dot(output_gradient, self.input.T)
        bias = np.mean(output_gradient, axis=1, keepdims=True)
        # bias = np.ones_like(output_gradient)  # * self.weights[:, -1:]
        weights_derivative = np.append(weights_gradient, bias, axis=1)

        input_gradient = np.dot(self.weights.T, output_gradient)

        if verbose:
            print("- - - - -")
            print(f"next_layer_deriv shape: {next_layer_derivative.shape}")
            print(f"activation_gradient shape: {activation_gradient.shape}")
            print(f"output_gradient shape: {output_gradient.shape}")
            print("")
            print(f"self.input shape: {self.input.shape}")
            print(f"weights_gradient shape: {weights_gradient.shape}")
            print(f"self.weights shape: {self.weights.shape}")
            print("")
            print(f"input_gradient shape: {input_gradient.shape}")
            print("")
            print(f"weights_derivative shape: {weights_derivative.shape}")
            print("\n")

        self.sgd(weights_derivative)

        return input_gradient
