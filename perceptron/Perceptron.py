import numpy as np

from Layer import Layer


class MLP:
    def __init__(self, layers: list[Layer]) -> None:
        self.layers = layers
        self.batch_size = layers[0].batch_size

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward propagation of x through all layers"""
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def __mse(self, f, y):
        diff = f - y
        batch = diff.shape[1]
        mse = np.zeros(shape=diff.shape)
        for i in range(batch):
            squared = 0.5 * (diff[:, i] ** 2) / diff.shape[0]
            mse[:, i] = squared.reshape(mse[:, i].shape)
        return mse

    def __mse_summed(self, f, y):
        diff = f - y
        batch = diff.shape[1]

        mse = self.__mse(f, y)
        mse_sum = np.zeros(shape=(batch, 1))

        for i in range(batch):
            mse_sum[i] = np.sum(mse[:, i])

        return mse_sum

    def __mse_derivative(self, f, y):
        return f - y

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int,
        learning_rate: float,
        verbose: int = 0,
    ):
        all_current_loss = []
        all_summed_mse = []
        epoch_loss = np.zeros((y_train.shape[0], y_train.shape[1]))

        for layer in self.layers:
            layer.set_learning_rate(learning_rate)

        for _ in range(epochs):
            shuffle = np.random.permutation(x_train.shape[0])

            X_batches = np.array_split(
                x_train[shuffle], x_train.shape[0] / self.batch_size
            )
            Y_batches = np.array_split(
                y_train[shuffle], y_train.shape[0] / self.batch_size
            )

            i = 0
            cycles = 0
            current_E_total = 0

            for batch_x, batch_y in zip(X_batches, Y_batches):

                currnet_batch_size = batch_y.shape[0]

                output = self(batch_x)
                target = batch_y.T

                current_loss = self.__mse(output, target)
                loss_derivative = self.__mse_derivative(output, target)
                current_E_total += np.mean(self.__mse_summed(output, target))

                cycles += 1
                epoch_loss[i:i+currnet_batch_size, :] = current_loss.T
                i += currnet_batch_size

                ones = np.ones((1, currnet_batch_size))
                next_layer_derivative = np.append(
                    loss_derivative, ones, axis=0
                    )
                for layer in reversed(self.layers):
                    layer_net_deriv = layer.backward(
                        next_layer_derivative[:-1, :]
                        )
                    next_layer_derivative = layer_net_deriv

            all_current_loss.append(np.mean(epoch_loss))
            all_summed_mse.append(current_E_total/cycles)

            if verbose:
                print(f"epoch: {_+1} | error: {current_E_total/cycles}")
                # print(np.mean(epoch_loss))

        self.summed_mse_errors = all_summed_mse
        return all_current_loss, all_summed_mse

    def predict(self, input_data):
        shape = input_data.T.shape
        if len(input_data.T.shape) == 1:
            input_data = input_data.reshape((shape[0], 1))
        else:
            input_data = input_data.reshape((shape[0], shape[1]))
        return self(input_data)
