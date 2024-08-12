import numpy as np
import matplotlib as plt
class Model:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.optimizer = None
        self.losses = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)

    def train(self, x, y, epochs, batch_size):
        self.y_true = y
        for i in range(epochs):
            output = self.forward(x)
            # print(output)
            loss = self.loss.forward(output, y)
            self.losses.append(loss)
            if i % 10 == 0:
                print(f"Epoch {i + 1}/{epochs}, Loss: {loss}")
            grad_output = self.loss.backward(output, y)
            self.backward(grad_output)
            for layer in self.layers:
                if hasattr(layer, 'update'):
                    layer.update(self.optimizer)

    def compile(self, loss, optimizer):
        self.loss = loss
        self.optimizer = optimizer

    def one_hot(self, Y):
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        return one_hot_Y

    def plot_loss(self):
        plt.plot(self.losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

    def predict(self, x_test):
        return self.forward(x_test)

    def evaluate(self, x_test, y_test):
        y_pred = self.predict(x_test)
        test_loss = CrossEntropyLoss().forward(y_pred, y_test)
        predicted_labels = np.argmax(y_pred, axis=1)
        y_test = np.argmax(y_test, axis=1)
        correct_predictions = np.sum(predicted_labels == y_test)
        test_accuracy = 100 * (correct_predictions / len(y_test))
        return test_loss, test_accuracy

    def save(self, filename='model_weights.npz'):
        weights = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'weight'):
                weights[f'W{i}'] = layer.weight
                weights[f'b{i}'] = layer.bias
        np.savez(filename, **weights)

    def load(self, filename='model_weights.npz'):
        weights = np.load(filename)
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'weight'):
                layer.weight = weights[f'W{i}']
                layer.bias = weights[f'b{i}']


class Linear:
    def __init__(self, input_dim, output_dim):
        self.weight = np.random.randn(input_dim, output_dim) * 0.01
        self.bias = np.zeros((1, output_dim))
        self.input = None

    def forward(self, x):
        self.input = x
        return np.dot(x, self.weight) + self.bias

    def backward(self, grad_out):
        weight_gradient = np.dot(self.input.T, grad_out)
        bias_gradient = np.sum(grad_out, axis=0, keepdims=True)
        self.weight_gradient = weight_gradient
        self.bias_gradient = bias_gradient
        grad_in = np.dot(grad_out, self.weight.T)
        return grad_in

    def update(self, optimizer):
        optimizer.step(self)


class ReLU:
    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    def backward(self, grad):
        return grad * np.where(self.input > 0, 1, 0)


class Sigmoid:
    def forward(self, x):
        self.input = x
        self.sigmoid = 1 / (1 + np.exp(-x))
        return self.sigmoid

    def backward(self, grad):
        return grad * (self.sigmoid * (1 - self.sigmoid))


class Tanh:
    def forward(self, x):
        self.input = x
        return ((np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)))

    def backward(self, grad):
        return grad * (1 - (np.tanh(self.input) ** 2))


class Softmax:
    def forward(self, x):
        shifted_x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted_x)
        prob = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return prob

    def backward(self, grad):
        # Softmax backward pass is handled by the loss function's backward pass
        return grad


class CrossEntropyLoss:
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        confidences = np.sum(y_pred_clipped * y_true, axis=1)
        loss = -np.mean(np.log(confidences))
        return loss

    def backward(self, y_pred, y_true):  # Backward loss of softmax function
        samples = len(y_pred)
        return (y_pred - y_true) / len(
            y_pred)  # Simplified expression of backward loss of softmax since it is used with  Cross Entropy Loss


class MSE:
    def forward(self, y_pred, y_true):
        error = y_true - y_pred
        squared_error = np.square(error)
        mse = np.mean(squared_error)
        return mse

    def backward(self, y_pred, y_true):
        return (y_pred - y_true) / len(y_pred)


class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def step(self, layer):
        layer.weight -= self.learning_rate * layer.weight_gradient
        layer.bias -= self.learning_rate * layer.bias_gradient


