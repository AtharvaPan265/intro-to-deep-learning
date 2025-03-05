import numpy as np
from download_mnist import load

x_train, y_train, x_test, y_test = load()
x_train, y_train, x_test, y_test = load()
x_train = x_train.reshape(60000, 28 * 28)
x_test = x_test.reshape(10000, 28 * 28)


class LinearClassifier:
    def __init__(self, input_dim=784, output_dim=10, learning_rate=0.01):
        # Initialize weights and bias
        self.W = np.random.randn(input_dim, output_dim) * 0.3  # small random numbers
        self.b = np.zeros(output_dim)
        self.learning_rate = learning_rate

    def forward(self, X):
        # Linear forward pass
        scores = np.dot(X, self.W) + self.b
        return scores


model = LinearClassifier()
scores = model.forward(x_train)
print(scores.shape)
