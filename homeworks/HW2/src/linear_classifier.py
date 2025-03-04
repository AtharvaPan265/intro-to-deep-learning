import numpy as np
from download_mnist import load

x_train, y_train, x_test, y_test = load()
x_train, y_train, x_test, y_test = load()
x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(10000, 28*28)

class LinearClassifier:
    def __init__(self, input_dim, num_classes):
        self.W = np.random.randn(input_dim, num_classes) * 0.001
        self.b = np.zeros((num_classes))

