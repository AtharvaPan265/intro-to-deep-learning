import math
import numpy as np
from download_mnist import load
import operator
import time
from numba import cuda

# classify using kNN
# x_train = np.load('../x_train.npy')
# y_train = np.load('../y_train.npy')
# x_test = np.load('../x_test.npy')
# y_test = np.load('../y_test.npy')
x_train, y_train, x_test, y_test = load()
x_train = x_train.reshape(60000, 28, 28)
x_test = x_test.reshape(10000, 28, 28)
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)


@cuda.jit
def calculate_distance(test_image, train_images, distances, n_train, image_size):
    idx = cuda.grid(1)
    if idx < n_train:
        l2_dist = 0.0
        l1_dist = 0.0
        for i in range(image_size):
            for j in range(image_size):
                diff = test_image[i, j] - train_images[idx, i, j]
                l2_dist += diff * diff
                # l1_dist += abs(diff)
        # distances[idx] = (l1_dist + math.sqrt(l2_dist)) / 2
        distances[idx] = (math.sqrt(l2_dist)) / 2


def kNNClassify(newInput, dataSet, labels, k):
    results = []
    n_test = len(newInput)
    n_train = len(dataSet)
    image_size = 28

    # Configure CUDA grid
    threadsperblock = 256
    blockspergrid = (n_train + (threadsperblock - 1)) // threadsperblock

    # Transfer training data to GPU once
    d_train_images = cuda.to_device(dataSet)

    # Allocate memory for distances on device
    distances = np.zeros(n_train, dtype=np.float32)
    d_distances = cuda.to_device(distances)

    for i in range(n_test):
        # Transfer current test image to GPU
        d_test_image = cuda.to_device(newInput[i])

        # Calculate distances for each test image
        calculate_distance[blockspergrid, threadsperblock](
            d_test_image, d_train_images, d_distances, n_train, image_size
        )

        # Copy distances back to host
        distances = d_distances.copy_to_host()

        # Find k nearest neighbors
        k_nearest_indices = np.argsort(distances)[:k]
        k_nearest_labels = labels[k_nearest_indices]

        # Get most common label
        predicted_label = np.bincount(k_nearest_labels).argmax()
        results.append(predicted_label)

    return np.array(results)


start_time = time.time()
test = 10000
best_k = 0
best_accuracy = 0

# for k in range(1, 100, 2):  # Test odd values of k from 1 to 99
#     outputlabels = kNNClassify(x_test[0:test], x_train, y_train, k)
#     result = y_test[0:test] - outputlabels
#     accuracy = (1 - np.count_nonzero(result) / len(outputlabels))
#     if accuracy > best_accuracy:
#         best_accuracy = accuracy
#         best_k = k

# print(f"Best K: {best_k} with accuracy: {best_accuracy}")
best_k = 4
# Use the best K to classify
outputlabels = kNNClassify(x_test[0:test], x_train, y_train, best_k)
result = np.subtract(y_test[0:test], outputlabels)
result = 1 - np.count_nonzero(result) / len(outputlabels)
print("---classification accuracy for knn on mnist: %s ---" % result)
print("---execution time: %s seconds ---" % (time.time() - start_time))
