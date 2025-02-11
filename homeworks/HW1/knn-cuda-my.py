import math
import numpy as np  
from download_mnist import load
import operator  
import time
from numba import cuda
# classify using kNN  
#x_train = np.load('../x_train.npy')
#y_train = np.load('../y_train.npy')
#x_test = np.load('../x_test.npy')
#y_test = np.load('../y_test.npy')
x_train, y_train, x_test, y_test = load()
x_train = x_train.reshape(60000,28,28).astype(np.float32)
x_test  = x_test.reshape(10000,28,28).astype(np.float32)

@cuda.jit() # this is the device funciton
def calculate_l2_cuda(test_image, train_images, distances, n_train, image_size):
    idx = cuda.grid(1)
    if idx < n_train:
        l2_dist = 0.0
        for i in range(image_size):
            for j in range(image_size):
                diff = test_image[i, j] - train_images[idx, i, j]
                l2_dist += diff * diff
        distances[idx] = math.sqrt(l2_dist)

def get_most_frequent(labels):
    values, counts = np.unique(labels, return_counts=True)
    return values[np.argmax(counts)]

# this runs on the host so we need to copy training data to the device and distances to the device
def kNNClassify(newInput, dataSet, labels, k): 
    result=[]    
    n_test = len(newInput)
    n_train = len(dataSet)
    image_size = 28

    d_train_images = cuda.to_device(dataSet)

    distances = np.zeros(n_train, dtype=np.float32)
    d_distances = cuda.to_device(distances)

    threadsperblock = 256
    blockspergrid = (n_train + (threadsperblock - 1)) // threadsperblock

    for test_image in newInput:
        d_test_image = cuda.to_device(test_image)
        calculate_l2_cuda[blockspergrid, threadsperblock](
            d_test_image, d_train_images, d_distances, n_train, image_size)
            
        # Find k nearest neighbors
        distances = np.array(d_distances.copy_to_host())
        k_nearest_indices = np.argsort(distances)[:k]
        k_nearest_labels = labels[k_nearest_indices]
        
        # Get most common label among k neighbors
        predicted_label = get_most_frequent(k_nearest_labels)
        result.append(predicted_label)
    return result

start_time = time.time()
test = 10000

# Use the best K to classify
outputlabels = kNNClassify(x_test[0:test], x_train, y_train, 4)
result = np.subtract(y_test[0:test], outputlabels, dtype=np.float32)
result = (1 - np.count_nonzero(result)/len(outputlabels))
print ("---classification accuracy for knn on mnist: %s ---" %result)
print ("---execution time: %s seconds ---" % (time.time() - start_time))
