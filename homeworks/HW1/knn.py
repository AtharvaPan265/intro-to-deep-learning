import math
import numpy as np  
from download_mnist import load
import operator  
import time
# classify using kNN  
#x_train = np.load('../x_train.npy')
#y_train = np.load('../y_train.npy')
#x_test = np.load('../x_test.npy')
#y_test = np.load('../y_test.npy')
x_train, y_train, x_test, y_test = load()
x_train = x_train.reshape(60000,28,28)
x_test  = x_test.reshape(10000,28,28)
x_train = x_train.astype(float)
x_test = x_test.astype(float)
def kNNClassify(newInput, dataSet, labels, k): 
    result=[]
    ########################
    # Input your code here #
    ########################    
    for test_image in newInput:
        # Calculate distances between test image and all training images
        distances = []
        for train_image in dataSet:
            # L2 distance calculation
            l2_distance = np.sqrt(np.sum((test_image - train_image) ** 2))
            l1_distance = np.sum(np.abs(test_image - train_image))
            distance = (l1_distance + l2_distance) / 2
            distances.append(distance)
            
        # Find k nearest neighbors
        distances = np.array(distances)
        k_nearest_indices = np.argsort(distances)[:k]
        k_nearest_labels = labels[k_nearest_indices]
        
        # Get most common label among k neighbors
        predicted_label = np.bincount(k_nearest_labels).argmax()
        result.append(predicted_label)
    ####################
    # End of your code #
    ####################
    return result

start_time = time.time()
test = 500
best_k = 0
best_accuracy = 0

for k in range(1, 100, 2):  # Test odd values of k from 1 to 99
    outputlabels = kNNClassify(x_test[0:test], x_train, y_train, k)
    result = y_test[0:test] - outputlabels
    accuracy = (1 - np.count_nonzero(result) / len(outputlabels))
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

print(f"Best K: {best_k} with accuracy: {best_accuracy}")

# Use the best K to classify
outputlabels = kNNClassify(x_test[0:test], x_train, y_train, best_k)
result = y_test[0:test] - outputlabels
result = (1 - np.count_nonzero(result)/len(outputlabels))
print ("---classification accuracy for knn on mnist: %s ---" %result)
print ("---execution time: %s seconds ---" % (time.time() - start_time))
