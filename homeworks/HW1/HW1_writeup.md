# Problem 1 (KNN)
## Classify Function Code
```python
def kNNClassify(newInput, dataSet, labels, k):
    result = []
    for test_image in newInput:
        # Calculate distances between test image and all training images
        distances = []￼

        for train_image in dataSet:
            l2_distance = np.sqrt(np.sum((test_image - train_image) ** 2))
            # l1_distance = np.sum(np.abs(test_image - train_image))
            distances.append(l2_distance)

        # Find k nearest neighbors
        distances = np.array(distances)
        k_nearest_indices = np.argsort(distances)[:k]
        k_nearest_labels = labels[k_nearest_indices]

        # Get most common label among k neighbors
        predicted_label = np.bincount(k_nearest_labels).argmax()
        result.append(predicted_label)
    return result
```
## Outputs
### L1 Distance (Manhattan)
```
---classification accuracy for knn on mnist: 0.9529529529529529 ---
---execution time: 149.7359402179718 seconds ---
```

$Training\ Set = 60,000$
$Test\ Set = 1,000$
$K = 3$
$Accuracy = 95.3\%$
#### L2 Distance (Euclidean)
```
---classification accuracy for knn on mnist: 0.9619619619619619 ---
---execution time: 177.7759883403778 seconds ---
```

$Training\ Set = 60,000$
$Test\ Set = 1,000$
$K = 3$
$Accuracy = 96.2\%$
## Summary of Work
### Testing with distance metrics
I tested L1 and L2 distances, and I found L2 to be a little more accurate
### Testing with K
I tested with various K values. according to some online searches through LLMs and  [1](https://www.datacamp.com/tutorial/k-nearest-neighbor-classification-scikit-learn)[2](https://customers.pyimagesearch.com/lesson-sample-k-nearest-neighbor-classification/) seemed like testing 1-13 would be good enough, and I would want to select an odd number so I iterated through the odd numbers to find the most accurate k.
```python
for k in range(1, 13, 2):
    start_time = time.time()
    outputlabels = kNNClassify(x_test, x_train, y_train, k)
    result = y_test - outputlabels
    accuracy = (1 - np.count_nonzero(result) / len(outputlabels))
    print(f"K: {k} with accuracy: {accuracy} ran in: {(time.time() - start_time)}\n")
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k
```
#### Results
```
K: 1 with accuracy: 0.9691 ran in: 140.34267377853394
K: 3 with accuracy: 0.9705 ran in: 139.70263123512268
K: 5 with accuracy: 0.9688 ran in: 143.7878577709198
K: 7 with accuracy: 0.9694 ran in: 139.48520708084106
K: 9 with accuracy: 0.9659 ran in: 140.7079656124115
K: 11 with accuracy: 0.9668 ran in: 138.46952533721924

Best K: 3 with accuracy: 0.9705
```
Based on these results I chose $k=3$
## Alternative Approach (CUDA)
## Code
I used the numba library to parallelize the calculation using CUDA.
### Distance Calculation Function

```python
@cuda.jit()  # this is the device funciton
def calculate_l2_cuda(test_image, train_images, distances, n_train, image_size):
    idx = cuda.grid(1)
    if idx < n_train:
        l2_dist = 0.0
        for i in range(image_size):
            for j in range(image_size):
                diff = test_image[i, j] - train_images[idx, i, j]
                l2_dist += diff * diff
        distances[idx] = math.sqrt(l2_dist)
```
This is the device function, basically it compares the `test_image` with the dataset by comparing the corresponding pixels.
### Classify Function
```python
def kNNClassify(newInput, dataSet, labels, k):
    result = []
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
            d_test_image, d_train_images, d_distances, n_train, image_size
        )

        # Find k nearest neighbors
        distances = np.array(d_distances.copy_to_host())
        k_nearest_indices = np.argsort(distances)[:k]
        k_nearest_labels = labels[k_nearest_indices]

        # Get most common label among k neighbors
        predicted_label = get_most_frequent(k_nearest_labels)
        result.append(predicted_label)
    return result
```
There is a bunch of Cuda steps here
1. `d_train_images` is the device copy of the training dataset which gets copied by the `cuda.to_device(dataSet)` command.
2. `D_distances` is the device copy of the distances matrix initialized as 0s.
3. The threads per block and blocks per grid is compute resource allocation.
4. Then the code iterates through the testing dataset and then calls the `calculate_l2_cuda` device function.
5. finally we need to copy the distances back to the host to sort, and identify the nearest neighbors.
# Problem 2 (Linear Classifier)

