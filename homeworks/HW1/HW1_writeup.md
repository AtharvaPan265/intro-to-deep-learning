# Problem 1 (KNN)
## Code
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
In the code I use the Euclidean distance that is calculated using $$L2\_Distance = \sqrt{((x_{test}-x_{train})+(y_{test}-y_{train}))^2}$$
which is done for the whole flattened image, repeatedly through the whole training dataset per test image using 
```python
l2_distance = np.sqrt(np.sum((test_image - train_image) ** 2))

```
Then it sorts the array of distances, and picks the $k$ nearest ones.

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
### Code
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

## Code
### Linear Classifier
```python
class LinearClassifier(nn.Module):
    def __init__(self):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(784, 10)

    def forward(self, x): 
        return self.linear(x)
```
This code defines the Linear Classifier as a NN Module. with input dimensions of $784$ which is derived form the $28*28$ image dimensions. and the output dimensions is $10$ as in the number of classifications.
### Cross Entropy
```python
criterion = nn.CrossEntropyLoss()
```
This sets the stopping conditions, for which I use a patience of 20. Which defines how many iterations it runs without a loss reduction.
### Random Search
```python
def random_search(model, X_train, y_train, num_iterations):
    W = torch.randn_like(model.linear.weight) * 0.001
    b = torch.zeros_like(model.linear.bias)
    bestloss = float("inf")
    patience = 20
    no_improve = 0

    # Create DataLoader for batch processing
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    for i in range(num_iterations):
        step_size = 0.0001

		# Randomly iterate weights and biases
        Wtry = W + torch.randn_like(W) * step_size
        btry = b + torch.randn_like(b) * step_size

        # Evaluate on batches
        total_loss = 0
        with torch.no_grad():
            model.linear.weight.data = Wtry
            model.linear.bias.data = btry

            for X_batch, y_batch in loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                total_loss += loss.item()

        # Update if better
        if total_loss < bestloss:
            W = Wtry
            b = btry
            bestloss = total_loss
            no_improve = 0
            print(f"iter {i} loss is {bestloss}")
        else:
            no_improve += 1

        # Early stopping
        if no_improve >= patience:
            print(f"Early stopping at iteration {i}")
            break

    with torch.no_grad():
        model.linear.weight.data = W
        model.linear.bias.data = b

    return bestloss

# Run random search
num_iterations = 100
best_loss = random_search(model, X_train, y_train, num_iterations)
```

## Output
```
iter 0 loss is 758.7401039600372
iter 2 loss is 752.5633029937744
iter 3 loss is 749.0556330680847
iter 5 loss is 745.2107131481171
iter 7 loss is 744.4927332401276
iter 12 loss is 740.841587305069
iter 13 loss is 740.5890018939972
iter 15 loss is 738.0674273967743
iter 16 loss is 736.3004505634308
iter 20 loss is 731.996808052063
iter 21 loss is 724.7164709568024
iter 22 loss is 713.7970941066742
iter 30 loss is 712.0403230190277
iter 34 loss is 711.0967862606049
iter 43 loss is 709.8625280857086
iter 44 loss is 704.0635945796967
iter 45 loss is 702.3413310050964
iter 46 loss is 696.8450560569763
iter 54 loss is 690.1411738395691
iter 55 loss is 687.0142946243286
iter 56 loss is 683.4893696308136
iter 60 loss is 680.3644208908081
iter 62 loss is 678.1397061347961
iter 70 loss is 675.7076778411865
iter 71 loss is 672.6531641483307
iter 72 loss is 663.1077909469604
iter 73 loss is 658.0528738498688
iter 76 loss is 656.096437215805
iter 77 loss is 651.9416856765747
iter 78 loss is 649.1053586006165
iter 85 loss is 647.2926115989685
iter 86 loss is 644.1083936691284
iter 87 loss is 643.1683225631714
iter 89 loss is 637.3858823776245
iter 91 loss is 637.0902035236359
iter 96 loss is 632.770087480545
iter 97 loss is 628.772213935852
Test Accuracy: 26.25%
```
## Summary of Work