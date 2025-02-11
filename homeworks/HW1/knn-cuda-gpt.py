import math
import numpy as np
from download_mnist import load
import operator
import time
from numba import cuda

# Constants
BATCH_SIZE = 100
THREADS_PER_BLOCK = 256
SHARED_MEM_SIZE = 16
IMAGE_SIZE = 28

def load_and_preprocess_data():
    x_train, y_train, x_test, y_test = load()
    x_train = x_train.reshape(60000, IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)
    x_test = x_test.reshape(10000, IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)
    return x_train, y_train, x_test, y_test

@cuda.jit
def calculate_L2distance_optimized(test_images, train_images, distances, n_train, n_test, batch_idx):
    # Shared memory for frequently accessed data
    shared_test = cuda.shared.array(shape=(28, 28), dtype=np.float32)
    
    # Get thread index
    idx = cuda.grid(1)
    tx = cuda.threadIdx.x
    
    # Load test image into shared memory
    if tx < IMAGE_SIZE:
        for j in range(IMAGE_SIZE):
            shared_test[tx, j] = test_images[batch_idx, tx, j]
    cuda.syncthreads()
    
    if idx < n_train:
        l2_dist = 0.0
        for i in range(IMAGE_SIZE):
            for j in range(IMAGE_SIZE):
                diff = shared_test[i, j] - train_images[idx, i, j]
                l2_dist += diff * diff
        distances[idx] = l2_dist

def find_k_nearest(distances, labels, k):
    k_nearest_indices = np.argsort(distances)[:k]
    k_nearest_labels = labels[k_nearest_indices]
    return np.bincount(k_nearest_labels).argmax()

def kNNClassify(newInput, dataSet, labels, k):
    try:
        results = np.zeros(len(newInput), dtype=np.int32)
        n_test = len(newInput)
        n_train = len(dataSet)
        
        # Configure CUDA grid
        threadsperblock = THREADS_PER_BLOCK
        blockspergrid = (n_train + (threadsperblock - 1)) // threadsperblock
        
        # Transfer training data to GPU (one-time transfer)
        try:
            d_train_images = cuda.to_device(dataSet)
        except cuda.CudaError as e:
            print(f"CUDA Error during training data transfer: {e}")
            return None
        
        # Allocate memory for distances on GPU
        distances = np.zeros(n_train, dtype=np.float32)
        d_distances = cuda.to_device(distances)
        
        # Process test images in batches
        for batch_start in range(0, n_test, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, n_test)
            current_batch = newInput[batch_start:batch_end]
            
            # Transfer current batch to GPU
            try:
                d_test_batch = cuda.to_device(current_batch)
            except cuda.CudaError as e:
                print(f"CUDA Error during batch transfer: {e}")
                continue
            
            # Process each image in the batch
            for i in range(batch_end - batch_start):
                # Calculate distances
                calculate_L2distance_optimized[blockspergrid, threadsperblock](
                    d_test_batch, d_train_images, d_distances, n_train, n_test, i)
                
                # Copy distances back to CPU
                distances = d_distances.copy_to_host()
                
                # Find k nearest neighbors and get prediction
                predicted_label = find_k_nearest(distances, labels, k)
                results[batch_start + i] = predicted_label
        
        return results
    
    except Exception as e:
        print(f"Error in kNNClassify: {e}")
        return None
    finally:
        # Cleanup GPU memory
        cleanup_gpu_memory()

def cleanup_gpu_memory():
    try:
        cuda.current_context().reset()
    except:
        pass

def evaluate_accuracy(predictions, ground_truth):
    if predictions is None:
        return 0.0
    result = np.subtract(ground_truth, predictions)
    return 1 - np.count_nonzero(result) / len(predictions)

def main():
    # Load and preprocess data
    x_train, y_train, x_test, y_test = load_and_preprocess_data()
    
    start_time = time.time()
    test_size = 10000
    k = 4  # Use the best K to classify
    
    # Perform classification
    outputlabels = kNNClassify(x_test[:test_size], x_train, y_train, k)
    
    # Calculate and print accuracy
    accuracy = evaluate_accuracy(outputlabels, y_test[:test_size])
    print(f"Classification accuracy for kNN on MNIST: {accuracy}")
    print(f"Execution time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
