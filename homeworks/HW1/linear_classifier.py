# In this problem, you need to review the contents about linear classifier and train a
# simple linear classifier to recognize handwritten digits.

# First, using ”Cross Entropy” as the loss function, and then adopt ”Random Search”
# to find the parameters W. Next, check the recognition accuracy using the testing set.

# Requirements: Write the code to implement the above linear classifier and print
# the recognition accuracy. Since this problem is to help you be familiar with linear 
# classifier, it is not required to achieve a high recognition accuracy.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from download_mnist import load

x_train, y_train, x_test, y_test = load()

# Convert numpy arrays to PyTorch tensors
X_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)
X_test = torch.FloatTensor(x_test)
y_test = torch.LongTensor(y_test)

# Create dataset and dataloader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Define the linear classifier
class LinearClassifier(nn.Module):
    def __init__(self):
        super(LinearClassifier, self).__init__()
        # Define a linear layer. 
        # The input size is 784 (28x28, the flattened size of MNIST images).
        # The output size is 10, representing the 10 possible digits (0-9)
        self.linear = nn.Linear(784, 10)  # 784 input features, 10 output features
        
    # Define the forward pass of the linear classifier
    def forward(self, x):
        # Pass the input x through the linear layer
        return self.linear(x)

# Initialize model, loss function, and optimizer
# Initialize the linear classifier model
model = LinearClassifier()
# Define the loss function as Cross Entropy Loss, which is suitable for multi-class classification
criterion = nn.CrossEntropyLoss()
# Define the optimizer as Stochastic Gradient Descent (SGD) with a learning rate of 0.0001
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training loop
num_epochs = 15
# Iterate over the number of epochs
for epoch in range(num_epochs):
    # Iterate over the batches in the training data
    for batch_X, batch_y in train_loader:
        # Forward pass: compute predicted outputs by passing inputs to the model
        outputs = model(batch_X)
        # Compute the loss: measure how well the predicted outputs match the target outputs
        loss = criterion(outputs, batch_y)

        # Backward pass and optimize
        # Zero the parameter gradients
        optimizer.zero_grad()
        # Compute the gradient of the loss with respect to the model parameters
        loss.backward()
        # Update the model parameters
        optimizer.step()
    
    # Print epoch loss
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
# Evaluate the model
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    _, predicted = torch.max(test_outputs.data, 1)
    accuracy = (predicted == y_test).sum().item() / len(y_test)
    print(f'Test Accuracy: {accuracy:.4f}')
