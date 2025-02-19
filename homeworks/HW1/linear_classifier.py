import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from download_mnist import load

# Load data
x_train, y_train, x_test, y_test = load()
X_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)
X_test = torch.FloatTensor(x_test)
y_test = torch.LongTensor(y_test)

class LinearClassifier(nn.Module):
    def __init__(self):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(784, 10)

    def forward(self, x): 
        return self.linear(x)

model = LinearClassifier()
criterion = nn.CrossEntropyLoss()

def random_search(model, X_train, y_train, num_iterations):
    # Initialize random weights
    W = torch.randn_like(model.linear.weight) * 0.001
    bestloss = float('inf')
    
    for i in range(num_iterations):
        # Random search for update value
        step_size = 0.0001
        Wtry = W + torch.randn_like(W) * step_size
        
        # Update model with trial weights
        with torch.no_grad():
            model.linear.weight.data = Wtry
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            
        # Update weight if loss decreases
        if loss < bestloss:
            W = Wtry
            bestloss = loss
            print(f'iter {i} loss is {bestloss}')
    
    # Set final best weights
    with torch.no_grad():
        model.linear.weight.data = W
    
    return bestloss

# Run random search
num_iterations = 1000
best_loss = random_search(model, X_train, y_train, num_iterations)

# Evaluate the model
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test).sum().item() / y_test.size(0)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
