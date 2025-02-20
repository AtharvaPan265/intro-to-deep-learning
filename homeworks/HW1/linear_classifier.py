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

# Evaluate the model
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test).sum().item() / y_test.size(0)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
