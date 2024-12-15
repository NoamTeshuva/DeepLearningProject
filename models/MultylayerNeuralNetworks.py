import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Directory setup
current_path = os.getcwd()  # Current working directory
data_dir = os.path.join(current_path, "..", "data", "Car-Bike-Dataset")  # Path to dataset
train_dir = os.path.join(data_dir, "train")  # Training data
test_dir = os.path.join(data_dir, "test")  # Test data

# Hyperparameters
batch_size = 32
learning_rate = 0.001
num_epochs = 10

# Define transformations for the dataset
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to 128x128
    transforms.ToTensor(),  # Convert images to tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values to [-1, 1]
])

# Load dataset
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(32 * 32 * 32, 128),  # Adjust input size based on final pooled dimensions
            nn.ReLU(),
            nn.Linear(128, 1),  # Binary classification
            nn.Sigmoid()  # Output probability
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)  # Flatten the output of the conv layer
        x = self.fc_layer(x)
        return x


# Initialize model, loss function, and optimizer
model = SimpleNN()
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Training loop
def train(model, loader, criterion, optimizer):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for images, labels in loader:
            labels = labels.float().unsqueeze(1)  # Reshape labels for binary classification

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(loader):.4f}")


# Testing loop
def test(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            labels = labels.float().unsqueeze(1)
            outputs = model(images)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {100 * correct / total:.2f}%")


# Run training and testing
train(model, train_loader, criterion, optimizer)
test(model, test_loader)
