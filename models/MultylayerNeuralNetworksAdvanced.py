import sys
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim

# Add the `data` folder to the system path
current_path = os.getcwd()
data_folder = os.path.join(current_path, "..", "data")
sys.path.append(data_folder)

from prepare_data import preprocess_data_with_validation, preprocess_data_without_validation

# Define the log file path
log_file_path = os.path.join(current_path, "..", "notebooks", "training_results.log")

# Function to append results to the log file
def log_results(message, log_file_path=log_file_path):
    with open(log_file_path, "a") as f:
        f.write("[Advanced Model] " + message + "\n")
    print(message)  # Also print to console

# Define the dataset path
data_dir = os.path.join(data_folder, "Car-Bike-Dataset")

# Choose between preprocessing with or without validation
use_validation = True

if use_validation:
    train_images, val_images, test_images, train_labels, val_labels, test_labels = preprocess_data_with_validation(
        dataset_path=data_dir, validation_size=0.2, test_size=0.5, pca_components=50
    )
else:
    train_images, test_images, train_labels, test_labels = preprocess_data_without_validation(
        dataset_path=data_dir, test_size=0.2, pca_components=50
    )
    val_images, val_labels = None, None

# Convert data to PyTorch tensors
train_images = torch.tensor(train_images, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.long)
test_images = torch.tensor(test_images, dtype=torch.float32)
test_labels = torch.tensor(test_labels, dtype=torch.long)

if use_validation:
    val_images = torch.tensor(val_images, dtype=torch.float32)
    val_labels = torch.tensor(val_labels, dtype=torch.long)

# Define a custom PyTorch Dataset
class BikeCarDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(BikeCarDataset(train_images, train_labels), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(BikeCarDataset(test_images, test_labels), batch_size=batch_size, shuffle=False)

if use_validation:
    val_loader = DataLoader(BikeCarDataset(val_images, val_labels), batch_size=batch_size, shuffle=False)
else:
    val_loader = None

# Define the advanced neural network
class AdvancedNN(nn.Module):
    def __init__(self, input_dim=50):
        super(AdvancedNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # Output layer for 2 classes (Bike, Car)
        )

    def forward(self, x):
        return self.fc(x)

# Initialize the model, loss function, and optimizer
model = AdvancedNN(input_dim=50)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train(model, loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        log_results(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_loss:.4f}")

# Validation loop
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == labels).sum().item()
    avg_loss = total_loss / len(loader)
    accuracy = correct / len(loader.dataset)
    log_results(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

# Train the model
train(model, train_loader, criterion, optimizer, epochs=10)

# Evaluate on validation set (if available)
if use_validation and val_loader is not None:
    validate(model, val_loader, criterion)

# Evaluate on test set
validate(model, test_loader, criterion)

# Save the trained model
model_path = os.path.join(current_path, "..", "notebooks", "advanced_nn.pth")
os.makedirs(os.path.dirname(model_path), exist_ok=True)
torch.save(model.state_dict(), model_path)
log_results(f"Model saved to {model_path}")
