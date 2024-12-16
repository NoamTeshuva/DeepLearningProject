import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import os

# Define the dataset path
current_path = os.getcwd()  # Current working directory
data_dir = os.path.join(current_path, "..", "data", "Car-Bike-Dataset")

# Paths for bike and car subdirectories
bike_path = os.path.join(data_dir, "Bike")
car_path = os.path.join(data_dir, "Car")

# Function to read and preprocess images and labels
def read_images_and_labels(path, label):
    images = []
    labels = []
    for image_name in os.listdir(path):
        image_path = os.path.join(path, image_name)
        try:
            # Open image, resize to 28x28, and convert to grayscale
            img = Image.open(image_path).convert("L").resize((28, 28))  # Grayscale and resize
            images.append(np.array(img))
            labels.append(label)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
    return np.array(images), np.array(labels)

# Load data from Bike and Car folders
bike_images, bike_labels = read_images_and_labels(bike_path, label=0)
car_images, car_labels = read_images_and_labels(car_path, label=1)

# Combine data
images = np.concatenate((bike_images, car_images), axis=0)
labels = np.concatenate((bike_labels, car_labels), axis=0)

# Normalize image data and flatten (e.g., 28x28 -> 784)
images = images.reshape(images.shape[0], -1).astype('float32') / 255.0  # Normalize to [0, 1]

# Split data into train (70%), validation (15%), and test (15%)
train_images, temp_images, train_labels, temp_labels = train_test_split(images, labels, test_size=0.3, random_state=42)
val_images, test_images, val_labels, test_labels = train_test_split(temp_images, temp_labels, test_size=0.5, random_state=42)

# Convert data to PyTorch tensors
train_images = torch.tensor(train_images, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.long)
val_images = torch.tensor(val_images, dtype=torch.float32)
val_labels = torch.tensor(val_labels, dtype=torch.long)
test_images = torch.tensor(test_images, dtype=torch.float32)
test_labels = torch.tensor(test_labels, dtype=torch.long)

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
val_loader = DataLoader(BikeCarDataset(val_images, val_labels), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(BikeCarDataset(test_images, test_labels), batch_size=batch_size, shuffle=False)

# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2),  # Output layer for 2 classes (Bike, Car)
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.fc(x)

# Initialize the model, loss function, and optimizer
model = SimpleNN()
criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for multi-class classification
optimizer = optim.SGD(model.parameters(), lr=0.001)

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
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(loader):.4f}")

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
    accuracy = correct / len(loader.dataset)
    print(f"Validation Loss: {total_loss / len(loader):.4f}, Accuracy: {accuracy:.4f}")

# Train the model
train(model, train_loader, criterion, optimizer, epochs=10)

# Evaluate the model on the test set
validate(model, test_loader, criterion)