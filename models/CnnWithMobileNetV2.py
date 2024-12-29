import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader

# Add the `data` folder to the system path
current_path = os.getcwd()
data_folder = os.path.join(current_path, "..", "data")
sys.path.append(data_folder)

from prepare_data import preprocess_data_with_validation, preprocess_data_without_validation

# Define the log file path
log_file_path = os.path.join(current_path, "..", "notebooks", "mobilenet_training_results.log")

# Function to append results to the log file
def log_results(message, log_file_path=log_file_path):
    with open(log_file_path, "a") as f:
        f.write("[MobileNetV2 Model] " + message + "\n")
    print(message)  # Also print to console

# Define the dataset path
data_dir = os.path.join(data_folder, "Car-Bike-Dataset")

# Choose between preprocessing with or without validation
use_validation = True

if use_validation:
    train_images, val_images, test_images, train_labels, val_labels, test_labels = preprocess_data_with_validation(
        dataset_path=data_dir, validation_size=0.2, test_size=0.5, pca_components=None
    )
else:
    train_images, test_images, train_labels, test_labels = preprocess_data_without_validation(
        dataset_path=data_dir, test_size=0.2, pca_components=None
    )
    val_images, val_labels = None, None

# Convert data to PyTorch tensors and preprocess for MobileNetV2
train_images = torch.tensor(train_images, dtype=torch.float32).unsqueeze(1)
train_labels = torch.tensor(train_labels, dtype=torch.long)
test_images = torch.tensor(test_images, dtype=torch.float32).unsqueeze(1)
test_labels = torch.tensor(test_labels, dtype=torch.long)

if use_validation:
    val_images = torch.tensor(val_images, dtype=torch.float32).unsqueeze(1)
    val_labels = torch.tensor(val_labels, dtype=torch.long)

# Define custom dataset class with transforms
class BikeCarDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Convert grayscale to 3 channels
        image = image.repeat(3, 1, 1)  # [1, H, W] -> [3, H, W]

        if self.transform:
            image = self.transform(image)
        return image, label

# Define transformations for MobileNetV2
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # MobileNetV2 normalization
])

# Create data loaders
batch_size = 256
train_dataset = BikeCarDataset(train_images, train_labels, transform=transform)
test_dataset = BikeCarDataset(test_images, test_labels, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

if use_validation:
    val_dataset = BikeCarDataset(val_images, val_labels, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
else:
    val_loader = None

# Load pretrained MobileNetV2 and modify the final layer
model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.last_channel, 2)  # Replace the final layer for 2 classes
model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train(model, loader, criterion, optimizer, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        log_results(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_loss:.4f}")

# Validation function
def validate(model, loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == labels).sum().item()
    avg_loss = total_loss / len(loader)
    accuracy = correct / len(loader.dataset)
    log_results(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

# Train and validate the model
train(model, train_loader, criterion, optimizer, epochs=10)
if use_validation:
    validate(model, val_loader, criterion)
validate(model, test_loader, criterion)

# Save the trained model
model_path = os.path.join(current_path, "..", "notebooks", "mobilenetv2_model.pth")
os.makedirs(os.path.dirname(model_path), exist_ok=True)
torch.save(model.state_dict(), model_path)
log_results(f"Model saved to {model_path}")
