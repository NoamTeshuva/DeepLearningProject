import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt  # Required for plotting
from torchviz import make_dot
import seaborn as sns
from torchsummary import summary

# Add the `data` folder to the system path
current_path = os.getcwd()
data_folder = os.path.join(current_path, "data")
sys.path.append(data_folder)

from data.prepare_data import preprocess_data_with_validation, preprocess_data_without_validation

# Define the dataset path
data_dir = os.path.join(data_folder, "Car-Bike-Dataset")

# Choose between preprocessing with or without validation
use_validation = True

if use_validation:
    train_images, val_images, test_images, train_labels, val_labels, test_labels = preprocess_data_with_validation(
        dataset_path=data_dir, validation_size=0.2, test_size=0.5
    )
else:
    train_images, test_images, train_labels, test_labels = preprocess_data_without_validation(
        dataset_path=data_dir, test_size=0.2
    )
    val_images, val_labels = None, None

# Convert data to PyTorch tensors
train_images = torch.tensor(train_images, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
train_labels = torch.tensor(train_labels, dtype=torch.long)
test_images = torch.tensor(test_images, dtype=torch.float32).unsqueeze(1)
test_labels = torch.tensor(test_labels, dtype=torch.long)

if use_validation:
    val_images = torch.tensor(val_images, dtype=torch.float32).unsqueeze(1)
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
batch_size = 256
train_loader = DataLoader(BikeCarDataset(train_images, train_labels), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(BikeCarDataset(test_images, test_labels), batch_size=batch_size, shuffle=False)

if use_validation:
    val_loader = DataLoader(BikeCarDataset(val_images, val_labels), batch_size=batch_size, shuffle=False)
else:
    val_loader = None

# Define the CNN model
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),  # Convolutional layer
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Max pooling
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 75 * 50, 256),  # Adjust based on input size
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # Output layer for 2 classes
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Initialize the model, loss function, and optimizer
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Metrics tracking
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

def train(model, loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for images, labels in loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Accuracy calculation
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(loader)
        accuracy = correct / total
        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    print("validation")
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Accuracy calculation
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    val_losses.append(avg_loss)
    val_accuracies.append(accuracy)
    print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

# Train the model
epochs = 10
for epoch in range(epochs):
    train(model, train_loader, criterion, optimizer, epochs=1)
    if use_validation and val_loader is not None:
        validate(model, val_loader, criterion)

# # Evaluate on test set
# validate(model, test_loader, criterion)

# Test loop
test_correct = 0
test_total = 0
with torch.no_grad():
    for test_images, test_labels in test_loader:
        outputs = model(test_images)
        _, test_predicted = torch.max(outputs, 1)
        test_correct += (test_predicted == test_labels).sum().item()
        test_total += test_labels.size(0)

test_accuracy = test_correct / test_total
print(f"Test Accuracy: {test_accuracy:.4f}")

# Plot the metrics
plt.figure(figsize=(12, 6))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
if use_validation and val_losses:
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Over Epochs')
plt.legend()
plt.grid(True)

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy')
if use_validation and val_accuracies:
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Over Epochs')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# Confusion Matrix
cm = confusion_matrix(test_labels.cpu(), test_predicted.cpu())

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Car', 'Bike'], yticklabels=['Car', 'Bike'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Generate the classification report
print("\nClassification Report:")
print(classification_report(test_labels.long(), test_predicted.long()))

# Save the trained model
model_path = os.path.join(current_path, "notebooks", "cnn_model.pth")
os.makedirs(os.path.dirname(model_path), exist_ok=True)
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")
