import sys
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import sys
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# Dynamically add the `data` folder to the system path
current_path = os.getcwd()  # Current working directory
data_folder = os.path.join(current_path, "data")  # Adjust the relative path to the data folder
sys.path.append(data_folder)  # Add the data folder to the system path


from data.prepare_data import preprocess_data_with_validation, preprocess_data_without_validation

# Define the dataset path
data_dir = os.path.join(data_folder, "Car-Bike-Dataset")

# Choose between preprocessing with or without validation
use_validation = False  # Set to False if you do not need a validation set

if use_validation:
    train_images, val_images, test_images, train_labels, val_labels, test_labels = preprocess_data_with_validation('data\Car-Bike-Dataset')
    # Reshape images for Logistic Regression (Flatten 2D images into 1D vectors)
    train_images = train_images.reshape(train_images.shape[0], -1)
    val_images = val_images.reshape(val_images.shape[0], -1)
    test_images = test_images.reshape(test_images.shape[0], -1)

    # Standardize features
    scaler = StandardScaler()
    train_images = scaler.fit_transform(train_images)
    val_images = scaler.transform(val_images)
    test_images = scaler.transform(test_images)

    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=50)  # Choose the number of components
    train_images = pca.fit_transform(train_images)
    val_images = pca.transform(val_images)
    test_images = pca.transform(test_images)
else:
    train_images, test_images, train_labels, test_labels = preprocess_data_without_validation('data\Car-Bike-Dataset')
    # Reshape images for Logistic Regression (Flatten 2D images into 1D vectors)
    train_images = train_images.reshape(train_images.shape[0], -1)
    test_images = test_images.reshape(test_images.shape[0], -1)
    # Standardize features
    scaler = StandardScaler()
    train_images = scaler.fit_transform(train_images)
    test_images = scaler.transform(test_images)
    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=50)  # Choose the number of components
    train_images = pca.fit_transform(train_images)
    test_images = pca.transform(test_images)
    val_images, val_labels = None, None  # No validation set in this case

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


# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self, input_dim=50):  # Input dimension matches PCA components
        super(SimpleNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2),  # Output layer for 2 classes (Bike, Car)
        )

    def forward(self, x):
        return self.fc(x)


# Initialize the model, loss function, and optimizer
model = SimpleNN(input_dim=50)  # Use 50 components as defined in PCA
criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for multi-class classification
optimizer = optim.SGD(model.parameters(), lr=0.001)


# Initialize lists to track metrics
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
num_epochs = 10

# Training loop with mini-batch processing
for epoch in range(num_epochs):
    epoch_loss = 0
    correct = 0
    total = 0

    for batch_images, batch_labels in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_images)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == batch_labels).sum().item()
        total += batch_labels.size(0)

    train_loss = epoch_loss / len(train_loader)
    train_accuracy = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # Validation step
    if use_validation:
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                output = model(val_images)
                loss = criterion(output, val_labels)
                val_loss += loss.item()
                _, predicted = torch.max(output, 1)
                val_correct += (predicted == val_labels).sum().item()
                val_total += val_labels.size(0)

        val_loss /= len(val_loader)
        val_accuracy = val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, accuracy: {train_accuracy:.4f}')
# Plot the metrics
plt.figure(figsize=(12, 6))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
if val_losses:
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Over Epochs')
plt.legend()
plt.grid(True)

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy')
if val_accuracies:
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Over Epochs')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


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


# Generate the classification report
print("\nClassification Report:")
print(classification_report(test_labels.long(), test_predicted.long()))

# Save the trained model
model_path = os.path.join(current_path, "notebooks", "simple_nn.pth")
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")