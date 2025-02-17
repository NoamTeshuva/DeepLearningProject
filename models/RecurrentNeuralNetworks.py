import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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
    train_images_pca = pca.fit_transform(train_images)
    test_images_pca = pca.transform(test_images)
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

# Define the RNN model
class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define RNN layers (using LSTM)
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Pass through LSTM
        out, _ = self.rnn(x, (h0, c0))

        # Pass through fully connected layer
        out = self.fc(out[:, -1, :])
        return out

# Hyperparameters
input_size = train_images.shape[1]  # Assuming input features are already flattened
hidden_size = 128
num_layers = 2
output_size = 2  # Number of classes

# Initialize the model, loss function, and optimizer
model = RNNClassifier(input_size, hidden_size, num_layers, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in loader:
            images = images.unsqueeze(1)  # Reshape to (batch_size, sequence_length=1, input_size)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_loss:.4f}")

def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.unsqueeze(1)  # Reshape to (batch_size, sequence_length=1, input_size)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == labels).sum().item()
    avg_loss = total_loss / len(loader)
    accuracy = correct / len(loader.dataset)
    print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

# Train the model
train(model, train_loader, criterion, optimizer, epochs=10)

# Evaluate on validation set (if available)
if use_validation and val_loader is not None:
    validate(model, val_loader, criterion)

# Evaluate on test set
validate(model, test_loader, criterion)

# Save the trained model
model_path = os.path.join(current_path, "notebooks", "rnn_model.pth")
os.makedirs(os.path.dirname(model_path), exist_ok=True)
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")
