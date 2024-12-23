# Import necessary modules
import torch
import torch.nn as nn
import os
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns  # For better visualization of the confusion matrix

# Run this by: python -m models.Logistic_Regression
# Define the dataset path
# Add the `data` folder to the system path
current_path = os.getcwd()
data_folder = os.path.join(current_path, "data")
sys.path.append(data_folder)
data_dir = os.path.join(data_folder, "Car-Bike-Dataset")

import prepare_data
# Choose between preprocessing with or without validation
use_validation = True

if use_validation:
    train_images, val_images, test_images, train_labels, val_labels, test_labels = prepare_data.preprocess_data_with_validation(
        dataset_path=data_dir, validation_size=0.2, test_size=0.5, pca_components=50
    )
else:
    train_images, test_images, train_labels, test_labels = prepare_data.preprocess_data_without_validation(
        dataset_path=data_dir, test_size=0.2, pca_components=50
    )
    val_images, val_labels = None, None

# Convert NumPy arrays to PyTorch tensors
train_images_tensor = torch.tensor(train_images, dtype=torch.float32)
test_images_tensor = torch.tensor(test_images, dtype=torch.float32)
train_labels_tensor = torch.tensor(train_labels, dtype=torch.float32).view(-1, 1)  # Reshape labels for binary classification
test_labels_tensor = torch.tensor(test_labels, dtype=torch.float32).view(-1, 1)

if use_validation:
    val_images_tensor = torch.tensor(val_images, dtype=torch.float32)
    val_labels_tensor = torch.tensor(val_labels, dtype=torch.float32).view(-1, 1)

# Define Logistic Regression Model
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # Linear layer for binary classification
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))
    
# Get the input dimension (number of features after PCA)
input_dim = train_images.shape[1]  # This will be the number of PCA components (50 in your case)

# Create the model instance
model = LogisticRegressionModel(input_dim)

# Define loss and optimizer
loss_function = torch.nn.BCELoss() 
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
num_epochs = 100
train_losses = []
val_losses = []

# Training loop
def train_and_validate(model, loss_function, optimizer, epochs=100):
    for epoch in range(epochs):
        model.train()  # Set model to training mode for the training phase
        
        # Forward pass and loss computation
        predicted = model(train_images_tensor)
        loss = loss_function(predicted, train_labels_tensor)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        
        # Backward pass and optimization
        optimizer.zero_grad()  # Clear gradients for the next iteration
        
        # Validation
        if use_validation:  # Validation only if validation data is available
            with torch.no_grad():
                model.eval()  # Set model to evaluation mode for validation
                predicted_val = model(val_images_tensor)
                loss_val = loss_function(predicted_val, val_labels_tensor)
                val_losses.append(loss_val.item())

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            if use_validation:
                print(f'Validation Loss: {loss_val.item():.4f}')

# Train and validate the model
train_and_validate(model, loss_function, optimizer, epochs=num_epochs)

# Evaluate on test set
print("TEST: ")
# Evaluation
model.eval()
with torch.no_grad():
    y_predicted = model(test_images_tensor)  # Predicted probabilities
    y_predicted_cls = (y_predicted >= 0.5).float()  # Convert probabilities to binary class labels
    accuracy = y_predicted_cls.eq(test_labels_tensor).sum().item() / test_labels_tensor.shape[0]
    print(f'Accuracy on test set: {accuracy:.4f}')

# Plot training and validation loss over epochs
plt.plot(range(num_epochs), train_losses, label='Training Loss')
if use_validation:
    plt.plot(range(num_epochs), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.title('Training Loss')
plt.show()