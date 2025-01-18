# Import necessary modules
import torch
import torch.nn as nn
from data.prepare_data import preprocess_data_with_validation, preprocess_data_without_validation
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns  # For better visualization of the confusion matrix

# Run this by: python -m models.LogisticRegression

# Define Logistic Regression Model
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # Linear layer for binary classification
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# Load preprocessed data
train_images, test_images, train_labels, test_labels = preprocess_data_without_validation('data\Car-Bike-Dataset')
#  Reshape images for Logistic Regression (Flatten 2D images into 1D vectors)
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

# # Load data with validation
# train_images, val_images, test_images, train_labels, val_labels, test_labels = preprocess_data_with_validation('data\Car-Bike-Dataset')
# # Reshape images for Logistic Regression (Flatten 2D images into 1D vectors)
# train_images = train_images.reshape(train_images.shape[0], -1)
# val_images = val_images.reshape(val_images.shape[0], -1)
# test_images = test_images.reshape(test_images.shape[0], -1)

# # Standardize features
# scaler = StandardScaler()
# train_images = scaler.fit_transform(train_images)
# val_images = scaler.transform(val_images)
# test_images = scaler.transform(test_images)

# # Apply PCA to reduce dimensionality
# pca = PCA(n_components=50)  # Choose the number of components
# train_images = pca.fit_transform(train_images)
# val_images = pca.transform(val_images)
# test_images = pca.transform(test_images)

# Convert NumPy arrays to PyTorch tensors
train_images_tensor = torch.tensor(train_images, dtype=torch.float32)
# val_images_tensor = torch.tensor(val_images, dtype=torch.float32)
test_images_tensor = torch.tensor(test_images, dtype=torch.float32)
train_labels_tensor = torch.tensor(train_labels, dtype=torch.float32).view(-1, 1)  # Reshape labels for binary classification
# val_labels_tensor = torch.tensor(val_labels, dtype=torch.float32).view(-1, 1)
test_labels_tensor = torch.tensor(test_labels, dtype=torch.float32).view(-1, 1)

# Get the input dimension (number of features after PCA)
input_dim = train_images.shape[1]  # This will be the number of PCA components (50 in your case)

# Create the model instance
model = LogisticRegressionModel(input_dim)

# Define loss and optimizer
loss_function = torch.nn.BCELoss() 
learning_rate = 0.0001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
num_epochs = 100
train_losses = []
val_losses = []
# Training loop with accuracy tracking
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    optimizer.zero_grad()  # Clear gradients for the next iteration
    # Forward pass and loss computation
    predicted = model(train_images_tensor)
    loss = loss_function(predicted, train_labels_tensor)
    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())
    
    # Compute training accuracy
    train_predicted_cls = (predicted >= 0.5).float()
    train_accuracy = train_predicted_cls.eq(train_labels_tensor).sum().item() / train_labels_tensor.shape[0]
    train_accuracies.append(train_accuracy)

    # # Validation
    # with torch.no_grad():
    #     # Forward pass and loss computation
    #     predicted_val = model(val_images_tensor)
    #     loss_val = loss_function(predicted_val, val_labels_tensor)
    #     val_losses.append(loss_val.item())
        
    #     # Compute validation accuracy
    #     val_predicted_cls = (predicted_val >= 0.5).float()
    #     val_accuracy = val_predicted_cls.eq(val_labels_tensor).sum().item() / val_labels_tensor.shape[0]
    #     val_accuracies.append(val_accuracy)

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Train Acc: {train_accuracy:.4f}')

# Plot training and validation loss over epochs
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(num_epochs), train_losses, label='Training Loss')
# plt.plot(range(num_epochs), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.title('Loss Over Epochs')

# Plot training and validation accuracy over epochs
plt.subplot(1, 2, 2)
plt.plot(range(num_epochs), train_accuracies, label='Training Accuracy')
# plt.plot(range(num_epochs), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.title('Accuracy Over Epochs')

plt.tight_layout()
plt.show()


# Evaluation
with torch.no_grad():
    y_predicted = model(test_images_tensor)  # Predicted probabilities
    y_predicted_cls = (y_predicted >= 0.5).float()  # Convert probabilities to binary class labels
    accuracy = y_predicted_cls.eq(test_labels_tensor).sum().item() / test_labels_tensor.shape[0]
    print(f'Accuracy on test set: {accuracy:.4f}')

# Generate the classification report
print("\nClassification Report:")
print(classification_report(test_labels_tensor.long(), y_predicted_cls.long()))