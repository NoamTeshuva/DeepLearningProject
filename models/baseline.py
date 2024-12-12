import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score
import os

# 1. Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 2. Construct the path to the dataset dynamically
# Get the current working directory
current_path = os.getcwd()

# Build the dataset path relative to the current working directory
# Assuming the dataset is in 'data/Car-Bike-Dataset'
data_dir = os.path.join(current_path, "..", "data", "Car-Bike-Dataset")


# Check if the dataset directory exists
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

# 3. Load the dataset using ImageFolder
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# 4. Create a DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# 5. Print class to index mapping to confirm which class is 0 and which is 1
print("Class to index mapping:", dataset.class_to_idx)
# This will output something like: {'Bike': 0, 'Car': 1} or {'Car': 0, 'Bike': 1}

# 6. Create a baseline model that always predicts one class.
# For example, always predicting 'Bike' which is class 0.
class ConstantModel(torch.nn.Module):
    def __init__(self, constant_class=0):
        super().__init__()
        self.constant_class = constant_class

    def forward(self, x):
        batch_size = x.shape[0]
        return torch.full((batch_size,), self.constant_class, dtype=torch.long)

# Initialize the baseline model to always predict class 0
baseline_model = ConstantModel(constant_class=0)

# 7. Run the baseline predictions
y_true = []
y_pred = []

with torch.no_grad():
    for images_batch, labels_batch in dataloader:
        predictions = baseline_model(images_batch)  # All zeros (predicting 'Bike')
        y_true.extend(labels_batch.tolist())
        y_pred.extend(predictions.tolist())

# 8. Compute metrics
acc = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, pos_label=0)
recall = recall_score(y_true, y_pred, pos_label=0)

print("Baseline Results:")
print("Accuracy:", acc)
print("Precision (for 'Bike'):", precision)
print("Recall (for 'Bike'):", recall)
