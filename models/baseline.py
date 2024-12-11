import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score

# 1. Define image transformations
# Adjust the transforms as needed for your images.
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 2. Load the dataset using ImageFolder
# Assuming 'project/data/' contains two folders: 'car' and 'motorcycle'
data_dir = "data"  # Relative to your 'baseline.py' file location
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Let's say we just run the baseline on the entire dataset.
# If you have a separate test set, you could load that similarly or split your dataset.
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# Print class to index mapping to confirm which class is 0 and which is 1
print("Class to index mapping:", dataset.class_to_idx)
# Suppose it prints {'car': 0, 'motorcycle': 1} - this is often alphabetical.

# 3. Create a baseline model that always predicts one class.
# In this case, we'll always predict "car" which we assume is class 0.
class ConstantModel(torch.nn.Module):
    def __init__(self, constant_class=0):
        super().__init__()
        self.constant_class = constant_class

    def forward(self, x):
        batch_size = x.shape[0]
        return torch.full((batch_size,), self.constant_class, dtype=torch.long)

baseline_model = ConstantModel(constant_class=0)

# 4. Run the baseline predictions
y_true = []
y_pred = []

with torch.no_grad():
    for images_batch, labels_batch in dataloader:
        predictions = baseline_model(images_batch)  # All zeros ("car")
        y_true.extend(labels_batch.tolist())
        y_pred.extend(predictions.tolist())

# 5. Compute metrics
acc = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, pos_label=0)
recall = recall_score(y_true, y_pred, pos_label=0)

print("Baseline Results:")
print("Accuracy:", acc)
print("Precision (for 'car'):", precision)
print("Recall (for 'car'):", recall)
