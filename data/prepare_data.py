# imports
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import random

# Set labels to images
def load_and_preprocess_images(dataset_path, augment=True, image_size=(100, 75)):
    class_folders = {'Bike': 0, 'Car': 1}
    all_images = []
    all_labels = []

    for class_name, label in class_folders.items():
        folder_path = os.path.join(dataset_path, class_name)
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            img = cv2.imread(image_path)
            if img is None:
                continue

            # Resize and grayscale
            img = cv2.resize(img, image_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            all_images.append(img)
            all_labels.append(label)

            if augment:
                if random.random() > 0.5:
                    img = cv2.flip(img, 1)
                angle = random.randint(-15, 15)  # Smaller rotation
                matrix = cv2.getRotationMatrix2D((image_size[0] // 2, image_size[1] // 2), angle, 1)
                img = cv2.warpAffine(img, matrix, image_size)
                all_images.append(img)
                all_labels.append(label)

    return np.array(all_images, dtype='float32'), np.array(all_labels)


# Devide the data to train and test, reshape, normalized and PCAed the data
def preprocess_data_without_validation(dataset_path, test_size=0.2, pca_components=50):
    # Load and preprocess images
    images, labels = load_and_preprocess_images(dataset_path)
    # Split into train and test (80% train, 20% test)
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=test_size, random_state=42)

    # Reshape images for Logistic Regression (Flatten 2D images into 1D vectors)
    train_images_flat = train_images.reshape(train_images.shape[0], -1)
    test_images_flat = test_images.reshape(test_images.shape[0], -1)

    # Standardize features
    scaler = StandardScaler()
    train_images_scaled = scaler.fit_transform(train_images_flat)
    test_images_scaled = scaler.transform(test_images_flat)

    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=pca_components)  # Choose the number of components
    train_images_pca = pca.fit_transform(train_images_scaled)
    test_images_pca = pca.transform(test_images_scaled)

    return train_images_pca, test_images_pca, train_labels, test_labels

# Devide the data to train, validation and test, reshape, normalized and PCAed the data
def preprocess_data_with_validation(dataset_path, validation_size=0.2, test_size=0.5, pca_components=50):
    # Load and preprocess images
    images, labels = load_and_preprocess_images(dataset_path)
    # Split into train (80%), temp (20%)
    train_images, temp_images, train_labels, temp_labels = train_test_split(images, labels, test_size=validation_size, random_state=42)

    # Split temp into validation (50% of temp) and test (50% of temp)
    val_images, test_images, val_labels, test_labels = train_test_split(temp_images, temp_labels, test_size=test_size, random_state=42)

    # Reshape images for Logistic Regression (Flatten 2D images into 1D vectors)
    train_images_vflat = train_images.reshape(train_images.shape[0], -1)
    val_images_flat = val_images.reshape(val_images.shape[0], -1)
    test_images_vflat = test_images.reshape(test_images.shape[0], -1)

    # Standardize features
    scaler = StandardScaler()
    train_images_vscaled = scaler.fit_transform(train_images_vflat)
    val_images_scaled = scaler.transform(val_images_flat)
    test_images_vscaled = scaler.transform(test_images_vflat)

    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=pca_components)  # Choose the number of components
    train_images_vpca = pca.fit_transform(train_images_vscaled)
    val_images_pca = pca.transform(val_images_scaled)
    test_images_vpca = pca.transform(test_images_vscaled)

    return train_images_vpca, val_images_pca, test_images_vpca, train_labels, val_labels, test_labels

