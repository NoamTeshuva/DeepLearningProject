# Deep Learning Course - Final Project

## Car vs Bike Classification
Submitters: Noy Rosenbaum and Noam Tshuva 

## Table of Contects
- [Deep Learning Course - Final Project](#deep-learning-course---final-project)
  - [Project Title: Car vs Bike Classification](#project-title-car-vs-bike-classification)
  - [Overview](#overview)
  - [Dataset](#dataset)
    - [Preprocessing Data](#preprocessing-data)
  - [Models](#models)
    - [Baseline Model](#baseline-model)
    - [Logistic Regression](#logistic-regression)
    - [Fully Connected Neural Network](#fully-connected)
    - [CNN](#cnn)

### Overview
This project is part of my Machine Learning coursework, where the goal is to classify images into two categories: **Car** and **Bike**. The dataset consists of labeled images stored in separate folders, and the project is structured to demonstrate different machine learning techniques, using the following models:
- **Baseline** as the simplest model used as a reference point. (50% accuracy)
- **Logistic Regression** (74% accuracy)
- **Fully connected neural network** (81% accuracy) 
- **CNN** (95% accuracy).

### Dataset
Link to the Kaggle's Dataset: [Dataset](https://www.kaggle.com/datasets/utkarshsaxenadn/car-vs-bike-classification-dataset/data) \
- The dataset is stored in the `data/carBikeDataset` directory.
- It contains two subfolders:
  - `Car/` for car images
  - `Bike/` for bike images
The dataset is balanced, with an equal number of images in each class
It consists 2000 images of bikes and 2000 images of cars, where bike is labeled as 0 and car is labeled as 1. \
Firstly, we did preprocess on our raw data in order to ease the algorithms’ work and customize the data for our needs. \
Secondly, we initialized the models with a set of hyper parameters and optimizers based on previous knowledge we had and were adjusting it in order to improve the performance based on train and validation outputs. 

#### Preprocessing Data
Before training our model to classify images into two classes (Bike and Car), we performed comprehensive preprocessing on the dataset.
The goal was to prepare the data in a form suitable for efficient training and robust models’ performance. \
Firstly, images were categorized into two classes: Bike and Car, with each class assigned a numerical label (0 for Bike, 1 for Car).
Secondly, images are **resized** to a fixed size of 300x200 pixels and were converted to grayscale. \
**Data augmentation** techniques are applied to each image, such as: horizontal flip and rotation. \
Lastly, the pixel values of the image were **normalized** to the range of [0,1]. \
The dataset is split at first as follows: 
- 80% training
- 10% validation
- 10% test
Since the hyper-parameters are not part of the optimization problem, we used the validation set to examine the performance obtained for different values of the hyper-parameters. \
After the optimal hyper-parameters are set in place, the dataset is split once again from scratch into:
- 80% training
- 20% test
For several models such as: Logistic Regression and Fully connected neural network we have **flattened** the images into a 1D vector, **standardizes** each feature (pixel value) to have a mean of 0 and a standard deviation of 1 and used **PCA** to reduce the dimensionality of the feature space by projecting data onto a lower-dimensional space. \
This data preparation pipeline ensured that the model was trained on clean, normalized, and augmented data, allowing it to generalize well to unseen data.

### Models
NOTE: *Get in the directory where you want to execute the program before.* \
To clone a repository locally do:
```
git clone <repo URL>
```
For example:
```
git clone https://github.com/NoamTeshuva/DeepLearningProject.git
```

#### Baseline Model
The first step in this project is creating a baseline model. This model:
- Always predicts the majority class (in this case, "Bike").
- Metrics are calculated to evaluate its performance:
  - **Accuracy**: 50%
  - **Precision** (for "Bike"): 50%
  - **Recall** (for "Bike"): 100%

These results serve as a benchmark for more advanced models.

#### Logistic Regression
run this code by these commands:
1. `cd` to the existing repo directory.
```sh
cd ~/DeepLearningProject
```
2. Run file:
```sh
python -m models.LogisticRegression
```
#### Fully Connected
run this code by these commands:
1. `cd` to the existing repo directory.
```sh
cd ~/DeepLearningProject
```
2. Run file:
```sh
python -m MultylayerNeuralNetworks
```
#### CNN
run this code by these commands:
1. `cd` to the existing repo directory.
```sh
cd ~/DeepLearningProject
```
2. Run file:
```sh
python -m models.ConvolutionalNeuralNetworks
```
