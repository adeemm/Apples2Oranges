import os
import cv2
import numpy as np


IMG_SIZE = 50
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
TRAINING_DIR = os.path.join(CURRENT_DIR, "training_data")


# return one hot encoding of the image category
def encode_label(img):
    label = os.path.basename(os.path.dirname(img))

    if label == "apple":
        return [0, 1]
    elif label == "orange":
        return [1, 0]


# normalize pixel values of img
def normalize(img):
    for i in range(len(img)):
        img[i] = img[i] / 255.0

    return img


# store pre-processed images and their respective labels (remember to remove .DS_Store or Thumbs.db if they exist)
def generate_training_data():
    features = []
    labels = []

    for category in os.listdir(TRAINING_DIR):
        for img in os.listdir(os.path.join(TRAINING_DIR, category)):
            path = os.path.join(TRAINING_DIR, category, img)
            label = encode_label(path)
            img = cv2.resize(cv2.imread(path), (IMG_SIZE, IMG_SIZE))
            features.append(np.array(img).flatten())
            labels.append(label)

    normalize(features)
    features = np.array(features).reshape(-1, 7500, 1)
    labels = np.array(labels).reshape(-1, 2, 1)
    return features, labels


# splits dataset into training and validation sets (to check accuracy of predictions)
def split_dataset(x, y, ratio):
    split = int(ratio * x.shape[0])
    indices = np.random.permutation(x.shape[0])

    training_index, validation_index = indices[:split], indices[split:]
    x_training, x_validation = x[training_index, :], x[validation_index, :]
    y_training, y_validation = y[training_index, :], y[validation_index, :]

    print("Training dataset size: ", x_training.shape[0])
    print("Validation dataset size: ", x_validation.shape[0])
    return x_training, x_validation, y_training, y_validation
