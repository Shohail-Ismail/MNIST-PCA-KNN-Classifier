import os
import pandas as pd
from PIL import Image
import numpy as np
import sys
import joblib  # For saving and loading models
from sklearn.base import BaseEstimator, ClassifierMixin

class NullModel(BaseEstimator, ClassifierMixin):
    def __init__(self, default_prediction=0):
        self.default_prediction = default_prediction

    def fit(self, X, y):
        # Sets a default prediction based on the most common label in y
        if y is not None and len(y) > 0:
            self.default_prediction = np.bincount(y).argmax()
        else:
            self.default_prediction = 0
        return self

    def predict(self, X):
        # Returns the default prediction for all inputs
        return np.full((X.shape[0],), self.default_prediction)

def get_dataset(split):
    # Load dataset based on split type (train, noise_test, mask_test)
    csv_file_path = "./mnist_subset/image_labels.csv"
    df = pd.read_csv(csv_file_path)
    data_df = df[df['Split'] == split]

    # Select appropriate directory based on split type
    if split == 'train':
        data_images_dir = "./mnist_subset/images/"
    elif split == 'noise_test':
        data_images_dir = "./mnist_subset/noisy_images/"
    elif split == 'mask_test':
        data_images_dir = "./mnist_subset/masked_images/"
    else:
        print('Wrong Split!')
        sys.exit()

    data_images = []
    data_labels = []

    # Load and flatten images, collect labels
    for idx, row in data_df.iterrows():
        image_path = os.path.join(data_images_dir, row['Filename'])
        img = Image.open(image_path)
        data_images.append(np.array(img).flatten())
        data_labels.append(row['Label'])

    return np.array(data_images), np.array(data_labels)

def save_model(model, filename='trained_model.pkl'):
    # Save the model to a file
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def load_model(filename='trained_model.pkl'):
    # Load model from file with error handling if file is missing
    try:
        model = joblib.load(filename)
        print(f"Model loaded from {filename}")
        return model
    except FileNotFoundError:
        print(f"File {filename} not found.")
        sys.exit()