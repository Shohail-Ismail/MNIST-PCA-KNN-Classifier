from sklearn.metrics import accuracy_score  # For calculating accuracy
from utils import *  # Utility functions for data loading and model management
import system  # Module containing feature extraction functions


def main():
    # Load the trained model
    model = load_model()

    # Evaluate model on noisy test data
    noise_test_images, noise_test_labels = get_dataset('noise_test')
    noise_test_feature_vectors = system.image_to_reduced_feature(noise_test_images)
    noise_test_predictions = model.predict(noise_test_feature_vectors)
    noise_test_accuracy = accuracy_score(noise_test_labels, noise_test_predictions)
    print(f"Accuracy on noise_test set: {noise_test_accuracy * 100:.2f}%")

    # Evaluate model on masked test data
    mask_test_images, mask_test_labels = get_dataset('mask_test')
    mask_test_feature_vectors = system.image_to_reduced_feature(mask_test_images)
    mask_test_predictions = model.predict(mask_test_feature_vectors)
    mask_test_accuracy = accuracy_score(mask_test_labels, mask_test_predictions)
    print(f"Accuracy on mask_test set: {mask_test_accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
