from utils import *
import system


def main():
    # Load training data
    train_images, train_labels = get_dataset('train')

    # Extract dimension-reduced features for training
    train_feature_vectors = system.image_to_reduced_feature(train_images, 'train')

    # Train the classifier
    model = system.training_model(train_feature_vectors, train_labels)

    # Save the trained model
    save_model(model)


# Only run main() if this script is executed directly
if __name__ == "__main__":
    main()