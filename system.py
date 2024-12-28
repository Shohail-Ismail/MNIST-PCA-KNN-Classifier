import numpy as np
from scipy.spatial.distance import cdist
from utils import load_model

## Global variables
# PCA params set in training
PCA_MEAN = None
PCA_STD_DEV = None
PCA_COMPONENTS = None

##KNN for classification with k = 5 (5 neighbours) and uses inverse distance weighting
class KNN:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None
        self.pca_mean = None
        self.pca_std = None
        self.pca_components = None
    
    # Stores dimension-reduced features and labels for predicion
    def fit(self, training_features, training_labels):
        self.X_train = training_features
        self.y_train = training_labels

    # Predicts lables for new data
    def predict(self, test_features):
        # Euclidean distance between test features
        # (euclidean chosen over cosine method as it easier to implement and does not hurt performance much)
        distances = cdist(test_features, self.X_train, metric='euclidean')

        # Loops through each row in matrix
        predictions = []
        for i in range(distances.shape[0]):
            neighbor_dist_vector = distances[i, :]

            # Gets labels/distances of only 5 nearest neighbours (doesnt sort whole array for speed)
            nn_indices = np.argpartition(neighbor_dist_vector, self.k)[:self.k]
            nn_labels = self.y_train[nn_indices]
            nn_distances = neighbor_dist_vector[nn_indices]

            # Applies inverse-distance weighting with a/0 accounted for
            weights = 1.0 / (nn_distances + 1e-9)

            # Loops through labels and weights, summing labels and predicting highest-summed label
            label_weights = {}
            for lbl, w in zip(nn_labels, weights):
                if lbl not in label_weights:
                    label_weights[lbl] = 0.0
                label_weights[lbl] += w
            best_label = max(label_weights, key=label_weights.get)
            predictions.append(best_label)

        return np.array(predictions, dtype=np.int64)

# PCA for dimensionality reduction
def image_to_reduced_feature(images, mode='inference'):
    global PCA_MEAN, PCA_STD_DEV, PCA_COMPONENTS

    # Images cast to int64 format to prevent under/overflow
    images = images.astype(np.int64)

    # In 'train' mode, PCA params calculated and stored in global variabels
    # PCA-reduced features returned
    if mode == 'train':
        
        # Gets mean and standard deviation of training set, with 0 std. dev. accounted for
        PCA_MEAN = np.mean(images, axis=0)
        PCA_STD_DEV = np.std(images, axis=0, ddof=1)
        PCA_STD_DEV[PCA_STD_DEV == 0] = 1e-9

        # Performs SVD on standardised data to get components for PCA
        #(SVD is used instead of directly computing covariance matrix for effifciency)
        stdised_data = (images - PCA_MEAN) / PCA_STD_DEV
        left_singular_vects, singular_vals, right_singular_vects = np.linalg.svd(stdised_data, full_matrices=False)

        # Projects standardised data onto 55 transposed PCA components for feature reducion
        PCA_COMPONENTS = right_singular_vects[:55]
        reduced_features = np.dot(stdised_data, PCA_COMPONENTS.T)
        return reduced_features

    # In 'inference' mode, existing PCA params applied to new data
    else:
        # If they aren't in memory yet, load saved model's params
        if (PCA_MEAN is None or PCA_STD_DEV is None or PCA_COMPONENTS is None):
            existing_model = load_model('trained_model.pkl')
            PCA_MEAN = existing_model.pca_mean
            PCA_STD_DEV = existing_model.pca_std
            PCA_COMPONENTS = existing_model.pca_components
            
        # Standardises data then projects onto existing PCA components
        stdised_data = (images - PCA_MEAN) / PCA_STD_DEV
        reduced_features = np.dot(stdised_data, PCA_COMPONENTS.T)
        return reduced_features

# Instantiates KNN, saves PCA params for reuse, and fits classifier
def training_model(feature_vectors, labels):
    # Create KNN with k = 5
    knn = KNN(k=5)

    # Store PCA params in the classifier to be saved in trained_model.pkl
    knn.pca_mean = PCA_MEAN
    knn.pca_std = PCA_STD_DEV
    knn.pca_components = PCA_COMPONENTS

    # Fit classifier onto PCA-reduced features
    knn.fit(feature_vectors, labels)

    return knn