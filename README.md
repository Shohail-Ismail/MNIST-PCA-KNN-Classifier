# MNIST PCA-KNN Classifier

   * [Background information](#background-information)
   * [Core features](#core-features)
   * [Limitations](#limitations)
   * [Environment setup](#environment-setup)


# Background information

PCA-KNN classifier for a modified MNIST dataset, where the goal is to accurately predict labels (digits 0 - 9) from 28x28 grayscale images of handwritten digits. 
The classifier is trained on 3,000 images and has been tested on 2 datasets, namely noisy images simulated using Gaussian noise and masked images with 15x15 block occlusions.




# Core features

- Principal Component Analysis (PCA) is used to reduce the dimensionality of the 786-dimensional (28x28) images, projecting them onto the top 55 principal components to retain the most information while
  discarding noise, thereby boosting computational efficiency and preventing overfitting.

- A K-Nearest Neighbours (KNN) classifier with a K value of 5 is used, along with inverse-distance weighting to prioritise closer neighbours in the feature space,
  which reduces misclassification caused by outliers.

- Tools for training the model, evaluating it on the test datasets, and saving/loading trained models are included for ease of testing and deployment.


# Limitations

 * The KNN classifier requires storing the entire training dataset in memory and performing computationally-expensive distance calculations during inference, which means
   that scalability to larger datasets is limited.
 * PCA parameters are manually tuned, so future iterations could benefit from automated optimisation methods.
 * Euclidean distance is used as the metric for KNN, which may not be optimal for datasets wih lots of overlap (for these, a metric like cosine similarity
   would be a better choice as it accounts for directional similarity).


# Environment setup

To run the project, ensure you have the following environment set up:

- **Required Python Version**: Python 3.12.8
- **Libraries**:
   - `numpy`
   - `scipy`
   - `scikit-learn`
   - `pandas`
   - `Pillow`
   - `joblib`

- **Structure**:
   - Ensure that the MNIST dataset and its subsets (`train`, `noise_test`, `mask_test`) are placed in the correct directories, as expected by the `get_dataset` function in `utils.py`.

- **Model**:
   - A trained model can be saved as `trained_model.pkl` and loaded for inference.

Run the training and evaluation scripts in the following order:

```bash
# To train and save the model
python train.py

# To evaluate the model on the test datasets
python evaluate.py 
