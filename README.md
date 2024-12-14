# Convolutional Neural Network (CNN) for Image Classification

## Project Overview
This project implements a Convolutional Neural Network (CNN) to classify images into two categories: cats and dogs. The model is trained using TensorFlow and Keras libraries. The dataset consists of:
- **Training Set**: 4000 images (2000 each of cats and dogs).
- **Test Set**: 1000 images (500 each of cats and dogs).
- **Single Prediction Folder**: A folder containing individual images for testing the model's predictions.

The model's performance is evaluated on the test dataset, and predictions are demonstrated on images from the "single_prediction" folder.

---

## Project Structure
### Directories:
- `dataset/training_set`: Contains subdirectories `cats` and `dogs` with training images.
- `dataset/test_set`: Contains subdirectories `cats` and `dogs` with test images.
- `dataset/single_prediction`: Contains images for single predictions.

### Scripts:
- **Model Training Script**: Implements the CNN and trains it on the dataset.
- **Single Prediction Script**: Predicts and visualizes results for individual images.

---

## Key Steps

### 1. Data Preprocessing
- The training set is augmented using techniques such as rescaling, shear, zoom, and horizontal flipping.
- The test set is rescaled for evaluation.

### 2. CNN Architecture
- **Input Layer**: Accepts 64x64 RGB images.
- **Convolutional Layers**: Extracts features using three convolutional blocks with increasing filters (32, 64, 128).
- **MaxPooling Layers**: Reduces dimensionality and computational cost.
- **Flattening**: Converts the matrix to a 1D vector.
- **Fully Connected Layers**: Combines features with a dense layer (128 units).
- **Output Layer**: Produces a binary classification result (cat or dog).

### 3. Model Training
- The model is compiled with the Adam optimizer and binary crossentropy loss function.
- Trained for 25 epochs with validation.

### 4. Single Prediction
- A helper function loads, preprocesses, and predicts results for a single image.
- Displays the image and prints the prediction ("Cat" or "Dog").

---

## Results
- The model achieves high accuracy on the test dataset.
- Single predictions are demonstrated with correctly classified results.

---

## Requirements
- Python 3.7+
- TensorFlow 2.x
- Keras (built into TensorFlow)
- Matplotlib
- Numpy
- ImageDataGenerator for data preprocessing

---

## How to Run
1. Clone the repository and navigate to the project folder.
2. Ensure the dataset structure is correct as mentioned above.
3. Install required libraries using:
   ```
   pip install tensorflow matplotlib numpy
   ```
4. Run the training script to train the model:
   ```
   python train_cnn.py
   ```
5. Run the single prediction script for testing individual images:
   ```
   python predict_image.py
   ```

---

## Medium Blog Post
Learn more about the project [here](#) (add the link to your Medium post).

---

## License
This project is open-source and available for modification and distribution under the MIT license.

