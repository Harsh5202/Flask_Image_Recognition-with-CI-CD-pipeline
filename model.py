"""
Model Module for Hand Sign Digit Recognition
This module handles image preprocessing and prediction using a trained Keras model.
"""

# Importing required libs
from keras.models import load_model
from keras.utils import img_to_array
import numpy as np
from PIL import Image

# Loading model
model = load_model("digit_model.h5")


# Preparing and pre-processing the image
def preprocess_img(img_path):
    """
    Preprocess an image for model prediction.

    This function loads an image, resizes it to 224x224 pixels,
    normalizes pixel values to [0, 1], and reshapes it for the model.

    Args:
        img_path (str or file-like object): Path to the image file or file stream

    Returns:
        numpy.ndarray: Preprocessed image array with shape (1, 224, 224, 3)

    Raises:
        FileNotFoundError: If the image file does not exist
        IOError: If there's an error opening or processing the image
    """
    op_img = Image.open(img_path)
    img_resize = op_img.resize((224, 224))
    img2arr = img_to_array(img_resize) / 255.0
    img_reshape = img2arr.reshape(1, 224, 224, 3)
    return img_reshape


# Predicting function
def predict_result(predict):
    """
    Predict the digit class from a preprocessed image.

    Args:
        predict (numpy.ndarray): Preprocessed image array from preprocess_img()

    Returns:
        numpy.integer: Predicted digit class (0-9)
    """
    pred = model.predict(predict, verbose=0)
    return np.argmax(pred[0], axis=-1)
