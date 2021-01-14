import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from PIL import Image
def Predict(image_path):
    model = tf.keras.models.load_model('det_model.h5')
    
    img = cv2.imread(image_path)
    img = tf.image.rgb_to_grayscale(img).numpy()

    img = img.reshape(1,28,28,1,1)
    preds = model.predict(img)

    print("My Prediction: " + str(np.where(np.isclose(preds[0], max(preds[0])))[0][0]))

if __name__ == "__main__":
    Predict("img.jpg")
