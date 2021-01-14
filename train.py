import cv2
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers

# Hyperparamerters
num_classes = 10
input_shape = (28, 28, 1)
batch_size = 128
epochs = 12


def Load_data(dir):
    x = []
    y = []
    for i in range(10):
        for filename in os.listdir(dir+str(i)):
            if filename == '.DS_Store':
                continue
            img = cv2.imread(os.path.join(dir+str(i),filename))
            if img is not None:
                x.append(img)
                y.append(i)
    x = np.array(x)
    y = np.array(y)
    return x,y

def Train(X_Data,Y_Data):
    p = np.random.permutation(len(Y_Data))
    X_Data = X_Data[p]
    Y_Data = Y_Data[p]
    X_Data = tf.image.rgb_to_grayscale(X_Data).numpy()

    train_size = int(0.9*Y_Data.shape[0])
    X_Train = X_Data[:train_size]
    X_Test = X_Data[train_size:]
    Y_Train = Y_Data[:train_size]
    Y_Test = Y_Data[train_size:]

    X_Train = X_Train.astype("float32") / 255
    X_Test = X_Test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    X_Train = np.expand_dims(X_Train, -1)
    X_Test = np.expand_dims(X_Test, -1)


    # convert class vectors to binary class matrices
    Y_Train = keras.utils.to_categorical(Y_Train, num_classes)
    Y_Test = keras.utils.to_categorical(Y_Test, num_classes)
 
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(X_Train, Y_Train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    score = model.evaluate(X_Test, Y_Test, verbose=1)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    
    preds = model.predict(X_Test[X_Test.shape[0]-10:])
    # make predicition on the last 10 images in the testing set and show them 
    for i in range(10):
        # print(str(np.where(np.isclose(preds[i], max(preds[i])))[0][0]))
        plt.figure("The Prediction: "+str(np.where(np.isclose(preds[i], max(preds[i])))[0][0]))
        img = X_Test[X_Test.shape[0]-10+i]
        img = img.reshape(img.shape[0],img.shape[1],img.shape[2])
        plt.imshow(img)
        plt.show()
    # save the model
    model.save("det_model.h5")


if __name__ == "__main__":
    X_Data,Y_Data = Load_data("./trainingSet/")
    Train(X_Data,Y_Data)

# X_Data = np.array(X_Data)
# Y_Data = np.array(Y_Data)
# print(X_Data.shape)
# print(Y_Data.shape)
# (X_Train, Y_Train), (X_Test, Y_Test) = keras.datasets.mnist.load_data()
# print(X_Train.shape)
# print(Y_Train.shape)

# X_Data = np.array(X_Data)
# Y_Data = np.array(Y_Data)
# print(X_Data.shape)

# X_Data = X_Data.reshape(X_Data.shape[0], X_Data.shape[1], X_Data.shape[2], 1)
# Y_Data = Y_Data.reshape((Y_Data.shape[0],1))
# p = np.random.permutation(len(Y_Data))
# X_Data = X_Data[p]
# Y_Data = Y_Data[p]
# X_Data = tf.image.rgb_to_grayscale(X_Data).numpy()
# print(X_Data.shape)
# train_size = int(0.8*Y_Data.shape[0])
# X_Train = X_Data[:train_size]
# X_Test = X_Data[train_size:]
# Y_Train = Y_Data[:train_size]
# Y_Test = Y_Data[train_size:]

# X_Train = X_Train.astype("float32") / 255
# X_Test = X_Test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
# X_Train = np.expand_dims(X_Train, -1)
# X_Test = np.expand_dims(X_Test, -1)
# print("x_train shape:", X_Train.shape)
# print(X_Train.shape[0], "train samples")
# print(X_Test.shape[0], "test samples")


# convert class vectors to binary class matrices
# Y_Train = keras.utils.to_categorical(Y_Train, num_classes)
# Y_Test = keras.utils.to_categorical(Y_Test, num_classes)
# print("y_train shape:", Y_Train.shape)


# print("Data Set Size = "+str(Y_Data.shape[0]))
# print("Training Set Size = "+str(Y_Train.shape[0]))
# print("Testing Set Size = "+str(Y_Test.shape[0]))
# train_model(X_Train,Y_Train,X_Test,Y_Test)

# (42000, 28, 28, 3)
# (42000,)
# (60000, 28, 28)
# (60000,)
# x_train shape: (60000, 28, 28, 1)
# 60000 train samples
# 10000 test samples
# y_train shape: (60000, 10)
# Data Set Size = 42000
# Training Set Size = 60000
# Testing Set Size = 10000

# x_train shape: (33600, 28, 28, 3, 1)
# 33600 train samples
# 8400 test samples
# y_train shape: (33600, 10)
# Data Set Size = 42000
# Training Set Size = 33600
# Testing Set Size = 8400