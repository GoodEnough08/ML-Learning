import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron    # Used for simple linear classification tasks.

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from tensorflow.keras.models import Sequential     # Sequential lets you build a neural network layer-by-layer in Keras.

from tensorflow.keras.layers import Dense     #Dense makes the final predictions
from tensorflow.keras.layers import Conv2D     # Conv2D extracts features
from tensorflow.keras.layers import Flatten    # Flatten reshapes them

from tensorflow.keras.layers import MaxPooling2D     # MaxPooling2D reduces size
from tensorflow.keras.layers import Dropout          # Dropout prevents overfitting

from tensorflow.keras.utils import to_categorical

""" Using Perceptrons"""
df = pd.read_csv("mnist_train.csv")
df_test = pd.read_csv("mnist_test.csv")

X_train = df.drop("label", axis=1).values
y_train = df["label"].values
X_test = df_test.drop("label", axis=1).values
y_test = df_test["label"].values
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0
     

X_train_img = X_train.reshape(-1, 28, 28)
X_test_img = X_test.reshape(-1, 28, 28)
     

y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)
     

perceptron = Sequential([
    Flatten(input_shape=(28,28)),
    Dense(10, activation="softmax")
])
perceptron.compile(optimizer="sgd",loss='categorical_crossentropy',metrics=['accuracy'])
history_percp = perceptron.fit(X_train_img, y_train_cat, epochs=5, batch_size=32, validation_data=(X_test_img, y_test_cat), verbose=1)
acc_percp = perceptron.evaluate(X_test_img, y_test_cat, verbose=0)[1]
print("Accuracy of Perceptron:", acc_percp)

""" Using ANN"""
ann = Sequential([
    Flatten(input_shape=(28,28)),
    Dense(128, activation="relu"),
    Dense(64, activation="relu"),
    Dense(10, activation="softmax")
])
ann.compile(optimizer="adam",loss='categorical_crossentropy',metrics=['accuracy'])
history_ann = ann.fit(X_train_img, y_train_cat, epochs=10, batch_size=32, validation_data=(X_test_img, y_test_cat), verbose=1)
acc_ann = ann.evaluate(X_test_img, y_test_cat, verbose=0)[1]
print("Accuracy of ANN:", acc_ann)
""" Using CNN"""
cnn = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(10, activation="softmax")
])
cnn.compile(optimizer="adam",loss='categorical_crossentropy',metrics=['accuracy'])
history_cnn = cnn.fit(X_train_img, y_train_cat, epochs=10, batch_size=32, validation_data=(X_test_img, y_test_cat), verbose=1)
acc_cnn = cnn.evaluate(X_test_img, y_test_cat, verbose=0)[1]
print("Accuracy of CNN:", acc_cnn)
plt.figure(figsize=(12, 5))

# Perceptron
plt.subplot(1, 3, 1)
plt.plot(history_percp.history['accuracy'], label='train accuracy', color='blue')
plt.plot(history_percp.history['val_accuracy'], label='val accuracy', color='orange')
plt.title('Perceptron')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# ANN
plt.subplot(1, 3, 2)
plt.plot(history_ann.history['accuracy'], label='train accuracy', color='blue')
plt.plot(history_ann.history['val_accuracy'], label='val accuracy', color='orange')
plt.title('ANN')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# CNN
plt.subplot(1, 3, 3)
plt.plot(history_cnn.history['accuracy'], label='train accuracy', color='blue')
plt.plot(history_cnn.history['val_accuracy'], label='val accuracy', color='orange')
plt.title('CNN')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()