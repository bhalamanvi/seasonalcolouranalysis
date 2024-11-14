#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

# Load and preprocess images
IMG_SIZE = (224, 224)  # Resize images to 224x224 pixels
data_dir = "C:/Users/rgbha/OneDrive/Desktop/Seasons"
categories = sorted(os.listdir(data_dir))  # Assumes each folder is a category (season)

images = []
labels = []

for category in categories:
    path = os.path.join(data_dir, category)
    class_num = categories.index(category)
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img))
            img_array = cv2.resize(img_array, IMG_SIZE)
            images.append(img_array)
            labels.append(class_num)
        except Exception as e:
            print(f"Error loading image {img}: {e}")

images = np.array(images) / 255.0  # Normalize pixel values
labels = np.array(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=0)



# In[2]:


# Building the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(12, activation='softmax')  # 12 output classes for the 12 seasons
])
from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train, num_classes=12)
y_test = to_categorical(y_test, num_classes=12)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])





# In[3]:


# Train the model
model.fit(X_train, y_train, epochs=25, validation_data=(X_test, y_test))


# In[8]:


model.save('C:/Users/rgbha/OneDrive/Desktop/season_model.h5')

