#!/usr/bin/env python
import numpy as np 
import pandas as pd 
import os
import cv2
from tensorflow import keras
import tensorflow as tf
from utils import load_dataset
from config import *


train_data = pd.read_csv(train_path)

train_data.head()

X_train = train_data["image_id"].apply(lambda id: os.path.join(images_path, id + '.jpg'))

X_train.head()

y_train = train_data.loc[:, 'healthy':'scab']

y_train.head()

train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((X_train, y_train))
    .map(load_dataset)
    .repeat()
    .shuffle(256)
    .batch(batch_size)
)

model = keras.Sequential([
    keras.layers.Conv2D(64, (5, 5), input_shape=input_shape, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.GlobalMaxPool2D(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(y_train.shape[1], activation='softmax')
])


model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])


model.summary()


model.fit(train_dataset, epochs=epochs, steps_per_epoch=X_train.shape[0]//batch_size)

models_path = os.path.join('.', 'models')
if not os.path.exists(models_path):
    os.makedirs(models_path)

model.save(os.path.join(models_path, 'model.hdf5'))