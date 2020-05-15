#!/usr/bin/env python
import numpy as np 
import pandas as pd 
import os
import cv2
from tensorflow import keras
import tensorflow as tf

main_path = './dataset'
images_path = os.path.join(main_path, 'images')
test_path = os.path.join(main_path, 'test.csv')
train_path = os.path.join(main_path, 'train.csv')

test_data = pd.read_csv(test_path)
train_data = pd.read_csv(train_path)

train_data.head()

def load_image(image_id):
    file_path = image_id + ".jpg"
    image = cv2.imread(os.path.join(images_path, file_path))
    k = 0.20
    height, wide, _ = image.shape
    dim = (int(wide*k), int(height*k))
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

input_shape = load_image(test_data['image_id'][0]).shape
print(input_shape)

X_train = train_data["image_id"].apply(lambda id: os.path.join(images_path, id + '.jpg'))

X_train.head()

y_train = train_data.loc[:, 'healthy':'scab']

y_train.head()

batch_size = 20
epochs = 20

def load_dataset(filename, res):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float16) / 255.0
    image = tf.image.resize(image, input_shape[:2])
    return image, res


train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((X_train, y_train))
    .map(load_dataset)
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


model.fit(train_dataset, epochs=20, steps_per_epoch=X_train.shape[0]//batch_size)

os.makedirs(os.path.join('.', 'models'))
model.save(os.path.join('.', 'models', 'model.hdf5'))