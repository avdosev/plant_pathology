#!/usr/bin/env python
import pandas as pd 
import os
import tensorflow as tf
from utils import load_dataset, data_augment
from config import *
from networks import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import math


train_data = pd.read_csv(train_path)

train_data.head()

X_data = train_data["image_id"].apply(lambda id: os.path.join(images_path, id + '.jpg'))
y_data = train_data.loc[:, 'healthy':'scab']

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data,  test_size=0.20,  random_state=42)


train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((X_train, y_train))
    .map(load_dataset)
    .map(data_augment)
    .repeat()
    .shuffle(256)
    .batch(batch_size)
)

test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(X_test)
    .map(load_dataset)
    .batch(batch_size)
)

classes = y_train.shape[1]

model = mobilenet_model(input_shape, classes)


model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])


model.summary()

if not os.path.exists(models_path):
    os.makedirs(models_path)

model.fit(train_dataset,
          epochs=epochs,
          steps_per_epoch=X_train.shape[0]//batch_size*2,
          callbacks=[
              keras.callbacks.EarlyStopping(monitor="loss", min_delta=0, patience=7, verbose=0, mode="min"),
              keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join(models_path, 'model_best.hdf5'),
                    save_weights_only=False,
                    monitor='loss',
                    mode='min',
                    save_best_only=True
              )
          ]
          )

steps = math.ceil(X_test.shape[0]/batch_size)
predicted = model.predict(test_dataset, verbose=1, steps=steps)

score = roc_auc_score(y_test, predicted)

print("roc auc score:", score)

model.save(os.path.join(models_path, 'model.hdf5'))
