#!/usr/bin/env python
import pandas as pd 
import os
import tensorflow as tf
from utils import load_dataset, data_augment
from config import *
from networks import *


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
    .map(data_augment)
    .repeat()
    .shuffle(256)
    .batch(batch_size)
)

classes = y_train.shape[1]

model = resnet_model_v2(input_shape, classes)


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
                    filepath=os.path.join(models_path, 'model.hdf5'),
                    save_weights_only=False,
                    monitor='loss',
                    mode='min',
                    save_best_only=True
              )
          ]
          )
