#!/usr/bin/env python
# coding: utf-8

# In[33]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import cv2
from tensorflow import keras
import tensorflow as tf


# In[2]:


# mainPath = '../input/plant-pathology-2020-fgvc7'
main_path = './dataset'
images_path = os.path.join(main_path, 'images')
test_path = os.path.join(main_path, 'test.csv')
train_path = os.path.join(main_path, 'train.csv')


# In[3]:


test_data = pd.read_csv(test_path)
train_data = pd.read_csv(train_path)


# In[4]:


train_data.head()


# In[10]:


def load_image(image_id):
    file_path = image_id + ".jpg"
    image = cv2.imread(os.path.join(images_path, file_path))
    k = 0.20
    height, wide, _ = image.shape
    dim = (int(wide*k), int(height*k))
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# In[30]:


input_shape = load_image(test_data['image_id'][0]).shape
print(input_shape)


# In[53]:


X_train = train_data["image_id"].apply(lambda id: os.path.join(images_path, id + '.jpg'))


# In[54]:


X_train.head()


# In[43]:


y_train = train_data.loc[:, 'healthy':'scab']


# In[44]:


y_train.head()


# In[67]:


batch_size = 20
epochs = 20


# In[78]:


def load_dataset(filename, res):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float16) / 255.0
    image = tf.image.resize(image, input_shape[:2])
    return image, res


# In[85]:


train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((X_train, y_train))
    .map(load_dataset)
    .repeat()
    .shuffle(256)
    .batch(batch_size)
)


# In[90]:


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


# In[91]:


model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])


# In[92]:


model.summary()


# In[93]:


model.fit(train_dataset, epochs=20, steps_per_epoch=X_train.shape[0]//batch_size)


# In[ ]:


model.save(os.path.join('.', 'models', 'model.hdf5'))


# In[ ]:




