import numpy as np 
import pandas as pd 
import os
import cv2
from tensorflow import keras
import tensorflow as tf
from config import *
from utils import *
import math
from datetime import datetime

model = keras.models.load_model('./models/model.hdf5')

X_test = pd.read_csv(test_path)['image_id'].apply(lambda id: os.path.join(images_path, id + '.jpg'))

test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(X_test)
    .map(load_dataset)
    .batch(batch_size)
)

steps = math.ceil(X_test.shape[0]/batch_size)
probs = model.predict(test_dataset, verbose=1, steps=steps)

sub = pd.read_csv(sub_path)
sub.loc[:, 'healthy':] = probs
submission_path = os.path.join('.', 'output', datetime.now().strftime("%Y%m%d_%H%M"))
if not os.path.exists(submission_path):
    os.makedirs(submission_path)
sub.to_csv(os.path.join(submission_path, 'submission.csv'), index=False)
sub.head()
