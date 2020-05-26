import tensorflow as tf
from config import input_shape
from albumentations import Compose, OneOf,\
    Blur, MotionBlur, MedianBlur, HorizontalFlip, VerticalFlip, ShiftScaleRotate


def load_dataset(filename, res=None):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, input_shape[:2])
    if res is None:
        return image
    return image, res


def augment(image):
    aug = Compose([
        OneOf([
            MotionBlur(),
            MedianBlur(blur_limit=3),
            Blur(blur_limit=10),
        ]),
        OneOf([
            Compose([
                HorizontalFlip(),
                VerticalFlip(),
            ]),
            ShiftScaleRotate()
        ])
    ])
    return aug(image=image.numpy())['image']


def data_augment(image, label=None):
    [image] = tf.py_function(augment, [image], [tf.float32])
    image = tf.reshape(image, input_shape)
    if label is None:
        return image
    else:
        return image, label
