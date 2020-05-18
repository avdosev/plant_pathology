import tensorflow as tf
from config import input_shape


def load_dataset(filename, res=None):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float16) / 255.0
    image = tf.image.resize(image, input_shape[:2])
    if res is None:
        return image
    return image, res
