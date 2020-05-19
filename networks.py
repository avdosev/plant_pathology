from tensorflow import keras


def my_model(input_shape, classes):
    return keras.Sequential([
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
        keras.layers.Dense(classes, activation='softmax')
    ])


def mobilenet_model(input_shape, classes):
    return keras.Sequential([
        keras.applications.MobileNetV2(input_shape=input_shape, weights='imagenet'),
        keras.layers.Dense(classes, activation='softmax')
    ])
