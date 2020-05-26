from tensorflow import keras
import efficientnet.tfkeras as efn


def my_model(input_shape, classes):
    return keras.Sequential([
        keras.layers.Conv2D(64, (5, 5), input_shape=input_shape, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
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
        keras.applications.MobileNetV2(input_shape=input_shape, weights='imagenet', include_top=False),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(classes, activation='softmax')
    ])


def resnet_model(input_shape, classes):
    return keras.Sequential([
        keras.applications.InceptionResNetV2(input_shape=input_shape, weights='imagenet', include_top=False),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(classes, activation='softmax')
    ])


def resnet_model_v2(input_shape, classes):
    return keras.Sequential([
        keras.applications.InceptionResNetV2(input_shape=input_shape, weights='imagenet', include_top=False),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(classes, activation='softmax')
    ])


def effnet_model_b2(input_shape, classes):
    base_model = efn.EfficientNetB2(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x1 = keras.layers.GlobalAveragePooling2D()(x)
    x2 = keras.layers.GlobalMaxPooling2D()(x)
    x = keras.layers.concatenate([x1, x2])
    x = keras.layers.Dropout(0.2)(x)
    predictions = keras.layers.Dense(classes, activation="softmax")(x)
    return keras.models.Model(inputs=base_model.input, outputs=predictions)
