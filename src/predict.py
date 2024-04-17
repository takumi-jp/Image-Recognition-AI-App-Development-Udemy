import sys

import keras
import numpy as np
from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPool2D
from keras.models import Sequential
from PIL import Image

classes = ['boar', 'crow', 'tiger']
num_classes = len(classes)
image_size = 50


def build_model() -> Sequential:
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(50, 50, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3))
    model.add(Activation('softmax'))

    opt = keras.optimizers.RMSprop(learning_rate=0.0001, weight_decay=1e-6)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt, metrics=['accuracy'])

    model = keras.models.load_model('./animal_cnn.h5')

    return model


def main() -> None:
    image = Image.open(sys.argv[1])
    image = image.convert('RGB')
    image = image.resize((image_size, image_size))
    data = np.asarray(image) / 255
    X = []
    X.append(data)
    X = np.array(X)

    model = build_model()

    result = model.predict([X])[0]
    predicted = result.argmax()
    percentage = int(result[predicted] * 100)
    print("{0} ({1} %)".format(classes[predicted], percentage))


if __name__ == '__main__':
    main()
