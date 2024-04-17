import keras
import numpy as np
from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPool2D
from keras.models import Sequential
from tensorflow.keras.utils import to_categorical

classes = ['boar', 'crow', 'tiger']
num_classes = len(classes)
image_size = 50


def main() -> None:
    data = np.load("./animal_augmented.npy", allow_pickle=True)
    X_train = data.item()['X_train']
    X_test = data.item()['X_test']
    y_train = data.item()['y_train']
    y_test = data.item()['y_test']

    X_train = X_train.astype("float") / 256
    X_test = X_test.astype("float") / 256
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    model = keras.models.load_model('./animal_cnn.h5')
    model_augmented = model_train(X_train, y_train)
    # model_augmented = keras.models.load_model('./animal_cnn_augmented.h5')
    model_eval(model, X_test, y_test)
    model_eval(model_augmented, X_test, y_test)


def model_train(X_train, y_train) -> Sequential:
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=X_train.shape[1:]))
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

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.fit(X_train, y_train, batch_size=32, epochs=40)

    model.save('./animal_cnn_augmented.h5')

    return model


def model_eval(model: Sequential, X_test, y_test) -> None:
    scores = model.evaluate(X_test, y_test, verbose=1)
    print('Test Loss: ', scores[0])
    print('Test Accuracy ', scores[1])


if __name__ == '__main__':
    main()
