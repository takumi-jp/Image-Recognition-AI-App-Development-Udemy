import glob
import os

import numpy as np
from PIL import Image
from sklearn import model_selection

classes = ['boar', 'crow', 'tiger']
num_classes = len(classes)
image_size = 50

X = []
Y = []

for index, class_name in enumerate(classes):
    photos_dir = "data/" + class_name
    files = glob.glob(photos_dir + "/*.jpg")
    for i, file in enumerate(files):
        if i >= 190:
            break
        image = Image.open(file)
        image = image.convert("RGB").resize((image_size, image_size))
        data = np.asarray(image)
        X.append(data)
        Y.append(index)

X = np.array(X)
Y = np.array(Y)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y)
xy = {"X_train": X_train, "X_test": X_test,
      "y_train": y_train, "y_test": y_test}

np.save("animal.npy", xy)
