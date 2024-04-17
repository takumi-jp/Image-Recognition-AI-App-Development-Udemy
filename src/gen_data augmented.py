import glob

import numpy as np
from PIL import Image

classes = ['boar', 'crow', 'tiger']
num_classes = len(classes)
image_size = 50
num_testdata = 100

X_train = []
Y_train = []
X_test = []
Y_test = []

for index, class_name in enumerate(classes):
    photos_dir = "data/" + class_name
    files = glob.glob(photos_dir + "/*.jpg")
    for i, file in enumerate(files):
        if i >= 190:
            break
        image = Image.open(file)
        image = image.convert("RGB").resize((image_size, image_size))
        data = np.asarray(image)

        if i < num_classes:
            X_test.append(data)
            Y_test.append(index)
        else:
            for angle in range(-20, 21, 5):
                img_r = image.rotate(angle)
                data = np.asarray(img_r)
                X_train.append(data)
                Y_train.append(index)

                img_trans = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                data = np.asarray(img_trans)
                X_train.append(data)
                Y_train.append(index)

X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)

xy = {"X_train": X_train, "X_test": X_test,
      "y_train": Y_train, "y_test": Y_test}

np.save("animal_augmented.npy", xy)
