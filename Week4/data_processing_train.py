import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

data_directory = "Week4/chest_x_ray/data/train"

categories = ["NORMAL", "PNEUMONIA"]

training_data = []
bad_file = []

IMG_SIZE = 500

def create_training_data():
    for category in categories:
        path = os.path.join(data_directory, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                file_ = os.path.join(path, img)
                img_array = cv2.imread(file_, cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                bad_file.append(file_)
                pass

create_training_data()
print(bad_file)

print(training_data[3000][1])
import random

random.shuffle(training_data)

for i in training_data[:10]:
    print(i[1])

X = []
y = []

for features, labels in training_data:
    X.append(features)
    y.append(labels)

print(X[:1])

X = np.array(X).reshape(-1, IMG_SIZE,IMG_SIZE, 1)


import pickle 
with open ("Week4/chest_x_ray/exported_models/X.pickle", "wb") as file_x:
    pickle.dump(X, file_x)
with open ("Week4/chest_x_ray/exported_models/y.pickle", "wb") as file_y:
    pickle.dump(y, file_y)


