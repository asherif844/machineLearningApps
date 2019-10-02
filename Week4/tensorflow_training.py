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

