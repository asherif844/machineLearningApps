import os
import pprint
from collections import Counter, defaultdict

import cv2
import tensorflow as tf
from keras import models

CATEGORIES = ['NORMAL', 'PNEUMONIA']


def processing(path):
    IMG_SIZE = 500
    img_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


model = models.load_model(
    'Week4/chest_x_ray/exported_models/pneumonia_convnn_1570158292.h5')

# prediction = model.predict([processing('Week4/chest_x_ray/data/validation/PNEUMONIA/person1_virus_6.jpeg')])

path = 'Week4/chest_x_ray/data/validation/NORMAL/'
# for i in os.listdir(path)[:100]:
    # print(model.predict([processing(i)]))
    # print(i)
    # prediction_score = model.predict(
    #     [processing(path+i)])
    # print(CATEGORIES[int(prediction_score[0][0])])
    # print('--------')


# normal_count = Counter()
# pneumonia_count = Counter()

# for i in os.listdir(path):
#         prediction_score = model.predict(
#         [processing(path+i)])
#         a = int(prediction_score[0][0])
#         print(a)
        # if a == 0:
        #     normal_count+=1
        # else:
        #     pneumonia_count+=1

# print(normal_count)
# print(pneumonia_count)
confusion_matrix = []

validation_path = 'Week4/chest_x_ray/data/validation/'
for i in CATEGORIES:
    full_path = validation_path+i
    # print(full_path)
    for item in os.listdir(full_path):
        paths_ = os.path.join(full_path, item)
        a = int(model.predict(
        [processing(paths_)])[0][0])
        # print(i)
        result = f'ACTUAL: {i}, PREDICTION: {CATEGORIES[a]}'
        confusion_matrix.append(result)

print(Counter(confusion_matrix))
# confusion_matrix_dict_2 = dict()
confusion_matrix_dict = dict()
for i in confusion_matrix:
    confusion_matrix_dict[i] = confusion_matrix_dict.get(i, 0) + 1
    # confusion_matrix_dict_2[i] = confusion_matrix_dict_2.get(i,'a') +1
# print(confusion_matrix_dict)
pprint.pprint(confusion_matrix_dict)
