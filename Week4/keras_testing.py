# try testing with this
# https://towardsdatascience.com/basics-of-image-classification-with-keras-43779a299c8b

from keras.models import load_model
import cv2
import numpy as np

model = load_model('mlappsPythonJS/Week4/chest_x_ray/exported_models/first_try.h5')

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# img = cv2.imread('mlappsPythonJS/Week4/chest_x_ray/data/validation/NORMAL/IM-0001-0001.jpeg')
# img = cv2.resize(img,(320,240))
# img = np.reshape(img,[320,240,3])

# classes = model.predict_classes(img)

# print(classes)