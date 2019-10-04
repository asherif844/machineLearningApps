import pickle

import keras
# import tensorflow as tf
from tensorflow.python.keras import backend
from keras.layers import (Activation, Conv2D, Dense, Dropout, Flatten,
                          MaxPooling2D)
from keras.models import Sequential

# print(tf.__version__)
print(keras.__version__)

x = pickle.load(open('Week4/chest_x_ray/exported_models/X.pickle', "rb"))
y = pickle.load(open('Week4/chest_x_ray/exported_models/y.pickle', "rb"))

print(x.shape)
# print(y.shape)


X = x/255.0
inputshape = X.shape[1:]

model = Sequential()

model.add(Conv2D(64, (3,3), input_shape = inputshape))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))



model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy']
              )

model.fit(X, y, batch_size=32, epochs = 10, validation_split=0.1)

import time 

model_name = f'pneumonia_convnn_{int(time.time())}.h5'
model_weights = f'pneumonia_convnn_wts_{int(time.time())}.h5'
model.save('Week4/chest_x_ray/exported_models/'+model_name)
model.save_weights('Week4/chest_x_ray/exported_models/'+model_weights)

model.summary()
model_loaded = model.load_weights('/Users/theahmedsherif/Dropbox/Python/conda_environments/machineLearningApps/Week4/chest_x_ray/exported_models/pneumonia_convnn_1570124819.h5')

model_loaded

# test this with the validation dataset by creating x_test and y_test  for scoring purposes
score = model.evaluate(x_test, y_test, batch_size=16)
