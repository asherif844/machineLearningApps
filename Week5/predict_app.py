import base64
import io

import keras
import numpy as np
from flask import Flask, jsonify, request
from keras import backend as K
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from PIL import Image

app = Flask(__name__)

def get_model():
    global model 
    model = load_model('models/pneumonia_convnn_1570158292.h5')
    print(" * Model Loaded!")

get_model()
app.run()