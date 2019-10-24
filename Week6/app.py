import os
import shutil

import cv2
from flask import Flask, jsonify, render_template, request
from keras import models

app = Flask(__name__)

root_directory = os.path.dirname(os.path.abspath(__file__))

classifications = ['Normal', 'Pneumonia']


def image_conversion(location):
    img_size = 500
    img_array = cv2.imread(location, cv2.IMREAD_GRAYSCALE)
    new_img_array = cv2.resize(img_array, (img_size, img_size))
    return new_img_array.reshape(-1, img_size, img_size, 1)


model = models.load_model('Week5/models/pneumonia_convnn_1570158292.h5')


@app.route('/')
def index():
    return render_template('upload.html')


@app.route('/upload', methods=['POST'])
def upload():
    prediction_score = '0.5'
    # target = os.path.join(root_directory,  'static/')

    # if len(os.listdir(target)) != 0:
    #     shutil.rmtree(target)

    # if not os.path.isdir(target):
    #     os.mkdir(target)
    # for image_ in request.files.getlist("file"):
    #     filename = image_.filename
    #     destination = "/".join([target, filename])
    #     image_.save(destination)

    # for item in os.listdir(target):
    #     item_path = os.path.join(target, item)
    #     model_score = int(model.predict([image_conversion(item_path)])[0][0])
    #     prediction_score = classifications[model_score]

    return render_template('complete.html', img_array=prediction_score)


if __name__ == '__main__':
    app.run(port=4555, debug=True)
