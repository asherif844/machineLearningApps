import os
import shutil

import cv2
from flask import Flask, jsonify, render_template, request
from keras import models


app = Flask(__name__, static_folder="images")

root_directory = os.path.dirname(os.path.abspath(__file__))
classifications = ['Normal', 'Pneumonia']


def image_conversion(location):
    img_size = 500
    img_array = cv2.imread(location, cv2.IMREAD_GRAYSCALE)
    new_img_array = cv2.resize(img_array, (img_size, img_size))
    return new_img_array.reshape(-1, img_size, img_size, 1)


model = models.load_model('Week5/models/pneumonia_convnn_1570158292.h5')


@app.route("/")
def homepage():
    return render_template('landingpage.html')


@app.route("/upload", methods=['POST'])
def upload():
    target = os.path.join(root_directory, 'images/')
    # print(f'targes is {target}')

    if len(os.listdir(target)) != 0:
        shutil.rmtree(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    for pnu_img in request.files.getlist("pneumonia_image"):
        # print(f'image name is {pnu_img}')
        filename = pnu_img.filename
        # print(f'filename is {filename}')
        destination = "/".join([target, filename])
        # destination = "/".join([target, 'temp.jpg'])
        # print(f'destination is {destination}')
        pnu_img.save(destination)

    for item in os.listdir(target):
        item_path = os.path.join(target, item)
        model_score = int(model.predict([image_conversion(item_path)])[0][0])
        prediction_score = classifications[model_score]

    return render_template('final.html', img_name=filename, dest=destination, score_array=prediction_score)


if __name__ == '__main__':
    app.run(debug=True)
