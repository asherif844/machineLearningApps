@ffimport os
import shutil

import cv2
from flask import Flask, jsonify, render_template, request
from keras import models
import pprint
from collections import Counter, defaultdict

app = Flask(__name__, static_folder="images")

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
print('---------')
print(APP_ROOT)
print('---------')


CATEGORIES = ['NORMAL', 'PNEUMONIA']


def processing(path):
    IMG_SIZE = 500
    img_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


model = models.load_model(
    'Week4/chest_x_ray/exported_models/pneumonia_convnn_1570158292.h5')

confusion_matrix = []
@app.route('/')
def index():
    return render_template('upload.html')


@app.route('/upload', methods=['POST'])
def upload():

    target = os.path.join(APP_ROOT, 'images/')
    print(target)

    if len(os.listdir(target)) != 0:
        shutil.rmtree(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist("file"):
        print(file)
        filename = file.filename
        destination = "/".join([target, filename])
        file.save(destination)

    test_image_path = 'Week6/images/'
    for item in os.listdir(test_image_path):
        paths_ = os.path.join(test_image_path, item)
        a = int(model.predict(
        [processing(paths_)])[0][0])
        # print(i)
        result = CATEGORIES[a]
        # confusion_matrix.append(result)

    return render_template('complete.html', img_array=result)


if __name__ == '__main__':

    app.run(port=4555, debug=True)
