import os
import shutil

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

root_directory = os.path.dirname(os.path.abspath(__file__))


@app.route("/")
def homepage():
    return render_template('landingpage.html')


@app.route("/upload", methods=['POST'])
def upload():
    target = os.path.join(root_directory, 'static/')
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
    return render_template('final.html', img_name=filename, dest=destination)


if __name__ == '__main__':
    app.run(debug=True)
