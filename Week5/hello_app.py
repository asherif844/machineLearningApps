from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route('/')
def basic():
    return 'Flask is running!'


@app.route('/hello', methods=['POST'])
def hello():
    message = request.get_json(force=True)
    name = message['name']
    response = {
        'greeting': 'Hello, ' + name + '!'
    }
    return jsonify(response)


app.run(debug=True)