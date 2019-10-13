from flask import Flask, jsonify, request
app = Flask(__name__)
# define a predict function as an endpoint 
@app.route("/predict", methods=["GET","POST"])
def predict():
    data = {"success": False}
    # get the request parameters
    params = request.json
    if (params == None):
        params = request.args
    # if parameters are found, echo the msg parameter 
    if (params != None):
        data["response"] = params.get("msg")
        data["success"] = True
        data['fuckery'] = True
    # return a response in json format 
    return jsonify(data)
# start the flask app, allow remote connections
app.run(host='0.0.0.0')