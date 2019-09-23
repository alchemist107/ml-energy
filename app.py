from flask import Flask, request
from keras.models import load_model
import pandas as pd
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the model
model = load_model("static/myModel.h5", compile=False)
graph = tf.get_default_graph()

# key values
batteryCapacity = 48*30*0.95  # W.h
powerConsupmtion = 800   # W
boardSize=1*0.67*4
carSpeed=25

def predict(data: str):
    inputData = np.array([float(i) for i in data.split(',')])
    inputData = inputData.reshape(1, 1, 11)
    with graph.as_default():
        prediction = model.predict(inputData)
    return boardSize*prediction[0][0]*60*2


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/predict_energy')
def predict_energy():
    # global graph
    inputString = request.args['inputdata']
    return str(predict(inputString))


@app.route('/predict_charging_time')
def predict_charging_time():
    inputData = request.args['inputdata']
    batteryState = float(request.args['batteryState'])
    return str((1-batteryState)*batteryCapacity / predict(inputData))


@app.route('/predict_distance')
def predict_distance():
    inputData =request.args['inputdata']
    batteryState = float(request.args['batteryState'])
    prediction = predict(inputData)
    consuptionTime = (batteryState*batteryCapacity+prediction)/powerConsupmtion
    return str(consuptionTime*carSpeed)


if __name__ == '__main__':
    app.run(host="0.0.0.0")
