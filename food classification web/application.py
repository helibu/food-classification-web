import os
from flask import Flask, render_template, request
from flask import send_from_directory
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
from skimage.data import imread
import os,cv2
from keras.models import Sequential, Model, load_model
from tqdm import tqdm
import operator

import random
application = app = Flask(__name__)

dir_path = os.path.dirname(os.path.realpath(__file__))
# UPLOAD_FOLDER = dir_path + '/uploads'
# STATIC_FOLDER = dir_path + '/static'
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'

graph = tf.get_default_graph()
with graph.as_default():
    # load model at very first
   food_model = load_model(STATIC_FOLDER + '/' + 'model.h5')


# call model to predict an image
def api(full_path):
    images = []
    images.append(cv2.resize(cv2.imread(full_path), (299,299), interpolation=cv2.INTER_CUBIC))
    images = np.array(images)
    image = images[0]
    image = np.expand_dims(image, axis=0)
    #print(image.shape)
    with graph.as_default():
        predicted = food_model.predict(image)
        return predicted


# home page
@app.route('/')
def home():
   return render_template('index.html')


# procesing uploaded file and predict it
@app.route('/upload', methods=['POST','GET'])
def upload_file():

    if request.method == 'GET':
        return render_template('index.html')
    else:
        file = request.files['image']
        full_name = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(full_name)
        train = pd.read_csv('train.csv')
        To_class = train.food_type.unique()

        result = api(full_name)
        y_num = np.argmax(result, axis=1)

        def num_to_class(index):
            return To_class[index]

        top_5_predict = np.argpartition(result, -5)[:, -5:]
        top_5_chances = np.partition(result, -5)[:, -5:]

        chances = []
        for j in range(5):
            chance = "{:.5f}".format(top_5_chances[0][j]*100)
            chances.append(chance)


        #y_class = num_to_class(y_num)
        y_top5_class = num_to_class(top_5_predict)


        top_5_dict = {y_top5_class[0][j]: top_5_chances[0][j] for j in range(5)}
        sorted_top_5_dict = sorted(top_5_dict.items(), key=operator.itemgetter(1), reverse=True)

        prediction=[]
        accuracy_pre = []
        for i in range(5):
            chance = "{:.5f}".format(sorted_top_5_dict[i][1] * 100)
            accuracy_pre.append(chance)



    return render_template('predict.html', image_file_name = file.filename, prediction = sorted_top_5_dict, accuracy = accuracy_pre)


@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.debug = True
    app.run(debug=True)
    app.debug = True
