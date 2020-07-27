_1import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import scipy
import pandas as pd

from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
import numpy
from keras.layers import Dense, Dropout, Flatten
import tensorflow as tf
from keras.models import load_model

from flask import Markup
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

app = Flask(__name__, template_folder = 'index_1.html')
from tensorflow import keras
load_model = keras.models.load_model('')

type(load_model)

@app.route('/')
def home():
    return render_template('index__1.html')

@app.route('/predict',methods=['POST'])
def predict():
    print("You are in Predict")
    int_features = [int(x) for x in request.form.values()]
    values=84.,88.,144.,125., 124., 211., 303., 242., 111., 132.,  70.,53.,  47.,  54.,  61.,  66.,  33.,  40.,  29.,  57.,  57.,  75., 89.,  92.
    scaler = MinMaxScaler(feature_range=(0, 1))
    values = np.reshape(values, (-1, 1))
    values=scaler.fit_transform(values)
    pred =[]
    for i in range(0, int_features[0]):
        #p = load_model.predict(np.asarray([a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x]).astype(np.float32).reshape(1,24,1))
        p = load_model.predict(np.asarray(values).astype(np.float32).reshape(1,24,1))
        pred.append(p)
        values[0]=values[1]
        values[1]=values[2]
        values[2]=values[3]
        values[3]=values[4]
        values[4]=values[5]
        values[5]=values[6]
        values[6]=values[7]
        values[7]=values[8]
        values[8]=values[9]
        values[9]=values[10]
        values[10]=values[11]
        values[11]=values[12]
        values[12]=values[13]
        values[13]=values[14]
        values[14]=values[15]
        values[15]=values[16]
        values[16]=values[17]
        values[17]=values[18]
        values[18]=values[19]
        values[19]=values[20]
        values[20]=values[21]
        values[21]=values[22]
        values[22]=values[23]
        values[23]=p
        
    aa = pd.date_range('2018-04-20 01:00:00', periods=int_features[0], freq='H')
    df = pd.DataFrame(aa, columns = ['Date'])
    
    prediction = []
    for i in range(0, int_features[0]):
         prediction.append(pred[i][0][0])
    
    prediction=np.reshape(prediction, (-1, 1))             
    pred=scaler.inverse_transform(prediction)    
    df["Prediction"] = pred
        
    table = df.to_html(escape = False)
    table = Markup(table)
    #plt.plot(table)
    print("End of def")
    
#<img src={{ name }} name=plot alt="Chart" height="42" width="42">
    return render_template('index__1.html', table  = table)

if __name__ == "__main__":
    app.run(debug = True)
