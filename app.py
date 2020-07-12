# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 14:29:36 2020

@author: RADHIKA
"""

from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect


import pandas as pd
import datetime as dt
import pickle
from flask import Markup
import keras
from tensorflow.keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform


app = Flask(__name__)  
with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        model = load_model('C:\\Users\\abc\\Desktop\\AirPollution_Vinod\\LSTM_Model.h5')
       
 
@app.route("/")
def home():    
    return render_template('DAQ(1).html')

@app.route("/", methods=['GET', 'POST'])
def predict():    
    if request.method == 'POST':
        # Get the input from post request
        datevalue=request.form['ddate']
        timestamps=pd.date_range(start='2018-04-20 1:00:00',end=datevalue,freq='1H')
        tsc=len(timestamps)
        print(tsc)       
        prediction=model.predict(tsc)
        output=list(prediction)  
        print(output)
        df=pd.DataFrame(output,columns = ['Prediction'])
        df.set_index(keys=timestamps)
        df["Date"]=timestamps
        table=df.to_html(escape=False)
        table=Markup(table)
        print("End of def")        
        
        return render_template("DAQ(1).html",prediction=table)

if __name__ == '__main__':  
   app.run(debug = False)  