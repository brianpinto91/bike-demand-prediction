"""*********************************************************
*Programming language: Python
*Filename: online_predictor.py
*Description: This module is used for online predictions using a trained model.
*Author: Brian Pinto
*Version: 1.0
*Date: 03.04.2020
*********************************************************"""
import numpy as np
import pandas as pd

#import from local module
from custom_functions import loadModel

FEATURE_COL = ['yr', 'season', 'mnth', 'hr', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']

#read input features file for predictions
data_file_path = "input/sample.csv"
df = pd.read_csv(data_file_path)

#load the trained model
model_path = "saved_models/hourly_count_predictor.save"
predictor_model = loadModel(model_path) 

#predict (convert back to original scale from the log scale of the model output)
predictions = np.expm1(predictor_model.predict(df[FEATURE_COL]))

#write the predictions to a new column of the inout dataframe
df['count_pred'] = predictions