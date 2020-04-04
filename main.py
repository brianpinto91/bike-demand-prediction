"""*********************************************************
*Programming language: Python
*Filename: trainer.py
*Description: This module is used to train a random forest regression model to predict the hourly bike demand.
*Author: Brian Pinto
*Version: 1.0
*Date: 03.04.2020
*********************************************************"""
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor

#import local module which has useful functions
import custom_functions

RANDOM_STATE = 42 #Keep the training results deterministic during runs

# Read the data
# source: https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset
data_file_path = 'data/hour.csv'
df = pd.read_csv(data_file_path)
       
FEATURE_COL = ['yr', 'season', 'mnth', 'hr', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']


train_set, test_set = custom_functions.testTrainStratSplit(df, target_col = 'cnt', n_splits = 1, test_size = 0.2, random_state = RANDOM_STATE)
X_train = train_set[FEATURE_COL]
y_train = train_set['cnt']
y_train_norm = np.log1p(y_train) #Normalize the target variable which is skewed
X_test  = test_set[FEATURE_COL]
y_test = test_set['cnt']
y_test_norm = np.log(y_test) 

#Grid Search for hyperparameter selection through cross validation
GRID_SEARCH = True #Takes time, only enalbe when required
if GRID_SEARCH == True:
    param_grid = [{'n_estimators': [25, 50, 100, 200], 'max_features': [4,6,8,10], 'max_depth':[25,50,75], 'bootstrap':[True,False]}]
    custom_functions.gridSearch(model = RandomForestRegressor(), X = X_train, y = y_train_norm, param_grid_list = param_grid, cvFolds = 10)

#Fit the model with the best parameters obatained from grid search
predictor = RandomForestRegressor(n_estimators = 200, max_features = 8, max_depth = 50, bootstrap = True, random_state=RANDOM_STATE)

#Cross validate the model and print the result
custom_functions.kFoldCV_RMSLE(predictor, X_train, y_train_norm, folds = 10, shuffle_ = True, random_state_ = RANDOM_STATE)

#fit the model
predictor.fit(X_train, y_train_norm)

#Evaluation and visualization of model performance on test set
custom_functions.modelEvaluation(predictor, X_train, y_train_norm, X_test, y_test_norm)
custom_functions.scatterPlot(y_test, np.expm1(predictor.predict(X_test)))

#Save the model
#custom_functions.saveModel(predictor, "saved_models", "hourly_count_predictor")