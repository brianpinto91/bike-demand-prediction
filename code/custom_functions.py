"""*********************************************************
*Programming language: Python
*Filename: custom_functions.py
*Description: This module consists of useful functions such a splitting data into test-train, evaluating and visualizing model performance, saving and loading models. 
*Author: Brian Pinto
*Version: 1.0
*Date: 03.04.2020
*********************************************************"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error

def contToDiscreteQuantiles(df_col):
    '''
    This method is used to convert a continuous target value to 10 equal quantiles which can be used for stratified split.
    
    Args:
        df_col (DataFrame): A pandas dataframe with single column of target values 
        
    Returns:
        df (DataFrame): A pandas dataframe column with discrete lables according to the quantile of the target value 
    '''
    quantiles = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    labels_ = ['below ' + str(q*10)+' percentile' for q in range(1,11,1)]
    return pd.qcut(df_col, q = quantiles, labels  = labels_).values

def testTrainStratSplit(df, target_col, n_splits = 1, test_size = 0.2, random_state = 1):
    '''
    Splits the input dataframe into train and test dataframes with the defined ratio and random state. The split is performed on the target_col after assigning it into bins which are the quantiles. This type of split ensures that the variance in the train set is equally represented in the test set. Model evaluation on such a test set would give better indication of model generalization compared to a random split.
    
    Args:
        df (pandas dataframe): Dataframe which is to be split.
        
        n_splits (int): Number of splits required. Default = 1
        
        test_size (float): A value between 0 and 1 representing the split size for test set. EX: 0.2 represents 20% test samples. Default = 0.3
        
        random_state: Ensures that the split is consisitent across runs. Default = 1
        
    Returns:
        train_set, test_set (DataFrames): Two dataframes corressponding to train and test sets after Stratified split
    '''
    df['categorized'] = contToDiscreteQuantiles(df[target_col])

    split = StratifiedShuffleSplit( n_splits=n_splits, test_size=test_size, random_state=random_state )
    for train_index, test_index in split.split( df, df["categorized"]):
        strat_train_set = df.loc[train_index]
        strat_test_set = df.loc[test_index]
    return strat_train_set, strat_test_set

def kFoldCV_RMSLE(model, X, y, folds = 10, shuffle_ = False, random_state_ = 1):
    '''
    The function evaluates the passed scikit learn model and prints the RMSLE mean and standard deviation for K folds. 

    Args:
        model (scikit-learn model): Model with fixed parameters whose performance is to be evaluated.
        
        X (DataFrame): A pandas dataframe of the input features.
        
        y (DataFrame): A pandas dataframe with single column of target values.
        
        folds (int): Number of cross validations
        
        shuffle_ (bool): If True the data is shuffled before each cross validation split
            
        random_state: Ensures that the results are consisitent across runs. Default = 1
        
    Returns:
        prints the mean root mean squared log error and its standard deviation for K folds 
    '''
    kf = KFold(n_splits = folds, shuffle = shuffle_, random_state = random_state_)
    rmse = np.sqrt(-cross_val_score(model, X.values, y.values, scoring="neg_mean_squared_error", cv = kf))
    print("Mean RMSLE for {} splits is: {:4f} with a standard deviation of: {:5f}".format(folds, rmse.mean(), rmse.std()))
    
def gridSearch(model, X, y, param_grid_list, cvFolds = 10):
    '''
    Evaluate the model parameter space given by the input grid and print the best parameters among them.
    
    Args:
        model (scikit-learn model): Model whose hyperparameters are to be evaluated.
        
        X (DataFrame): A pandas dataframe of the input features.
        
        y (DataFrame): A pandas dataframe with single column of target values.
        
        param_grid_list (List of dictionaries): Requires a list with dictionaries inside: Syntax: 
        [{'hyperparameter_1_name':[value_1, value_2], 'hyperparameter_2_name':[value_1, value_2]}]
        
        cvFolds (int): Number of cross validations
        
    Returns:
        prints the best hyper-parameters for the model from the given grid space.
    '''
    grid_search = GridSearchCV(model, param_grid_list, cv = cvFolds, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1)
    grid_search.fit(X,y)
    print('Best parameters from grid search are:')
    print(grid_search.best_params_)

def modelEvaluation(model, train_X, train_y_norm, test_X, test_y_norm):
     '''
     Prints the MAE and RMSE in the original scale for both the training and testing set
    
     Args:
         model (scikit-learn model): A trained model
        
         train_X (DataFrame): A pandas dataframe of training features.
         
         train_y_norm (DataFrame): A pandas dataframe column of normalized training target values
         
         test_X (DataFrame): A pandas dataframe of training features.
         
         test_y_norm (DataFrame): A pandas dataframe column of normalized training target values         
         
     Returns:
         Print the MAE and RMSE in the original scale for both the training and testing set
     '''
     train_y = np.expm1(train_y_norm)
     test_y = np.expm1(test_y_norm)
     train_y_hat = np.expm1(model.predict(train_X))
     test_y_hat = np.expm1(model.predict(test_X))
     print('------Training Metrics------')
     print('Number of samples:', len(train_y))
     print('MAE:', mean_absolute_error(train_y,train_y_hat))
     print('RMSE:', np.sqrt(mean_squared_error(train_y,train_y_hat)))
     print('------Testing Metrics-------')
     print('Number of samples:', len(test_y))
     print('MAE:', mean_absolute_error(test_y, test_y_hat))
     print('RMSE:', np.sqrt(mean_squared_error(test_y, test_y_hat))) 
     
def scatterPlot(y, y_hat):
    '''
    Args:
        y (numpy 1D array): Actual values of the target variable
        
        y_hat (numpy 1D array): Predicted values
            
    '''
    fig,ax = plt.subplots(1,1,sharex = False,figsize=(15,8))
    ax.plot(range(0,max(y.max(),y_hat.max()),1),range(0,max(y.max(),y_hat.max()),1),color='red')
    ax.scatter(y,y_hat,alpha=0.5)
    ax.legend(['Target','Predictions'])
    ax.set_title('Actual values vs Model Predictions',y=0.9)
    ax.set_xlabel('Actual counts')
    ax.set_ylabel('Predicted counts')

def saveModel(model, filepath, model_name):
    '''
    Args:
        model (scikit-learn): trained model to be saved
        
        filepath (string): Directory where the model is to be saved
             
    '''
    import os
    import joblib
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    model_full_name = os.path.join(filepath,model_name + '.save')
    joblib.dump(model, model_full_name)

def loadModel(model_path):
    '''
    Args:
        model_path (string): Full filepath and name of the model to be loaded
        
    Returns:    
        model (scikit-learn): feched model from the passed location.
             
    '''
    import os
    import joblib    
    if not os.path.exists(model_path):
        print("No model found")
        return None
    else:
        return joblib.load(model_path)