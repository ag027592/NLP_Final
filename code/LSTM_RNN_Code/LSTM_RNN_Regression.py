# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 11:54:01 2017

@author: David
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.contrib import learn as tflearn
from tensorflow.contrib import layers as tflayers
from tensorflow.contrib import learn
from sklearn.metrics import mean_squared_error
import glob
from scipy.stats import spearmanr




import timeit

start = timeit.default_timer()

def rnn_data(data, time_steps, labels=False):
    """
    creates new data frame based on previous observation
      * example:
        l = [1, 2, 3, 4, 5]
        time_steps = 2
        -> labels == False [[1, 2], [2, 3], [3, 4]]
        -> labels == True [3, 4, 5]
    """
    rnn_df = []
    for i in range(len(data) - time_steps):
        if labels:
            try:
                rnn_df.append(data.iloc[i + time_steps].as_matrix())
            except AttributeError:
                rnn_df.append(data.iloc[i + time_steps])
        else:
            data_ = data.iloc[i: i + time_steps].as_matrix()
            rnn_df.append(data_ if len(data_.shape) > 1 else [[i] for i in data_])

    return np.array(rnn_df, dtype=np.float64)


def split_data(data, val_size=0.1, test_size=0.1):
    """
    splits data to training, validation and testing parts
    """
    ntest = int(round(len(data) * (1 - test_size)))
    nval = int(round(len(data.iloc[:ntest]) * (1 - val_size)))

    df_train, df_val, df_test = data.iloc[:nval], data.iloc[nval:ntest], data.iloc[ntest:]

    return df_train, df_val, df_test


def prepare_data(data, time_steps, labels=False, val_size=0.1, test_size=0.1):
    """
    Given the number of `time_steps` and some data,
    prepares training, validation and test data for an lstm cell.
    """
    df_train, df_val, df_test = split_data(data, val_size, test_size)
    return (rnn_data(df_train, time_steps, labels=labels),
            rnn_data(df_val, time_steps, labels=labels),
            rnn_data(df_test, time_steps, labels=labels))


def load_csvdata(rawdata, time_steps, seperate=False):
    data = rawdata
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    train_x, val_x, test_x = prepare_data(data['a'] if seperate else data, time_steps)
    train_y, val_y, test_y = prepare_data(data['b'] if seperate else data, time_steps, labels=True)
    return dict(train=train_x, val=val_x, test=test_x), dict(train=train_y, val=val_y, test=test_y)


def generate_data(fct, x, time_steps, seperate=False):
    """generates data with based on a function fct"""
    data = fct(x)
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    train_x, val_x, test_x = prepare_data(data['a'] if seperate else data, time_steps)
    train_y, val_y, test_y = prepare_data(data['b'] if seperate else data, time_steps, labels=True)
    return dict(train=train_x, val=val_x, test=test_x), dict(train=train_y, val=val_y, test=test_y)


def lstm_model(num_units, rnn_layers, dense_layers=None, learning_rate=0.001, optimizer='Adagrad'):
    """
    Creates a deep model based on:
        * stacked lstm cells
        * an optional dense layers
    :param num_units: the size of the cells.
    :param rnn_layers: list of int or dict
                         * list of int: the steps used to instantiate the `BasicLSTMCell` cell
                         * list of dict: [{steps: int, keep_prob: int}, ...]
    :param dense_layers: list of nodes for each layer
    :return: the model definition
    """
    def lstm_cells(layers):
        if isinstance(layers[0], dict):
            return [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(layer['num_units'],
                                                                               state_is_tuple=True),
                                                  layer['keep_prob'])
                    if layer.get('keep_prob') else tf.contrib.rnn.BasicLSTMCell(layer['num_units'],
                                                                                state_is_tuple=True)
                    for layer in layers]
        return [tf.contrib.rnn.BasicLSTMCell(steps, state_is_tuple=True) for steps in layers]
    
    
    def dnn_layers(input_layers, layers):
        if layers and isinstance(layers, dict):
            return tflayers.stack(input_layers, tflayers.fully_connected,
                                  layers['layers'],
                                  activation=layers.get('activation'),
                                  dropout=layers.get('dropout'))
        elif layers:
            return tflayers.stack(input_layers, tflayers.fully_connected, layers)
        else:
            return input_layers

    def _lstm_model(X, y):
        stacked_lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells(rnn_layers), state_is_tuple=True)
        x_ = tf.unstack(X, axis=1, num=num_units)
        output, layers =tf.contrib.rnn.static_rnn(stacked_lstm, x_, dtype=dtypes.float64)
        output = dnn_layers(output[-1], dense_layers)
        prediction, loss = tflearn.models.linear_regression(output, y)
        train_op = tf.contrib.layers.optimize_loss(
            loss, tf.contrib.framework.get_global_step(), optimizer=optimizer,
            learning_rate=learning_rate)
        return prediction, loss, train_op

    return _lstm_model

#==============================================================================
# 
#==============================================================================
LOG_DIR = './ops_logs/lstm_NNIME_300_Arousal'
TIMESTEPS = 1
RNN_LAYERS = [{'num_units': 300}]
DENSE_LAYERS = [10, 10]
TRAINING_STEPS = 3000
BATCH_SIZE = 50
PRINT_STEPS = TRAINING_STEPS / 100

'''Load The Data'''

Team_list=["01","02","03","04","05","06","07","08","09","10","11","12","13","14"
           ,"15","16","17","18","19","20","21","22"]

Name_List = glob.glob1('./Data/total_jieba/','*txt')

Index_list=[]

for Name in Name_List:
    Team=Name.split("_")
    Index_list.append(int(Team[0]))
    
Index_array=np.array(Index_list).reshape(3754,1)    

Test_index=[]

for team in Team_list:
    Test_index.append(np.where(Index_array==int(team))[0].tolist())

Arousal=np.array(pd.read_csv('./Data/Arousal.csv'))
Valence=np.array(pd.read_csv('./Data/Valence.csv'))
Feature=np.array(pd.read_csv('./Data/Feature.csv'))[:,0:302].reshape((3754,1,302))

Index = np.arange(3754)
Index = set(Index)



Total_predict=[]
Total_Ture=[]
Team_Correlation=[]


for Test in Test_index:
    Train_idx = Index-set(Test)
    X = Feature[list(Train_idx)]
    y = Arousal[list(Train_idx)]
#    y = Valence[list(Train_idx)]
    
    
    
    X_test=Feature[Test]
    y_test=Arousal[Test]
#    y_test=Valence[Test]


    '''Run The Model and Fit Predictions'''
    regressor = learn.Estimator(model_fn=lstm_model(TIMESTEPS, RNN_LAYERS, DENSE_LAYERS),
                               model_dir=LOG_DIR)


    regressor.fit(X, y,batch_size=BATCH_SIZE,steps=TRAINING_STEPS)
    predicted = regressor.predict(X_test)
    predicted = np.array(list(predicted))
    #not used in this example but used for seeing deviations
    rmse = np.sqrt(((predicted - y_test) ** 2).mean(axis=0))
    
    score = mean_squared_error(predicted, y_test)
#    print ("MSE: %f" % score)
    correlation=spearmanr(predicted,y_test)
    Total_predict.extend(predicted)
    Total_Ture.extend(y_test)
#    print ("Correlation: %f" % correlation[0])
    Team_Correlation.append(correlation[0])
    
correlation=spearmanr(Total_predict,Total_Ture)
print ("Correlation: %f" % correlation[0])

stop = timeit.default_timer()

print (stop - start)    



