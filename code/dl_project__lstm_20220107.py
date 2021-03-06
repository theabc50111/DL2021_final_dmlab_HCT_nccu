# -*- coding: utf-8 -*-
"""DL_Project_ LSTM_20220107.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14XyHI8FwAEqVALUECQZ8lqatvguwX0fW
"""

from google.colab import drive
drive.mount('/content/drive')

# https://github.com/ninja3697/Stocks-Price-Prediction-using-Multivariate-Analysis/blob/master/Multivatiate-LSTM/.ipynb_checkpoints/Multivariate-3-LSTM-Copy1-checkpoint.ipynb
# Importing dependencies

import numpy as np
np.random.seed(1)
import tensorflow as tf
tf.random.set_seed(2)
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from keras.layers import Dropout
from sklearn.model_selection import KFold
from math import sqrt
import datetime as dt
plt.style.use('ggplot')

# Commented out IPython magic to ensure Python compatibility.
# %cd '/content/drive/Shareddrives/myShareDrive'

#data = pd.read_csv('./dataset/FS_sp500_Value.csv', parse_dates=['Date'], infer_datetime_format=True)
data = pd.read_csv('./dataset/FS_sp500_Value.csv', parse_dates=['Date'], infer_datetime_format=True, index_col=0)

#data_APA = data.query('Ticker == "APA"').drop(['Unnamed: 0', 'Ticker'], axis=1)
data_APA = data.query('Ticker == "APA"').drop(['Ticker'], axis=1)

# Using set_index() method on 'Name' column
data_APA = data_APA.set_index(data_APA['Date'])

# Correlation matrix
data_APA.corr()['Close']

#data_APA = data_APA.drop(['Volume'], axis=1)

#print(data_APA.describe().Volume)

data_APA.drop(data_APA[data_APA['Volume']==0].index, inplace = True) #Dropping rows with volume value 0

data_APA.shape

# Setting up an early stop
earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=80,  verbose=1, mode='min')
callbacks_list = [earlystop]

#Build and train the model
def fit_model(train,val,timesteps,hl,lr,batch,epochs,loss,activation):
    X_train = []
    Y_train = []
    X_val = []
    Y_val = []
  
    # Loop for training data
    for i in range(timesteps,train.shape[0]):
        X_train.append(train[i-timesteps:i])
        Y_train.append(train[i][0])
    X_train,Y_train = np.array(X_train),np.array(Y_train)
  
    # Loop for val data
    for i in range(timesteps,val.shape[0]):
        X_val.append(val[i-timesteps:i])
        Y_val.append(val[i][0])
    X_val,Y_val = np.array(X_val),np.array(Y_val)
    
    # Adding Layers to the model
    model = Sequential()
    #Adding the first LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50, activation = activation, return_sequences = True, input_shape = (X_train.shape[1],X_train.shape[2])))
    model.add(Dropout(0.2))
    # Adding LSTM layers and some Dropout regularisation
    for i in range(len(hl)-1):        
      model.add(LSTM(hl[i], activation = activation,return_sequences = True))
      #model.add(Dropout(0.2))
    model.add(LSTM(hl[-1],activation = activation))
    model.add(Dropout(0.2))
    # Adding the output layer
    model.add(Dense(units = 1))

    # Compiling the RNN
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate= lr), loss = loss)
    
    # Training the data
    history = model.fit(X_train,Y_train,epochs = epochs,batch_size = batch,validation_data = (X_val, Y_val),verbose = 1,
                        shuffle = False, callbacks=callbacks_list)
    model.reset_states()
    return model, history.history['loss'], history.history['val_loss']

# Evaluating the model
def evaluate_model(model,test,timesteps):
    X_test = []
    Y_test = []

    # Loop for testing data
    for i in range(timesteps,test.shape[0]):
        X_test.append(test[i-timesteps:i])
        Y_test.append(test[i][0])
    X_test,Y_test = np.array(X_test),np.array(Y_test)
    #print(X_test.shape,Y_test.shape)
  
    # Prediction Time !!!!
    Y_hat = model.predict(X_test)
    mse = mean_squared_error(Y_test,Y_hat)
    rmse = sqrt(mse)
    r = r2_score(Y_test,Y_hat)
    return mse, rmse, r, Y_test, Y_hat

# Plotting the predictions
def plot_data(Y_test,Y_hat):
    plt.plot(Y_test,c = 'r')
    plt.plot(Y_hat,c = 'y')
    plt.xlabel('Day')
    plt.ylabel('Price')
    plt.title('Stock Prediction Graph using Multivariate-LSTM model')
    plt.legend(['Actual','Predicted'],loc = 'lower right')
    plt.show()

# Plotting the training errors
def plot_error(train_loss,val_loss):
    plt.plot(train_loss,c = 'r')
    plt.plot(val_loss,c = 'b')
    plt.ylabel('Loss')
    plt.legend(['train','val'],loc = 'upper right')
    plt.show()

# Extracting the series
series = data_APA[['Close','High','Volume']] # Picking the series with high correlation
print(series.shape)
print(series.tail())

# Cross_Train_Val Test Split
cross_train_val_start = dt.date(2010,1,4)
cross_train_val_end = dt.date(2021,6,30)
cross_train_val_data = series.loc[cross_train_val_start:cross_train_val_end]

test_start = dt.date(2021,7,1)
test_end = dt.date(2021,12,31)
test_data = series.loc[test_start:test_end]

print(cross_train_val_data.shape,test_data.shape)

# Normalisation
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))

train = sc.fit_transform(cross_train_val_data)
test = sc.transform(test_data)
print(train.shape,test.shape)

timesteps = [50] 
hl = []
for i in range(40,50,10):
    hl.append([i,i-5])
lr = [1e-3]
batch_size = [64]
num_epochs = [50]
#optimizers = ['SGD', 'RMSprop', 'Adam']
loss = ['mean_squared_error']
activation = ['selu']

results = list()
cross_val_results = list()
n_split = 5
X = train

for t in timesteps:
  for l in hl:
      for rate in lr:
          for batch in batch_size:
              for epochs in num_epochs:
                for loss in loss:
                  for a in activation:
                    
                    train_loss = pd.DataFrame()
                    val_loss = pd.DataFrame()
                    train_loss_value = 0
                    val_loss_value = 0
                
                    for train_index,test_index in KFold(n_split).split(X):
                      #print(train_index,test_index)
                      x_train = X[train_index]
                      x_test = X[test_index]
                      #y_train,y_test=Y[train_index],Y[test_index]
                      model,train_error,val_error = fit_model(x_train,x_test,t,l,rate,batch,epochs,loss,a)
                      train_loss['fold'] = train_error
                      val_loss['fold'] = val_error
                      mse, rmse, r2_value,true,predicted = evaluate_model(model,test,t)
                      #print("Split 1", fold_no)
                      print('MSE = {}'.format(mse))
                      print('RMSE = {}'.format(rmse))
                      print('R-Squared Score = {}'.format(r2_value))
                      
                      plot_data(true,predicted)
                      cross_val_results.append([mse,rmse,r2_value,0])
                      #model,train_error,val_error = fit_model(x_train,x_test,timesteps,hl,lr,batch_size,num_epochs)
                      print('train_error ',train_error)
                      #model,train_loss,val_loss = fit_model(train,val,t,l,rate,batch,epochs)
                      train_loss_value = train_loss_value + train_loss.iloc[-1]['fold']
                      val_loss_value = val_loss_value + val_loss.iloc[-1]['fold']
                
                    results.append([t,l,rate,batch,epochs,loss,a,train_loss_value/n_split,val_loss_value/n_split])
                    print(results)                  

pd.DataFrame(results,columns=['Timestep','Hidden_Layers','Learning_Rate','Batch_Size','epochs','Loss','Activation','Train_Loss','Val_Loss']).to_csv('Multivariate-LSTM_model_Timesteps_0107.csv')
pd.DataFrame(cross_val_results,columns=['mse','rmse','r2_value','0']).to_csv('Multivariate-LSTM_model_Mse_0107.csv')

results

cross_val_results

timesteps = 50
hl = [60,55]
lr = 1e-3
batch = 64
epochs = 100
#optimizers = ['SGD', 'RMSprop', 'Adam']
loss = 'mean_squared_error'
activation = 'selu'

X_train = []
Y_train = []
X_val = []
Y_val = []
  
# Loop for training data
for i in range(timesteps,train.shape[0]):
    X_train.append(train[i-timesteps:i])
    Y_train.append(train[i][0])
X_train,Y_train = np.array(X_train),np.array(Y_train)
  
# Loop for val data
for i in range(timesteps,test.shape[0]):
    X_val.append(test[i-timesteps:i])
    Y_val.append(test[i][0])
X_val,Y_val = np.array(X_val),np.array(Y_val)
    
# Adding Layers to the model
model = Sequential()
#Adding the first LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, activation = activation, return_sequences = True, input_shape = (X_train.shape[1],X_train.shape[2])))
model.add(Dropout(0.2))
# Adding LSTM layers and some Dropout regularisation
for i in range(len(hl)-1):        
  model.add(LSTM(hl[i], activation = activation,return_sequences = True))
  #model.add(Dropout(0.2))
model.add(LSTM(hl[-1],activation = activation))
model.add(Dropout(0.2))
# Adding the output layer
model.add(Dense(units = 1))

# Compiling the RNN
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate= lr), loss = loss)
    
# Training the data
history = model.fit(X_train,Y_train,epochs = epochs,batch_size = batch,validation_data = (X_val, Y_val),verbose = 1,
                        shuffle = False, callbacks=callbacks_list)
model.reset_states()
#model, history.history['loss'], history.history['val_loss']

results = list()
cross_val_results = list()

mse, rmse, r2_value,true,predicted = evaluate_model(model,test,timesteps)

print('MSE = {}'.format(mse))
print('RMSE = {}'.format(rmse))
print('R-Squared Score = {}'.format(r2_value))
                      
plot_data(true,predicted)

import matplotlib.pyplot as pyplot
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

print(history.history['loss'])

print(history.history['val_loss'])

model.save('DL_Project_LSTM_20220107')