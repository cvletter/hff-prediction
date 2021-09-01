import pandas as pd
import numpy as np
import tensorflow as tf
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

data_raw = pd.read_csv('lstm_test.csv', sep=";", decimal=",")
data_raw.set_index('week', inplace=True)

data = pd.DataFrame(data_raw)
data.sort_index(inplace=True, ascending=True)
data = data.values
data = data.astype('float32')

dataset = data

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size


train, test = dataset.iloc[0:train_size, :], dataset.iloc[train_size:len(dataset), :]

trainX_scaler = MinMaxScaler(feature_range=(0, 1))
testX_scaler = MinMaxScaler(feature_range=(0, 1))
trainY_scaler = MinMaxScaler(feature_range=(0, 1))
testY_scaler = MinMaxScaler(feature_range=(0, 1))


trainY, trainX = pd.DataFrame(train.iloc[:, 0]), train.iloc[:, 1:]
testY, testX = pd.DataFrame(test.iloc[:, 0]), test.iloc[:, 1:]

trainY, trainX = trainY_scaler.fit_transform(trainY), trainX_scaler.fit_transform(trainX)
testY, testX = testY_scaler.fit_transform(testY), testX_scaler.fit_transform(testX)



# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

model = Sequential()
model.add(LSTM(30, input_shape=(1, 4)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

inv_y = np.concatenate((trainPredict, trainX[:, 1:]), axis=1)

pd.concat([pd.DataFrame(trainPredict), pd.DataFrame(trainX)], axis=1)

trainPredict = trainY_scaler.inverse_transform(trainPredict)
trainY = trainY_scaler.inverse_transform(trainY)
testPredict = testY_scaler.inverse_transform(testPredict)
testY = testY_scaler.inverse_transform(testY)

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[:, 0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[:, 0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
