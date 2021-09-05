import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

data_raw = pd.read_csv('lstm_test.csv', sep=";", decimal=",")
data_raw.set_index('week', inplace=True)

data_train = data_raw.iloc[1:95, :]
trainY, trainX = data_train.iloc[:, 0], data_train.iloc[:, 1:]
data_test = data_raw.iloc[0, :]
testY, testX = data_test['y'], data_test.drop('y')
y = trainY
X = trainX
train_data = y.join(X, how='left')


def prepare_data(y, X):
    y = pd.DataFrame(y)
    y.sort_index(ascending=True, inplace=True)
    X.sort_index(ascending=True, inplace=True)

    y_scaler = y.mean(), y.std()
    y_scaled = (y - y_scaler[0]) / y_scaler[1]

    X_scaler = X.mean(), X.std()
    X_scaled = (X - X_scaler[0]) / X_scaler[1]

    feature_scaling = {'y_scaler' : y_scaler, 'X_scaler' : X_scaler}

    return y_scaled.values, X_scaled.values, feature_scaling


def neural_network_fit(y, X, hidden_layer=20):
    mdl = Sequential()

    mdl.add(Dense(hidden_layer, input_dim=X.shape[1], kernel_initializer='normal', activation='relu'))
    mdl.add(Dropout(0.2))
    mdl.add(Dense(hidden_layer, activation='relu'))
    mdl.add(Dense(1, activation='linear'))

    mdl.compile(loss='mse', optimizer='adam', metrics='mse')
    mdl.fit(X, y, epochs=100, batch_size=10, verbose=1, validation_split=0.2)

    return mdl


def neural_nework_predict(predict_data, scalers, fitted_model):
    X_scaled = (predict_data - scalers['X_scaler'][0]) / scalers['X_scaler'][1]
    X_scaled = X_scaled.values
    X_scaled = X_scaled.reshape(1, X_scaled.shape[0])
    prediction = fitted_model.predict(X_scaled)

    prediction_ds = (prediction * scalers['y_scaler'][1][0]) + scalers['y_scaler'][0][0]

    return prediction_ds[0][0]
