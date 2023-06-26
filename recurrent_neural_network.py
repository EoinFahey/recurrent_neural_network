# Recurrent neural network
# ========================

# Setup
# -----

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler


# Set working directory
os.chdir('C:/Users/Eoin/Documents/Important/Work/Portfolio/Deep learning/A-Z/Part 8 - Deep Learning/Recurrent neural networks')

# Import training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# Data preprocessing
# ------------------

# Feature scaling
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Create data structure to consider 3 months (60 timesteps) of previous stock data
x_train = []
y_train = []

# 1258 for 5 years of data
for i in range(60, 1258):
    x_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape training data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Building & training RNN
# -----------------------

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout


# Initialize RNN
regressor = Sequential()

# Add first LSTM layer with dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Add second LSTM layer
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Add third LSTM layer
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Add fourth LSTM layer
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Add output layer
regressor.add(Dense(units = 1))

# Compile RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fit RNN to training set
regressor.fit(x_train, y_train, epochs = 100, batch_size = 32)


# Testing RNN
# -----------

# Import real stock price (Jan 2017)
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values


# # Get predicted values. Need to scale since training was scaled too
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:]
inputs = inputs.values.reshape(-1, 1)  # Convert Series to numpy array and reshape
inputs = sc.transform(inputs)
x_test = []
for i in range(60, 80):
    x_test.append(inputs[i - 60:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
predicted_stock_price = regressor.predict(x_test)
# Inverse scaling transformation post-prediction
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualize results
plt.style.use('dark_background')
plt.plot(real_stock_price, color='cyan', label='Real Google stock')
plt.plot(predicted_stock_price, color='red', label='Predicted Google stock')
plt.title('Google stock price prediction', color='white')
plt.xlabel('Time', color='white')
plt.ylabel('Stock price', color='white')

plt.legend()

plt.show()
