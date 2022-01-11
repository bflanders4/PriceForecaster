
# Importing libraries
import pandas_datareader as DReader
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import numpy as np


# Function for making our RNN model
def make_rnn(X):
    layer1 = tf.keras.layers.Input(shape=(X.shape[1], 1))
    layer2 = tf.keras.layers.LSTM(units=30)(layer1)
    layer3 = tf.keras.layers.Dense(units=2000)(layer2)
    layer4 = tf.keras.layers.Dense(units=1)(layer3)
    optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.95)
    model = tf.keras.Model(inputs=layer1, outputs=layer4)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    model.summary()
    return model


# Getting the stock of interest
ticker = 'AAPL'
data_source = 'yahoo'
start_date = '2018-01-01'
end_date = '2022-01-01'
df = DReader.DataReader(ticker, data_source=data_source, start=start_date, end=end_date)

# Selecting only the closing price from the data
closing_price = df[['Close']]

# Plotting the closing price vs time for the given timeframe
plt.figure()
plt.title('Closing Price vs Time', fontweight='bold')
plt.xlabel('Date [yyyy-mm-dd]', fontweight='bold')
plt.ylabel('Closing Price [$]', fontweight='bold')
plt.plot(closing_price['Close'])
plt.show()

# Scaling the closing price between -1 and 1
scaler = MinMaxScaler(feature_range=(-1, 1))
closing_price_std = scaler.fit_transform(closing_price.values.reshape(-1, 1))
# Splitting closing_price_std into training and test data
# Since the data is sequential, order of the data is not changed
proportion_training = 0.9
number_days_train = int(np.ceil(len(closing_price_std) * proportion_training))
number_days_test = int(len(closing_price_std) - number_days_train)
closing_price_std_train = closing_price_std[:number_days_train]
closing_price_std_test = closing_price_std[number_days_train:]

# Creating X_train and y_train datasets
# number_prediction_days refers to the number of days in the sequence of days for a given time frame
# Each training example corresponds to number_prediction_days consecutive days within the total amount of
# days in the dataset
number_prediction_days = 30
X_train = []
y_train = []
for x in range(number_prediction_days, number_days_train):
    X_train.append(closing_price_std_train[(x-number_prediction_days):x])
    y_train.append(closing_price_std_train[x])
X_train = np.array(X_train)
y_train = np.array(y_train)
# Repeating this for the test dataset
X_test = []
y_test = []
for x in range(number_prediction_days, number_days_test):
    X_test.append(closing_price_std_test[(x-number_prediction_days):x])
    y_test.append(closing_price_std_test[x])
X_test = np.array(X_test)
y_test = np.array(y_test)

# Making the RNN model
rnn_model = make_rnn(X_train)
# Training the neural network using the training dataset
rnn_model.fit(X_train, y_train, epochs=1000)
# With the neural network trained, predict the price for the test dataset
y_pred = rnn_model.predict(X_test)

# Appending the predicted price onto the test dataset. Repeating this allows the predicted
# value to be appended to the test data, allowing the neural network to
# forecast multiple days instead of one day
X_test_wpred = X_test[:, :, 0]
for x in range(0, number_prediction_days):
    X_test_wpred_nextrow = np.append(X_test_wpred[-1, 1:], y_pred[-1])
    X_test_wpred = np.vstack((X_test_wpred, X_test_wpred_nextrow))
    y_pred = rnn_model.predict(X_test_wpred)
# Performing inverse transform on the scaled data
closing_price_train = scaler.inverse_transform(closing_price_std_train)
closing_price_test = scaler.inverse_transform(closing_price_std_test)
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test)
print('X_train dimensions: ' + str(X_train.shape))
print('X_test dimensions: ' + str(X_test.shape))
print('y_train dimensions: ' + str(y_train.shape))
print('y_test dimensions: ' + str(y_test.shape))
print('y_pred dimensions: ' + str(y_pred.shape))
print('X_test_wpred dimensions: ' + str(X_test_wpred.shape))

# Plotting the training, test and predicted prices
x_train = list(range(0, number_days_train))
x_test = list(range(number_days_train, number_days_train + number_days_test))
x_pred = list(range(number_days_train + number_days_test - len(y_pred) + number_prediction_days, number_days_train
                    + number_days_test + number_prediction_days))
fig, ax = plt.subplots()
ax.plot(x_train, closing_price_train[:], color='red')
ax.plot(x_test, closing_price_test[:], color='black')
ax.plot(x_pred, y_pred[:], color='blue')
plt.show()
