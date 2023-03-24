import os
from binance.client import Client
from binance_data import get_klines
from joblib import dump, load
# from sklearn.linear_model import LinearRegression


# from sklearn.preprocessing import MinMaxScaler
# from keras.models import Sequential
# from keras.layers import Dense, LSTM

# def preprocess_data(klines):
#     # Convert klines to numpy array
#     klines = np.array(klines)

#     # Extract features and target variable
#     X = klines[:, 1:6].astype(float)  # Open, High, Low, Close, Volume
#     y = klines[:, 6].astype(float)  # Quote asset volume

#     # Normalize data
#     X = (X - X.mean(axis=0)) / X.std(axis=0)

#     return X, y

# def train_model(X, y):
#     model = LinearRegression()
#     model.fit(X, y)
#     return model

# klines = get_klines(symbol='BTCUSDT', interval='1m')
# X, y = preprocess_data(klines)
# model = train_model(X, y)
# # Save model to file
# dump(model, 'model.joblib')

# # Load model from file
# # Load the model from file
# model = load('model.joblib')

# # Preprocess new data in the same way as the training data
# new_klines = get_klines(symbol='BTCUSDT', interval='1m')
# X_test, _ = preprocess_data(new_klines)

# # Use the model to generate predictions
# y_pred = model.predict(X_test.reshape(-1, 5))

# # Print the predicted values
# print(y_pred)
# import os
# from binance.client import Client
# from binance_data import get_klines
# from joblib import dump, load
from sklearn.linear_model import LinearRegression

import numpy as np
import pandas as pd
# from sklearn.preprocessing import MinMaxScaler


# def preprocess_data(klines):
#     # Convert klines to numpy array
#     klines = np.array(klines)

#     # Extract features and target variable
#     X = klines[:, 1:6].astype(float)  # Open, High, Low, Close, Volume
#     y = klines[:, 6].astype(float)  # Quote asset volume

#     # Normalize data
#     X = (X - X.mean(axis=0)) / X.std(axis=0)

#     return X, y

# def train_model(X, y):
#     model = LinearRegression()
#     model.fit(X, y)
#     return model

# def predict_price(btc_value):
#     # Load the model from file
#     model = load('model.joblib')

#     # Preprocess new data in the same way as the training data
#     klines = get_klines(symbol='BTCUSDT', interval='1m')
#     X, _ = preprocess_data(klines)

#     # Use the model to generate predictions
#     last_kline = np.array([btc_value, btc_value, btc_value, btc_value, 0])
#     X_test = np.vstack([X[-99:], last_kline])
#     y_pred = model.predict(X_test.reshape(-1, 5))

#     # Return the predicted value
#     return y_pred[0]
api_key = 'hvkXktWV7JwZYRZRIbodO7ZFoBnuAcCOceosOE0FTufksIvQafO2yPLcL3jdW7oP'
api_secret = '2z2tnZicu844s8YBdREGiT7OBRGDOxFFJlxbqOVQDTmO18vrkZLNkKlP9Vyog8PC'
client = Client(api_key, api_secret)

account_info = client.get_account()
print(account_info)
print(api_key, api_secret)
symbol = 'BTCUSDT'
interval = Client.KLINE_INTERVAL_1DAY
limit = 500
klines = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignored'])
data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
data.set_index('timestamp', inplace=True)
data['close'] = pd.to_numeric(data['close'])

# Create dataset for training model
window_size = 7
data['target'] = data['close'].shift(-1)
data.dropna(inplace=True)
X = np.array([data['close'].iloc[i:i+window_size].values for i in range(len(data) - window_size)])
y = data.iloc[window_size:]['target'].values

# Split data into training and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Train model on training data
model = LinearRegression()
model.fit(X_train.tolist(), y_train)

# Evaluate model on test data
score = model.score(X_test.tolist(), y_test)
print(f'Model R^2 score: {score:.2f}')

# Make prediction for next day's BTC price
last_7_days_data = data['close'].tail(window_size).values.reshape(1, -1)
print(f'Last 7 days BTC prices: {last_7_days_data[0]}')

prediction = model.predict(last_7_days_data)[0]
print(f'Predicted next day BTC price: {prediction:.2f}')

model_filename = 'btc_model.joblib'
dump(model, model_filename)