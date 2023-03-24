"""
Model which trains on live data!
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from joblib import dump

from binance.client import Client

API_KEY = 'hvkXktWV7JwZYRZRIbodO7ZFoBnuAcCOceosOE0FTufksIvQafO2yPLcL3jdW7oP'
API_SECRET = '2z2tnZicu844s8YBdREGiT7OBRGDOxFFJlxbqOVQDTmO18vrkZLNkKlP9Vyog8PC'

client = Client(API_KEY, API_SECRET)
SYMBOL = 'BTCUSDT'
INTERVAL = Client.KLINE_INTERVAL_1DAY
LIMIT = 500
klines = client.futures_klines(symbol=SYMBOL, interval=INTERVAL, limit=LIMIT)

data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                      'close_time', 'quote_asset_volume', 'trades',
                                        'taker_buy_base', 'taker_buy_quote', 'ignored'])
data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
data.set_index('timestamp', inplace=True)
data['close'] = pd.to_numeric(data['close'])

# Create dataset for training model
WINDOW_SIZE = 7
data['target'] = data['close'].shift(-1)
data.dropna(inplace=True)
X = np.array([data['close'].iloc[i:i+WINDOW_SIZE].values for i in range(len(data) - WINDOW_SIZE)])
y = data.iloc[WINDOW_SIZE:]['target'].values

# Split data into training and test sets
TRAIN_SIZE = int(len(X) * 0.8)
X_train, X_test = X[:TRAIN_SIZE], X[TRAIN_SIZE:]
y_train, y_test = y[:TRAIN_SIZE], y[TRAIN_SIZE:]

# Train model on training data
model = LinearRegression()
model.fit(X_train.tolist(), y_train)

# Evaluate model on test data
score = model.score(X_test.tolist(), y_test)
print(f'Model R^2 score: {score:.2f}')

# Make prediction for next day's BTC price
LAST_7_DAYS_DATA = data['close'].tail(WINDOW_SIZE).values.reshape(1, -1)
print(f'Last 7 days BTC prices: {LAST_7_DAYS_DATA[0]}')

prediction = model.predict(LAST_7_DAYS_DATA)[0]
print(f'Predicted next day BTC price: {prediction:.2f}')

MODEL_FILENAME = 'btc_model.joblib'
dump(model, MODEL_FILENAME)
