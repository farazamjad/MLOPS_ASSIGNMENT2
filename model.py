
from binance.client import Client
from binance_data import get_klines
from joblib import dump, load
from sklearn.linear_model import LinearRegression

import numpy as np
import pandas as pd
api_key = 'hvkXktWV7JwZYRZRIbodO7ZFoBnuAcCOceosOE0FTufksIvQafO2yPLcL3jdW7oP'
api_secret = '2z2tnZicu844s8YBdREGiT7OBRGDOxFFJlxbqOVQDTmO18vrkZLNkKlP9Vyog8PC'
client = Client(api_key, api_secret)

# account_info = client.get_account()
# print(account_info)
# print(api_key, api_secret)
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