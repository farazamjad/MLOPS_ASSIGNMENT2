"""
This is a Flask web application that
 predicts the future closing price of
   a cryptocurrency based on its historical data.
It uses linear regression machine learning model
 trained on the Binance exchange trading data.
"""

import numpy as np
import pandas as pd
import requests
from binance.client import Client
from flask import Flask, request, render_template
from joblib import  load
#pylint: disable=W0012
# pylint: disable=W3101
# pylint: disable=C0116


app = Flask(__name__)

# Load trained model from disk
MODEL_FILENAME = 'btc_model.joblib'
model = load(MODEL_FILENAME)
API_KEY = 'hvkXktWV7JwZYRZRIbodO7ZFoBnuAcCOceosOE0FTufksIvQafO2yPLcL3jdW7oP'
API_SECRET = '2z2tnZicu844s8YBdREGiT7OBRGDOxFFJlxbqOVQDTmO18vrkZLNkKlP9Vyog8PC'

client = Client(API_KEY, API_SECRET)

@app.route('/', methods=['GET', 'POST'])

def index():
    symbol = 'BTCUSDT'
    interval = Client.KLINE_INTERVAL_1DAY
    limit = 7
    klines = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
    data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low',
                                          'close', 'volume', 'close_time', 'quote_asset_volume',
                                            'trades', 'taker_buy_base',
                                              'taker_buy_quote', 'ignored'])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data.set_index('timestamp', inplace=True)
    data['close'] = pd.to_numeric(data['close'])
    response = requests.get('https://api.binance.com/api/v3/ticker/price',
                             params={'symbol': 'BTCUSDT'})
    data1 = response.json()
    current_price = float(data1['price'])

    # Make prediction for next day's BTC price
    last_7_days_data = data['close'].values.reshape(1, -1)
    prediction = model.predict(last_7_days_data)[0]
    prediction = f'{prediction:.2f}'

    return render_template('predict.html',
                            data=data.to_dict(orient='records'),
                              prediction=prediction, current_price=current_price)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    #function for predict route
    if request.method == 'POST':
        # Get last 7 days of BTC price data from form
        last_7_days_data = [float(request.form[f'day_{i}']) for i in range(7)]
        last_7_days_data = np.array(last_7_days_data).reshape(1, -1)

        # Make prediction for next day's BTC price
        prediction = model.predict(last_7_days_data)[0]
        prediction = f'{prediction:.2f}'

        return render_template('result.html', prediction=prediction)

    return render_template('predict.html')
