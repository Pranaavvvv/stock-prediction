import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Function to fetch and preprocess stock data
def get_data(stock_symbol):
    data = yf.download(stock_symbol, start='2015-01-01', end='2024-01-01')
    data['50_MA'] = data['Close'].rolling(window=50).mean()
    data['200_MA'] = data['Close'].rolling(window=200).mean()

    def compute_RSI(data, window=14):
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    data['RSI'] = compute_RSI(data)
    data['12_EMA'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['26_EMA'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['12_EMA'] - data['26_EMA']
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['Volume_MA'] = data['Volume'].rolling(window=50).mean()
    data['Close_Lag1'] = data['Close'].shift(1)
    data['Close_Lag2'] = data['Close'].shift(2)
    data = data.dropna()
    return data

# Function to prepare data for the model
def prepare_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close', '50_MA', '200_MA', 'RSI', 'MACD', 'Signal_Line', 'Volume_MA', 'Close_Lag1', 'Close_Lag2']])

    training_data_len = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:training_data_len]

    X_train = []
    y_train = []

    for i in range(60, len(train_data)):
        X_train.append(train_data[i-60:i, :])
        y_train.append(train_data[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
    
    return scaled_data, X_train, y_train, training_data_len, scaler

# Function to build the LSTM model
def build_model(X_train):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to predict stock prices
def predict_stock_price(model, scaled_data, training_data_len, scaler):
    test_data = scaled_data[training_data_len - 60:]
    X_test = []

    for i in range(60, len(test_data)):
        X_test.append(test_data[i-60:i, :])

    X_test = np.array(X_test)

    if len(X_test.shape) == 3:
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

    predictions = model.predict(X_test)
    
    # Ensure predictions match the scaler's input dimensions
    num_features = scaled_data.shape[1]  # Number of features in scaled_data
    padding = np.zeros((predictions.shape[0], num_features - 1))
    predictions_full = np.hstack((predictions, padding))

    predictions = scaler.inverse_transform(predictions_full)[:, 0]  # Only keep the predicted Close prices

    return predictions

# Main function for the Streamlit app
def main():
    st.set_page_config(page_title="Stock Price Predictor", layout="wide")
    
    st.markdown("""
        <style>
        body {
            background: linear-gradient(135deg, #ff5f6d, #ffc3a0);
            font-family: 'Arial', sans-serif;
        }
        .header {
            text-align: center;
            color: white;
            font-size: 40px;
            margin-top: 50px;
        }
        .input-box {
            display: flex;
            justify-content: center;
            margin-top: 30px;
        }
        .stock-card {
            display: flex;
            justify-content: center;
            flex-direction: column;
            align-items: center;
            margin-top: 30px;
            border-radius: 15px;
            padding: 20px;
            background: rgba(255, 255, 255, 0.1);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        }
        .stock-card h4 {
            color: #fff;
            font-size: 24px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="header">Stock Price Prediction using LSTM</div>', unsafe_allow_html=True)

    # Dropdown for selecting stock symbol
    stock_options = ['AAPL', 'GOOG', 'AMZN', 'MSFT', 'TSLA', 'NFLX', 'META']
    stock_symbol = st.selectbox('Select Stock Symbol:', stock_options)
    
    if stock_symbol:
        st.markdown("""
            <div class="stock-card">
            <h4>Processing Stock Data...</h4>
            <p>Fetching historical data and training the model...</p>
            </div>
        """, unsafe_allow_html=True)

        progress = st.progress(0)
        for i in range(100):
            progress.progress(i + 1)

        data = get_data(stock_symbol)
        scaled_data, X_train, y_train, training_data_len, scaler = prepare_data(data)

        model = build_model(X_train)
        model.fit(X_train, y_train, epochs=10, batch_size=32)

        predictions = predict_stock_price(model, scaled_data, training_data_len, scaler)

        train = data[:training_data_len]
        test = data[training_data_len:]
        test['Predictions'] = predictions

        fig, ax = plt.subplots(figsize=(14, 8))
        ax.plot(train['Close'], label='Training Data', color='blue', alpha=0.6)
        ax.plot(test['Close'], label='Test Data', color='green', alpha=0.6)
        ax.plot(test['Predictions'], label='Predicted Data', color='red', linestyle='--', alpha=0.8)

        ax.set_title(f'{stock_symbol} Stock Price Prediction', fontsize=20, color='white')
        ax.set_xlabel('Date', fontsize=14, color='white')
        ax.set_ylabel('Price', fontsize=14, color='white')
        ax.legend()

        ax.set_facecolor('#2c3e50')
        fig.patch.set_facecolor('#2c3e50')

        st.pyplot(fig)
    else:
        st.markdown('<div class="stock-card"><h4>Stock Symbol Required</h4><p>Please select a stock symbol to predict.</p></div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
