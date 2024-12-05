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
            background: linear-gradient(135deg, #1e3c72, #2a5298);
            font-family: 'Arial', sans-serif;
        }
        .header {
            text-align: center;
            color: white;
            font-size: 40px;
            margin-top: 20px;
        }
        .card {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 20px;
            margin-top: 20px;
            box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.2);
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="header">📈 Stock Price Prediction using LSTM</div>', unsafe_allow_html=True)
    
    stock_symbol = st.selectbox('Select Stock Symbol:', ['AAPL', 'GOOG', 'AMZN', 'MSFT', 'TSLA'])
    
    if stock_symbol:
        st.info(f"Analyzing stock data for {stock_symbol}...")
        data = get_data(stock_symbol)
        scaled_data, X_train, y_train, training_data_len, scaler = prepare_data(data)
        
        model = build_model(X_train)
        model.fit(X_train, y_train, epochs=10, batch_size=32)
        predictions = predict_stock_price(model, scaled_data, training_data_len, scaler)
        
        # Create graphs
        train = data[:training_data_len]
        test = data[training_data_len:]
        test['Predictions'] = predictions

        st.markdown("<h3 style='text-align: center; color: white;'>Performance Overview</h3>", unsafe_allow_html=True)
        
        # Plot
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.plot(train['Close'], label='Training Data', color='blue')
        ax.plot(test['Close'], label='Actual Price', color='green')
        ax.plot(test['Predictions'], label='Predicted Price', color='red', linestyle='--')
        
        ax.set_facecolor('white')
        ax.set_title(f"{stock_symbol} Stock Price", fontsize=16)
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()

        st.pyplot(fig)
        
        st.markdown(f"**Current RSI:** {data['RSI'].iloc[-1]:.2f}")
        st.markdown(f"**Current MACD:** {data['MACD'].iloc[-1]:.2f}")

if __name__ == '__main__':
    main()
