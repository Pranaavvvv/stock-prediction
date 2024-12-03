import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import time

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

    predictions = model.predict(X_test)
    num_features = scaled_data.shape[1]
    padding = np.zeros((predictions.shape[0], num_features - 1))
    predictions_full = np.hstack((predictions, padding))

    predictions = scaler.inverse_transform(predictions_full)[:, 0]
    return predictions

# Main Application
def main():
    # Custom CSS
    st.markdown("""
    <style>
    body { background: linear-gradient(135deg, #1a2b3c, #273542); color: white; font-family: 'Roboto', sans-serif; }
    .header-container { text-align: center; background: linear-gradient(90deg, #4b6cb7, #182848); padding: 20px; border-radius: 15px; }
    .header-container h1 { color: white; font-size: 3rem; margin-bottom: 5px; }
    .header-container p { font-size: 1.2rem; color: #ddd; }
    .card { background-color: rgba(255, 255, 255, 0.05); border-radius: 10px; padding: 15px; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div class="header-container">
        <h1>🚀 Stock Price Predictor</h1>
        <p>Analyze & Predict Market Trends with AI-Powered Insights</p>
    </div>
    """, unsafe_allow_html=True)

    # Input
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    stock_symbol = st.text_input("🔍 Enter Stock Symbol", "AAPL", help="Enter a valid stock ticker (e.g., AAPL, MSFT, TSLA)")
    st.markdown("</div>", unsafe_allow_html=True)

    if stock_symbol:
        with st.spinner("Fetching and processing data..."):
            time.sleep(2)  # Simulating loading

        try:
            data = get_data(stock_symbol)
            scaled_data, X_train, y_train, training_data_len, scaler = prepare_data(data)

            model = build_model(X_train)
            model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

            predictions = predict_stock_price(model, scaled_data, training_data_len, scaler)

            train = data[:training_data_len]
            test = data[training_data_len:]
            test['Predictions'] = predictions

            # Plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=train.index, y=train['Close'], name='Training Data'))
            fig.add_trace(go.Scatter(x=test.index, y=test['Close'], name='Actual Data'))
            fig.add_trace(go.Scatter(x=test.index, y=test['Predictions'], name='Predictions'))
            fig.update_layout(title=f"{stock_symbol} Stock Price Prediction", template="plotly_dark")
            st.plotly_chart(fig)

            # Metrics
            mse = np.mean((test['Close'] - test['Predictions'])**2)
            mae = np.mean(np.abs(test['Close'] - test['Predictions']))
            st.metric("📉 Mean Squared Error", f"{mse:.2f}")
            st.metric("📊 Mean Absolute Error", f"{mae:.2f}")

        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
