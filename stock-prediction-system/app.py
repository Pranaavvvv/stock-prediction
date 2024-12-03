import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go
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

def main():
    # Custom CSS for advanced styling
    st.markdown("""
    <style>
    /* Global Styles */
    body {
        background-color: #0E1117;
        color: #FFFFFF;
        font-family: 'Inter', sans-serif;
    }

    /* Custom Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #1a2b3c, #273542);
    }

    /* Enhanced Header Styling */
    .header-container {
        background: linear-gradient(90deg, #4b6cb7, #182848);
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
        margin-bottom: 30px;
        text-align: center;
    }

    .header-container h1 {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }

    .header-container p {
        color: #e0e0e0;
        font-size: 1.2rem;
    }

    /* Input Styling */
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
        border: 2px solid #4b6cb7;
        border-radius: 10px;
        padding: 10px;
        font-size: 1rem;
    }

    /* Card Styling */
    .card {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    }

    /* Progress Bar Styling */
    .stProgress > div > div > div {
        background-color: #4b6cb7;
    }
    </style>
    """, unsafe_allow_html=True)

    # Enhanced Header
    st.markdown("""
    <div class="header-container">
        <h1>🚀 Stock Price Predictor</h1>
        <p>Leveraging Machine Learning to Forecast Stock Trends</p>
    </div>
    """, unsafe_allow_html=True)

    # Stock Symbol Input with Enhanced Styling
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    stock_symbol = st.text_input('Enter Stock Symbol', 'AAPL', 
                                 help='Enter a valid stock symbol (e.g., AAPL, GOOGL, MSFT)')
    st.markdown("</div>", unsafe_allow_html=True)

    if stock_symbol:
        # Processing Notification
        st.markdown("""
        <div class="card">
        <h3>🔍 Processing Stock Data</h3>
        <p>Fetching historical data, training LSTM model...</p>
        </div>
        """, unsafe_allow_html=True)

        # Progress Indicator
        progress_bar = st.progress(0)
        for percent_complete in range(100):
            progress_bar.progress(percent_complete + 1)

        # Data Processing and Prediction
        try:
            data = get_data(stock_symbol)
            scaled_data, X_train, y_train, training_data_len, scaler = prepare_data(data)

            model = build_model(X_train)
            model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

            predictions = predict_stock_price(model, scaled_data, training_data_len, scaler)

            train = data[:training_data_len]
            test = data[training_data_len:]
            test['Predictions'] = predictions

            # Interactive Plotly Chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=train.index, y=train['Close'], mode='lines', 
                                      name='Training Data', line=dict(color='#1f77b4', width=2)))
            fig.add_trace(go.Scatter(x=test.index, y=test['Close'], mode='lines', 
                                      name='Actual Test Data', line=dict(color='#2ca02c', width=2)))
            fig.add_trace(go.Scatter(x=test.index, y=test['Predictions'], mode='lines', 
                                      name='Predicted Prices', line=dict(color='#ff7f0e', dash='dot', width=2)))

            fig.update_layout(
                title=f'{stock_symbol} Stock Price Prediction',
                xaxis_title='Date',
                yaxis_title='Price',
                template='plotly_dark',
                height=600,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )

            st.plotly_chart(fig, use_container_width=True)

            # Additional Insights
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("🔮 Prediction Insights")
            
            # Calculate prediction accuracy
            mse = np.mean((test['Close'] - test['Predictions'])**2)
            mae = np.mean(np.abs(test['Close'] - test['Predictions']))
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Mean Squared Error", f"{mse:.2f}")
            with col2:
                st.metric("Mean Absolute Error", f"{mae:.2f}")
            
            st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error processing stock data: {e}")

if __name__ == '__main__':
    main()
