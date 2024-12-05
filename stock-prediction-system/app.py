import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import plotly.graph_objs as go
import plotly.express as px

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
    num_features = scaled_data.shape[1]
    padding = np.zeros((predictions.shape[0], num_features - 1))
    predictions_full = np.hstack((predictions, padding))

    predictions = scaler.inverse_transform(predictions_full)[:, 0]

    return predictions

# Function to generate stock insights
def generate_insights(data, predictions):
    # Calculate performance metrics
    last_close = data['Close'].iloc[-1]
    first_close = data['Close'].iloc[0]
    total_return = ((last_close - first_close) / first_close) * 100
    
    # Volatility calculation
    volatility = data['Close'].pct_change().std() * np.sqrt(252)
    
    # Recent trend
    recent_trend = 'Bullish' if data['Close'].iloc[-1] > data['Close'].iloc[-30] else 'Bearish'
    
    # Prediction insights
    predicted_last = predictions[-1]
    prediction_direction = 'Positive' if predicted_last > last_close else 'Negative'
    
    insights = {
        'Total Return': f'{total_return:.2f}%',
        'Annualized Volatility': f'{volatility:.2f}',
        'Recent Trend': recent_trend,
        'Prediction Sentiment': prediction_direction,
        'Current Price': f'${last_close:.2f}',
        'Predicted Next Price': f'${predicted_last:.2f}'
    }
    
    return insights

# Main Streamlit App
def main():
    # Custom CSS for enhanced styling
    st.set_page_config(page_title="StockSage: Advanced Price Predictor", layout="wide")
    
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Inter', sans-serif;
    }
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        color: #2c3e50;
        margin-bottom: 30px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .stock-card {
        background: white;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    .stock-card:hover {
        transform: scale(1.02);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main Title
    st.markdown('<div class="main-header">StockSage: AI-Powered Price Predictor</div>', unsafe_allow_html=True)
    
    # Stock Selection
    col1, col2 = st.columns([2,1])
    
    with col1:
        stock_options = ['AAPL', 'GOOG', 'AMZN', 'MSFT', 'TSLA', 'NFLX', 'META']
        stock_symbol = st.selectbox('Select Stock Symbol:', stock_options)
    
    with col2:
        prediction_period = st.selectbox('Prediction Horizon:', 
                                         ['Short-term (30 days)', 
                                          'Medium-term (90 days)', 
                                          'Long-term (180 days)'])
    
    if stock_symbol:
        # Data Processing
        with st.spinner('Fetching data and training model...'):
            data = get_data(stock_symbol)
            scaled_data, X_train, y_train, training_data_len, scaler = prepare_data(data)
            
            model = build_model(X_train)
            model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
            
            predictions = predict_stock_price(model, scaled_data, training_data_len, scaler)
        
        # Generate Insights
        insights = generate_insights(data, predictions)
        
        # Create columns for layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="stock-card">', unsafe_allow_html=True)
            st.subheader('Stock Performance Insights')
            for key, value in insights.items():
                st.metric(key, value)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            # Technical Indicators Visualization
            fig_tech = go.Figure()
            fig_tech.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price', line=dict(color='blue')))
            fig_tech.add_trace(go.Scatter(x=data.index, y=data['50_MA'], mode='lines', name='50-Day MA', line=dict(color='red', dash='dot')))
            fig_tech.add_trace(go.Scatter(x=data.index, y=data['200_MA'], mode='lines', name='200-Day MA', line=dict(color='green', dash='dot')))
            
            fig_tech.update_layout(
                title=f'{stock_symbol} Technical Indicators',
                xaxis_title='Date',
                yaxis_title='Price',
                template='plotly_white'
            )
            
            st.plotly_chart(fig_tech, use_container_width=True)
        
        # Prediction Visualization
        st.markdown('<div class="stock-card">', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
        ax.plot(data.index[training_data_len:], data['Close'].iloc[training_data_len:], label='Actual Price', color='blue')
        ax.plot(data.index[training_data_len:], predictions, label='Predicted Price', color='red', linestyle='--')
        
        ax.set_title(f'{stock_symbol} Stock Price Prediction', fontsize=15)
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Detailed Prediction Analysis
        st.markdown('<div class="stock-card">', unsafe_allow_html=True)
        st.subheader('Prediction Analysis')
        
        # Confidence and Error Metrics
        pred_df = pd.DataFrame({
            'Actual': data['Close'].iloc[training_data_len:],
            'Predicted': predictions
        })
        pred_df['Absolute Error'] = abs(pred_df['Actual'] - pred_df['Predicted'])
        pred_df['Percentage Error'] = (pred_df['Absolute Error'] / pred_df['Actual']) * 100
        
        st.write("Prediction Accuracy Metrics:")
        st.dataframe(pred_df.describe())
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
