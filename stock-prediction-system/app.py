import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import ta
import requests
from textblob import TextBlob

# Function to fetch and preprocess stock data
def get_data(stock_symbol, start_date, end_date):
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    data['50_MA'] = data['Close'].rolling(window=50).mean()
    data['200_MA'] = data['Close'].rolling(window=200).mean()
    data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
    data['MACD'] = ta.trend.MACD(data['Close']).macd()
    data['Signal_Line'] = ta.trend.MACD(data['Close']).macd_signal()
    data['Volume_MA'] = data['Volume'].rolling(window=50).mean()
    data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close']).average_true_range()
    data['Bollinger_High'] = ta.volatility.BollingerBands(data['Close']).bollinger_hband()
    data['Bollinger_Low'] = ta.volatility.BollingerBands(data['Close']).bollinger_lband()
    data['Close_Lag1'] = data['Close'].shift(1)
    data['Close_Lag2'] = data['Close'].shift(2)
    data['Daily_Return'] = data['Close'].pct_change()
    data['Volatility'] = data['Daily_Return'].rolling(window=20).std() * np.sqrt(252)
    data = data.dropna()
    return data

# Function to prepare data for the model
def prepare_data(data, lookback=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close', '50_MA', '200_MA', 'RSI', 'MACD', 'Signal_Line', 'Volume_MA', 'ATR', 'Bollinger_High', 'Bollinger_Low', 'Close_Lag1', 'Close_Lag2', 'Volatility']])

    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, X_test, y_train, y_test, scaler

# Function to build the LSTM model
def build_model(input_shape):
    model = Sequential([
        LSTM(units=64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=64, return_sequences=True),
        Dropout(0.2),
        LSTM(units=64, return_sequences=False),
        Dropout(0.2),
        Dense(units=32, activation='relu'),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to predict stock prices
def predict_stock_price(model, data, scaler, lookback=60):
    last_sequence = data[-lookback:].values
    last_sequence = scaler.transform(last_sequence)
    last_sequence = np.reshape(last_sequence, (1, lookback, last_sequence.shape[1]))
    
    prediction = model.predict(last_sequence)
    prediction = scaler.inverse_transform(np.hstack((prediction, np.zeros((prediction.shape[0], scaler.n_features_in_ - 1)))))
    return prediction[0, 0]

# Function to create interactive Plotly chart
def create_interactive_chart(data, predictions):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, 
                        subplot_titles=('Stock Price', 'Volume', 'RSI'), 
                        row_heights=[0.6, 0.2, 0.2])

    # Stock price
    fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['50_MA'], name='50 MA', line=dict(color='orange', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['200_MA'], name='200 MA', line=dict(color='green', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['Bollinger_High'], name='Bollinger High', line=dict(color='red', width=1, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['Bollinger_Low'], name='Bollinger Low', line=dict(color='red', width=1, dash='dash')), row=1, col=1)
    
    if predictions is not None:
        fig.add_trace(go.Scatter(x=predictions.index, y=predictions['Predictions'], name='Predictions', line=dict(color='purple', width=2)), row=1, col=1)

    # Volume
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume'), row=2, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='blue', width=1)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

    fig.update_layout(
        height=1000,
        title_text="Stock Price Analysis",
        xaxis_rangeslider_visible=False,
        hovermode="x unified"
    )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)

    return fig

# Function to display key metrics
def display_metrics(data):
    current_price = data['Close'].iloc[-1]
    previous_close = data['Close'].iloc[-2]
    price_change = current_price - previous_close
    price_change_percent = (price_change / previous_close) * 100

    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Price", f"${current_price:.2f}", f"{price_change:.2f} ({price_change_percent:.2f}%)")
    
    with col2:
        rsi = data['RSI'].iloc[-1]
        st.metric("RSI", f"{rsi:.2f}", 
                  "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral")
    
    with col3:
        macd = data['MACD'].iloc[-1]
        signal = data['Signal_Line'].iloc[-1]
        st.metric("MACD", f"{macd:.2f}", 
                  f"Bullish ({macd - signal:.2f})" if macd > signal else f"Bearish ({macd - signal:.2f})")
    
    with col4:
        volatility = data['Volatility'].iloc[-1]
        st.metric("Volatility", f"{volatility:.2%}", 
                  f"{'High' if volatility > data['Volatility'].mean() else 'Low'} volatility")

# Function to fetch news and perform sentiment analysis
def get_news_sentiment(symbol):
    url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey=YOUR_NEWS_API_KEY"
    response = requests.get(url)
    news_data = response.json()
    
    if news_data["status"] == "ok":
        articles = news_data["articles"][:5]  # Get the top 5 articles
        sentiments = []
        
        for article in articles:
            blob = TextBlob(article["title"] + " " + article["description"])
            sentiment = blob.sentiment.polarity
            sentiments.append({"title": article["title"], "sentiment": sentiment})
        
        return sentiments
    else:
        return None

# Main function for the Streamlit app
def main():
    st.set_page_config(page_title="Advanced Stock Price Predictor", layout="wide")
    
    # Custom CSS
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #1e3c72, #2a5298);
            color: white;
        }
        .stSelectbox > div > div > div {
            background-color: rgba(255, 255, 255, 0.1);
            color: white;
        }
        .stButton > button {
            color: #2a5298;
            background-color: white;
            border-radius: 20px;
        }
        .metric-card {
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
        }
        .chart-container {
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("📈 Advanced Stock Price Prediction using LSTM")
    
    # Sidebar
    st.sidebar.header("Configuration")
    stock_symbol = st.sidebar.selectbox('Select Stock Symbol:', ['AAPL', 'GOOG', 'AMZN', 'MSFT', 'TSLA'])
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=365*2)  # 2 years of data
    date_range = st.sidebar.date_input("Select Date Range", value=(start_date, end_date))
    prediction_days = st.sidebar.slider("Prediction Days", 1, 30, 7)
    
    if st.sidebar.button("Analyze Stock"):
        with st.spinner(f"Analyzing stock data for {stock_symbol}..."):
            # Fetch and preprocess data
            data = get_data(stock_symbol, date_range[0], date_range[1])
            
            # Display key metrics
            display_metrics(data)

            # Prepare data for the model
            X_train, X_test, y_train, y_test, scaler = prepare_data(data)
            
            # Build and train the model
            model = build_model((X_train.shape[1], X_train.shape[2]))
            model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=0)
            
            # Make predictions
            test_predictions = model.predict(X_test)
            test_predictions = scaler.inverse_transform(np.hstack((test_predictions, np.zeros((test_predictions.shape[0], scaler.n_features_in_ - 1)))))[:, 0]
            
            # Future predictions
            future_predictions = []
            last_sequence = data[['Close', '50_MA', '200_MA', 'RSI', 'MACD', 'Signal_Line', 'Volume_MA', 'ATR', 'Bollinger_High', 'Bollinger_Low', 'Close_Lag1', 'Close_Lag2', 'Volatility']].values[-60:]
            
            for _ in range(prediction_days):
                next_pred = predict_stock_price(model, last_sequence, scaler)
                future_predictions.append(next_pred)
                last_sequence = np.roll(last_sequence, -1, axis=0)
                last_sequence[-1] = [next_pred] + [np.nan] * (last_sequence.shape[1] - 1)  # Filling other features with NaN as we don't predict them
            
            future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=prediction_days)
            future_df = pd.DataFrame(index=future_dates, data={'Predictions': future_predictions})

            # Create interactive chart
            fig = create_interactive_chart(data, future_df)
            st.plotly_chart(fig, use_container_width=True)

            # Display future predictions
            st.subheader(f"Future {prediction_days}-Day Price Prediction")
            st.dataframe(future_df.style.highlight_max(axis=0))

            # Model performance metrics
            mse = np.mean((y_test - test_predictions)**2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y_test - test_predictions))
            
            st.subheader("Model Performance Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Mean Squared Error (MSE)", f"{mse:.4f}")
            col2.metric("Root Mean Squared Error (RMSE)", f"{rmse:.4f}")
            col3.metric("Mean Absolute Error (MAE)", f"{mae:.4f}")

            # News sentiment analysis
            st.subheader("Recent News Sentiment")
            sentiments = get_news_sentiment(stock_symbol)
            if sentiments:
                for article in sentiments:
                    sentiment_color = "green" if article["sentiment"] > 0 else "red" if article["sentiment"] < 0 else "yellow"
                    st.markdown(f"<span style='color:{sentiment_color};'>{'🔺' if article['sentimentmarkdown(f"<span style='color:{sentiment_color};'>{'🔺' if article['sentiment'] > 0 else '🔻' if article['sentiment'] < 0 else '➖'} {article['title']}</span>", unsafe_allow_html=True)
                st.markdown(f"Sentiment Score: {article['sentiment']:.2f}")
                st.markdown("---")

    # Additional Information
    with st.expander("About this app"):
        st.write("""
        This advanced stock price prediction app uses Long Short-Term Memory (LSTM) neural networks to forecast stock prices. 
        It considers various technical indicators such as Moving Averages, RSI, MACD, Bollinger Bands, Volatility, and ATR for more accurate predictions. 
        The app provides interactive charts, key metrics, future price predictions, and news sentiment analysis to assist in your stock analysis.
        
        Features:
        - Historical stock data visualization with candlestick chart
        - Technical indicators overlay (50 MA, 200 MA, Bollinger Bands)
        - Key metrics display (RSI, MACD, Volatility)
        - LSTM-based price prediction for user-specified number of days
        - Model performance metrics
        - News sentiment analysis
        - Interactive charts with multiple views (Price, Volume, RSI)
        
        Please note that this app is for educational purposes only and should not be used as financial advice. Always conduct your own research and consult with a financial advisor before making investment decisions.
        """)

if __name__ == '__main__':
    main()

# Note: Remember to replace 'YOUR_NEWS_API_KEY' with an actual API key from newsapi.org

print("Enhanced Stock Price Predictor script execution completed.")
