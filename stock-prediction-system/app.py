import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf
from datetime import datetime, timedelta
import ta
import logging
from typing import Tuple, Dict, Optional
import time
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CACHE_TTL = 3600

class DataCache:
    def __init__(self):
        self.cache = {}
        self.lock = threading.Lock()
        
    def get(self, key: str) -> Optional[pd.DataFrame]:
        with self.lock:
            if key in self.cache:
                data, timestamp = self.cache[key]
                if time.time() - timestamp < CACHE_TTL:
                    return data
                else:
                    del self.cache[key]
            return None
    
    def set(self, key: str, data: pd.DataFrame):
        with self.lock:
            self.cache[key] = (data, time.time())

data_cache = DataCache()

class StockDataError(Exception):
    pass

class ModelError(Exception):
    pass

def create_custom_theme():
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            color: #e2e2e2;
        }
        .metric-container {
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
            margin: 10px 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease;
        }
        .metric-container:hover {
            transform: translateY(-5px);
        }
        .chart-container {
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 25px;
            margin-top: 30px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        }
        .custom-button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border-radius: 10px;
            border: none;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .custom-button:hover {
            background-color: #45a049;
            transform: scale(1.05);
        }
        .sidebar .sidebar-content {
            background-color: rgba(26, 26, 46, 0.95);
        }
        .stProgress > div > div > div > div {
            background-color: #4CAF50;
        }
        </style>
    """, unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def get_stock_data(stock_symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    try:
        cache_key = f"{stock_symbol}_{start_date}_{end_date}"
        cached_data = data_cache.get(cache_key)
        
        if cached_data is not None:
            return cached_data
        
        data = yf.download(stock_symbol, start=start_date, end=end_date)
        
        if data.empty:
            raise StockDataError(f"No data available for {stock_symbol}")

        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_columns):
            raise StockDataError(f"Missing required columns for {stock_symbol}")

        if len(data) < 200:
            raise StockDataError(f"Insufficient data points for {stock_symbol}")

        data = data.dropna()
        if data.empty:
            raise StockDataError(f"No valid data points after removing NaN values for {stock_symbol}")
        
        data['50_MA'] = data['Close'].rolling(window=50).mean()
        data['200_MA'] = data['Close'].rolling(window=200).mean()
        
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = exp1 - exp2
        data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
        
        data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
        data['ATR'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'])
        
        data['Bollinger_High'] = data['Close'].rolling(window=20).mean() + (data['Close'].rolling(window=20).std() * 2)
        data['Bollinger_Low'] = data['Close'].rolling(window=20).mean() - (data['Close'].rolling(window=20).std() * 2)
        
        data['Close_Lag1'] = data['Close'].shift(1)
        data['Close_Lag2'] = data['Close'].shift(2)
        data['Daily_Return'] = data['Close'].pct_change()
        data['Volatility'] = data['Daily_Return'].rolling(window=20).std() * np.sqrt(252)
        
        data = data.dropna()
        data_cache.set(cache_key, data)
        
        return data
    
    except Exception as e:
        logger.error(f"Error fetching stock data: {str(e)}")
        raise StockDataError(f"Failed to fetch data for {stock_symbol}: {str(e)}")

def create_enhanced_chart(data: pd.DataFrame, predictions: Optional[pd.DataFrame] = None) -> go.Figure:
    fig = make_subplots(
        rows=4, 
        cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Price Action', 'Volume Analysis', 'Technical Indicators', 'Momentum'),
        row_heights=[0.4, 0.2, 0.2, 0.2]
    )

    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price',
            increasing_line_color='#26A69A',
            decreasing_line_color='#EF5350'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['50_MA'],
            name='50 MA',
            line=dict(color='#FFD700', width=1)
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['200_MA'],
            name='200 MA',
            line=dict(color='#FF69B4', width=1)
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['Bollinger_High'],
            name='Bollinger High',
            line=dict(color='rgba(173, 216, 230, 0.5)', width=1, dash='dash')
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['Bollinger_Low'],
            name='Bollinger Low',
            line=dict(color='rgba(173, 216, 230, 0.5)', width=1, dash='dash'),
            fill='tonexty'
        ),
        row=1, col=1
    )

    colors = ['red' if close < open else 'green' 
              for close, open in zip(data['Close'], data['Open'])]
    
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['Volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.7
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['RSI'],
            name='RSI',
            line=dict(color='#9C27B0', width=1)
        ),
        row=3, col=1
    )

    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['MACD'],
            name='MACD',
            line=dict(color='#2196F3', width=1)
        ),
        row=4, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['Signal_Line'],
            name='Signal Line',
            line=dict(color='#FF9800', width=1)
        ),
        row=4, col=1
    )

    if predictions is not None:
        fig.add_trace(
            go.Scatter(
                x=predictions.index,
                y=predictions['Predictions'],
                name='Predictions',
                line=dict(color='#00FF00', width=2, dash='dash'),
                mode='lines+markers'
            ),
            row=1, col=1
        )

    fig.update_layout(
        height=1200,
        template='plotly_dark',
        title={
            'text': "Advanced Technical Analysis Dashboard",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24)
        },
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(0,0,0,0.5)"
        ),
        margin=dict(t=130, l=60, r=60, b=60)
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')

    return fig

def display_enhanced_metrics(data: pd.DataFrame):
    with st.container():
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        current_price = data['Close'].iloc[-1]
        previous_close = data['Close'].iloc[-2]
        price_change = current_price - previous_close
        price_change_percent = (price_change / previous_close) * 100
        
        with col1:
            st.metric(
                "Current Price",
                f"${current_price:.2f}",
                f"{price_change:+.2f} ({price_change_percent:+.2f}%)",
                delta_color="normal"
            )
        
        with col2:
            rsi = data['RSI'].iloc[-1]
            rsi_status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
            st.metric(
                "RSI",
                f"{rsi:.2f}",
                rsi_status,
                delta_color="off" if rsi_status == "Neutral" else "inverse"
            )
        
        with col3:
            macd = data['MACD'].iloc[-1]
            signal = data['Signal_Line'].iloc[-1]
            macd_diff = macd - signal
            st.metric(
                "MACD",
                f"{macd:.2f}",
                f"Signal: {signal:.2f} (Diff: {macd_diff:+.2f})",
                delta_color="normal" if macd_diff > 0 else "inverse"
            )
        
        with col4:
            volatility = data['Volatility'].iloc[-1]
            avg_volatility = data['Volatility'].mean()
            vol_status = "High" if volatility > avg_volatility else "Low"
            st.metric(
                "Volatility",
                f"{volatility:.2%}",
                f"{vol_status} ({(volatility/avg_volatility - 1):+.1%} vs avg)",
                delta_color="inverse" if vol_status == "High" else "normal"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)

def prepare_lstm_data(data: pd.DataFrame, lookback: int = 60) -> Tuple:
    try:
        features = [
            'Close', '50_MA', '200_MA', 'RSI', 'MACD', 'Signal_Line',
            'Volume_MA', 'ATR', 'Bollinger_High', 'Bollinger_Low',
            'Close_Lag1', 'Close_Lag2', 'Volatility'
        ]
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data[features])
        
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        return X_train, X_test, y_train, y_test, scaler
    
    except Exception as e:
        logger.error(f"Error in data preparation: {str(e)}")
        raise ModelError(f"Failed to prepare data: {str(e)}")

def build_enhanced_lstm_model(input_shape: Tuple[int, int]) -> Sequential:
    try:
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(96, return_sequences=True),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    except Exception as e:
        logger.error(f"Error in model building: {str(e)}")
        raise ModelError(f"Failed to build model: {str(e)}")

def predict_future_prices(
    model: Sequential,
    data: pd.DataFrame,
    scaler: MinMaxScaler,
    prediction_days: int,
    lookback: int = 60
) -> np.ndarray:
    try:
        last_sequence = data[['Close', '50_MA', '200_MA', 'RSI', 'MACD', 
                            'Signal_Line', 'Volume_MA', 'ATR', 'Bollinger_High',
                            'Bollinger_Low', 'Close_Lag1', 'Close_Lag2', 
                            'Volatility']].values[-lookback:]
        
        future_predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(prediction_days):
            scaled_sequence = scaler.transform(current_sequence)
            sequence_3d = np.reshape(scaled_sequence, (1, lookback, scaled_sequence.shape[1]))
            next_pred = model.predict(sequence_3d, verbose=0)
            next_pred_full = np.zeros((1, scaler.n_features_in_))
            next_pred_full[0, 0] = next_pred[0, 0]
            next_pred_price = scaler.inverse_transform(next_pred_full)[0, 0]
            
            future_predictions.append(next_pred_price)
            
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1] = [next_pred_price] + [np.nan] * (current_sequence.shape[1] - 1)
        
        return np.array(future_predictions)
    
    except Exception as e:
        logger.error(f"Error in future price prediction: {str(e)}")
        raise ModelError(f"Failed to predict future prices: {str(e)}")

def display_model_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    st.subheader("Model Performance Metrics")
    
    with st.container():
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        with col1:
            st.metric("MSE", f"{mse:.4f}")
        with col2:
            st.metric("RMSE", f"{rmse:.4f}")
        with col3:
            st.metric("MAE", f"{mae:.4f}")
        with col4:
            st.metric("MAPE", f"{mape:.2f}%")
        
        st.markdown('</div>', unsafe_allow_html=True)

def generate_trading_signals(data: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    signals = {
        'technical': {},
        'trend': {}
    }
    
    rsi = data['RSI'].iloc[-1]
    if rsi > 70:
        signals['technical']['RSI'] = 'Overbought'
    elif rsi < 30:
        signals['technical']['RSI'] = 'Oversold'
    else:
        signals['technical']['RSI'] = 'Neutral'
    
    macd = data['MACD'].iloc[-1]
    signal = data['Signal_Line'].iloc[-1]
    if macd > signal:
        signals['technical']['MACD'] = 'Bullish'
    else:
        signals['technical']['MACD'] = 'Bearish'
    
    current_price = data['Close'].iloc[-1]
    ma_50 = data['50_MA'].iloc[-1]
    ma_200 = data['200_MA'].iloc[-1]
    
    if current_price > ma_50 > ma_200:
        signals['trend']['Moving Averages'] = 'Strong Uptrend'
    elif current_price > ma_50 and ma_50 < ma_200:
        signals['trend']['Moving Averages'] = 'Potential Uptrend'
    elif current_price < ma_50 < ma_200:
        signals['trend']['Moving Averages'] = 'Strong Downtrend'
    elif current_price < ma_50 and ma_50 > ma_200:
        signals['trend']['Moving Averages'] = 'Potential Downtrend'
    else:
        signals['trend']['Moving Averages'] = 'Neutral'
    
    upper_bb = data['Bollinger_High'].iloc[-1]
    lower_bb = data['Bollinger_Low'].iloc[-1]
    
    if current_price > upper_bb:
        signals['technical']['Bollinger Bands'] = 'Overbought'
    elif current_price < lower_bb:
        signals['technical']['Bollinger Bands'] = 'Oversold'
    else:
        signals['technical']['Bollinger Bands'] = 'Within Bands'
    
    return signals

def display_trading_signals(data: pd.DataFrame):
    st.subheader("Trading Signals Analysis")
    
    with st.container():
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        
        signals = generate_trading_signals(data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Technical Signals")
            for signal, value in signals['technical'].items():
                st.markdown(f"**{signal}:** {value}")
        
        with col2:
            st.markdown("### Trend Analysis")
            for signal, value in signals['trend'].items():
                st.markdown(f"**{signal}:** {value}")
        
        st.markdown('</div>', unsafe_allow_html=True)

def calculate_risk_metrics(data: pd.DataFrame) -> Dict[str, float]:
    returns = data['Close'].pct_change().dropna()
    
    risk_metrics = {
        'var': np.percentile(returns, 5) * 100,
        'sharpe': (returns.mean() / returns.std()) * np.sqrt(252),
        'max_drawdown': ((data['Close'].cummax() - data['Close']) / data['Close'].cummax()).max() * 100
    }
    
    return risk_metrics

def display_risk_analysis(data: pd.DataFrame):
    st.subheader("Risk Analysis")
    
    with st.container():
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        
        risk_metrics = calculate_risk_metrics(data)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Value at Risk (95%)", f"{risk_metrics['var']:.2f}%")
        with col2:
            st.metric("Sharpe Ratio", f"{risk_metrics['sharpe']:.2f}")
        with col3:
            st.metric("Max Drawdown", f"{risk_metrics['max_drawdown']:.2f}%")
        
        st.markdown('</div>', unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="Advanced Stock Analysis Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    create_custom_theme()
    
    st.title("📈 Advanced Stock Analysis & Prediction Dashboard")
    st.markdown("---")
    
    with st.sidebar:
        st.header("Analysis Configuration")
        
        stock_symbol = st.selectbox(
            'Select Stock Symbol:',
            ['AAPL', 'GOOG', 'AMZN', 'MSFT', 'TSLA', 'META', 'NVDA'],
            help="Choose the stock symbol to analyze"
        )
        
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=365*2)
        
        date_range = st.date_input(
            "Select Date Range",
            value=(start_date, end_date),
            help="Choose the date range for analysis"
        )
        
        prediction_days = st.slider(
            "Prediction Horizon (Days)",
            1, 30, 7,
            help="Number of days to forecast"
        )
        
        analysis_button = st.button(
            "Analyze Stock",
            help="Click to start the analysis",
            use_container_width=True
        )
    
    if analysis_button:
        try:
            with st.spinner("Fetching and analyzing stock data..."):
                data = get_stock_data(stock_symbol, date_range[0], date_range[1])
                
                display_enhanced_metrics(data)
                
                with st.spinner("Training prediction model..."):
                    try:
                        X_train, X_test, y_train, y_test, scaler = prepare_lstm_data(data)
                        
                        model = build_enhanced_lstm_model(X_train.shape[1:])
                        
                        progress_bar = st.progress(0)
                        epochs = 50
                        
                        class TrainingCallback(tf.keras.callbacks.Callback):
                            def on_epoch_end(self, epoch, logs=None):
                                progress = (epoch + 1) / epochs
                                progress_bar.progress(progress)
                        
                        history = model.fit(
                            X_train, y_train,
                            epochs=epochs,
                            batch_size=32,
                            validation_split=0.1,
                            callbacks=[TrainingCallback()],
                            verbose=0
                        )
                        
                        test_predictions = model.predict(X_test)
                        future_predictions = predict_future_prices(
                            model, data, scaler, prediction_days
                        )
                        
                        future_dates = pd.date_range(
                            start=data.index[-1] + pd.Timedelta(days=1),
                            periods=prediction_days
                        )
                        predictions_df = pd.DataFrame(
                            index=future_dates,
                            data={'Predictions': future_predictions}
                        )
                        
                        st.subheader("Technical Analysis & Predictions")
                        fig = create_enhanced_chart(data, predictions_df)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        display_model_metrics(y_test, test_predictions)
                        
                        display_trading_signals(data)
                        
                        display_risk_analysis(data)
                        
                    except Exception as e:
                        st.error(f"Error in model training: {str(e)}")
                        logger.error(f"Model training error: {str(e)}", exc_info=True)
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            logger.error(f"Application error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
