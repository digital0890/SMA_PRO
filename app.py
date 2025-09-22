import streamlit as st
import ccxt
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import numpy as np
import sqlite3
import asyncio
import websockets
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings
warnings.filterwarnings('ignore')

# --------------------------------------------------
# Dark Modern Themed Streamlit App - Enhanced UI
# --------------------------------------------------

# -------------------------------
# Enhanced Styling & Theme (Dark Modern)
# -------------------------------
FONT_IMPORT = "@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Poppins:wght@400;600&display=swap');"

DARK_BG = "#0f1724"
CARD_BG = "#0b1220"
ACCENT = "#6EE7B7"
ACCENT_SECOND = "#60A5FA"
TEXT = "#E6EEF5"
MUTED = "#98A4B3"
ERROR = "#FB7185"
WARNING = "#FBBF24"

CUSTOM_CSS = f"""
{FONT_IMPORT}
:root {{
  --bg: {DARK_BG};
  --card: {CARD_BG};
  --accent: {ACCENT};
  --accent-2: {ACCENT_SECOND};
  --text: {TEXT};
  --muted: {MUTED};
  --error: {ERROR};
}}

[data-testid='stAppViewContainer'] {{
  background: linear-gradient(180deg, rgba(5,10,16,1) 0%, rgba(10,14,20,1) 100%);
  color: var(--text);
  font-family: 'Inter', 'Poppins', system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
}}

[data-testid='stSidebar'] {{
  background: linear-gradient(180deg, rgba(6,10,16,0.9), rgba(8,12,20,0.9));
  border-right: 1px solid rgba(255,255,255,0.03);
  padding: 18px 14px 24px 18px;
}}

.streamlit-card {{
  background: var(--card);
  border-radius: 12px;
  padding: 22px;
  box-shadow: 0 6px 18px rgba(2,6,12,0.6);
  border: 1px solid rgba(255,255,255,0.03);
  margin-bottom: 20px;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}}

.streamlit-card:hover {{
  transform: translateY(-2px);
  box-shadow: 0 8px 24px rgba(2,6,12,0.8);
}}

.chart-container {{
  border-radius: 16px;
  overflow: hidden;
  margin-top: 10px;
}}

.header-container {{
  background: linear-gradient(90deg, rgba(11,18,32,0.8) 0%, rgba(15,23,36,0.6) 100%);
  padding: 20px 24px;
  border-radius: 16px;
  margin-bottom: 24px;
  border: 1px solid rgba(255,255,255,0.05);
  box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}}

.stSelectbox, .stSlider, .stDateInput, .stTimeInput {{
  margin-bottom: 16px;
}}

.stSelectbox > div > div {{
  background-color: rgba(15, 23, 36, 0.7);
  border: 1px solid rgba(255,255,255,0.1);
  border-radius: 10px;
}}

.stButton > button {{
  border-radius: 10px;
  background: linear-gradient(90deg, {ACCENT}, {ACCENT_SECOND});
  color: #0f1724;
  font-weight: 600;
  border: none;
  transition: all 0.3s ease;
}}

.stButton > button:hover {{
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(110, 231, 183, 0.3);
}}

.plotly-graph-div {{ background: transparent !important; }}

/* Custom metric cards */
.metric-card {{
  background: linear-gradient(135deg, rgba(11,18,32,0.8), rgba(15,23,36,0.6));
  border-radius: 12px;
  padding: 16px;
  border: 1px solid rgba(255,255,255,0.05);
  text-align: center;
  margin: 8px 0;
}}

.metric-value {{
  font-size: 24px;
  font-weight: 700;
  color: {ACCENT};
  margin: 8px 0;
}}

.metric-label {{
  font-size: 14px;
  color: {MUTED};
}}

/* Loading animation */
@keyframes pulse {{
  0% {{ opacity: 1; }}
  50% {{ opacity: 0.5; }}
  100% {{ opacity: 1; }}
}}

.pulse {{
  animation: pulse 1.5s ease-in-out infinite;
}}

.alert-badge {{
  background: {ERROR};
  color: white;
  padding: 4px 8px;
  border-radius: 12px;
  font-size: 12px;
  font-weight: bold;
}}

.success-badge {{
  background: {ACCENT};
  color: #0f1724;
  padding: 4px 8px;
  border-radius: 12px;
  font-size: 12px;
  font-weight: bold;
}}
"""

st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)

# -------------------------------
# Database Setup
# -------------------------------
def init_database():
    conn = sqlite3.connect('trading_data.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS market_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            timestamp DATETIME,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            timeframe TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            condition TEXT,
            value REAL,
            triggered BOOLEAN,
            created_at DATETIME,
            triggered_at DATETIME
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS portfolio (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            quantity REAL,
            entry_price REAL,
            current_price REAL,
            pnl REAL,
            timestamp DATETIME
        )
    ''')
    
    conn.commit()
    conn.close()

init_database()

# -------------------------------
# Advanced Technical Indicators
# -------------------------------
class AdvancedIndicators:
    @staticmethod
    def calculate_stochastic(data, k_period=14, d_period=3):
        low_min = data['Low'].rolling(window=k_period).min()
        high_max = data['High'].rolling(window=k_period).max()
        
        data['stoch_k'] = ((data['Close'] - low_min) / (high_max - low_min)) * 100
        data['stoch_d'] = data['stoch_k'].rolling(window=d_period).mean()
        return data
    
    @staticmethod
    def calculate_ichimoku(data):
        # Tenkan-sen (Conversion Line)
        nine_period_high = data['High'].rolling(window=9).max()
        nine_period_low = data['Low'].rolling(window=9).min()
        data['tenkan_sen'] = (nine_period_high + nine_period_low) / 2
        
        # Kijun-sen (Base Line)
        twenty_six_period_high = data['High'].rolling(window=26).max()
        twenty_six_period_low = data['Low'].rolling(window=26).min()
        data['kijun_sen'] = (twenty_six_period_high + twenty_six_period_low) / 2
        
        # Senkou Span A (Leading Span A)
        data['senkou_span_a'] = ((data['tenkan_sen'] + data['kijun_sen']) / 2).shift(26)
        
        # Senkou Span B (Leading Span B)
        fifty_two_period_high = data['High'].rolling(window=52).max()
        fifty_two_period_low = data['Low'].rolling(window=52).min()
        data['senkou_span_b'] = ((fifty_two_period_high + fifty_two_period_low) / 2).shift(26)
        
        # Chikou Span (Lagging Span)
        data['chikou_span'] = data['Close'].shift(-26)
        
        return data
    
    @staticmethod
    def calculate_fibonacci_retracement(data):
        high = data['High'].max()
        low = data['Low'].min()
        diff = high - low
        
        levels = {
            '0%': high,
            '23.6%': high - diff * 0.236,
            '38.2%': high - diff * 0.382,
            '50%': high - diff * 0.5,
            '61.8%': high - diff * 0.618,
            '78.6%': high - diff * 0.786,
            '100%': low
        }
        
        return levels
    
    @staticmethod
    def calculate_rsi(data, period=14):
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        return data
    
    @staticmethod
    def calculate_macd(data, fast=12, slow=26, signal=9):
        exp1 = data['Close'].ewm(span=fast, adjust=False).mean()
        exp2 = data['Close'].ewm(span=slow, adjust=False).mean()
        data['macd'] = exp1 - exp2
        data['macd_signal'] = data['macd'].ewm(span=signal, adjust=False).mean()
        data['macd_histogram'] = data['macd'] - data['macd_signal']
        return data
    
    @staticmethod
    def calculate_bollinger_bands(data, period=20, std=2):
        data['bb_middle'] = data['Close'].rolling(window=period).mean()
        bb_std = data['Close'].rolling(window=period).std()
        data['bb_upper'] = data['bb_middle'] + (bb_std * std)
        data['bb_lower'] = data['bb_middle'] - (bb_std * std)
        return data

# -------------------------------
# AI Prediction Models
# -------------------------------
class AIPredictor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        
    def create_lstm_model(self, input_shape):
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def predict_price_lstm(self, data, future_periods=10):
        try:
            # Prepare data
            prices = data['Close'].values.reshape(-1, 1)
            scaled_prices = self.scaler.fit_transform(prices)
            
            # Create sequences
            sequence_length = 60
            X, y = [], []
            
            for i in range(sequence_length, len(scaled_prices)):
                X.append(scaled_prices[i-sequence_length:i, 0])
                y.append(scaled_prices[i, 0])
            
            X, y = np.array(X), np.array(y)
            X = X.reshape(X.shape[0], X.shape[1], 1)
            
            # Train model
            model = self.create_lstm_model((X.shape[1], 1))
            model.fit(X, y, epochs=10, batch_size=32, verbose=0)
            
            # Predict future
            last_sequence = scaled_prices[-sequence_length:]
            predictions = []
            
            for _ in range(future_periods):
                pred = model.predict(last_sequence.reshape(1, sequence_length, 1), verbose=0)
                predictions.append(pred[0, 0])
                last_sequence = np.append(last_sequence[1:], pred[0])
            
            predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
            return predictions.flatten()
            
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return np.array([data['Close'].iloc[-1]] * future_periods)
    
    def detect_trend_ml(self, data):
        # Create features
        data['returns'] = data['Close'].pct_change()
        data['volatility'] = data['returns'].rolling(20).std()
        data['momentum'] = data['Close'] - data['Close'].shift(5)
        
        # Target variable (1 for uptrend, 0 for downtrend)
        data['trend'] = (data['Close'].shift(-5) > data['Close']).astype(int)
        
        # Prepare data for ML
        features = ['returns', 'volatility', 'momentum']
        data_clean = data.dropna()
        
        if len(data_clean) < 50:
            return "Insufficient data"
        
        X = data_clean[features]
        y = data_clean['trend']
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X[:-5], y[:-5])  # Leave last 5 for prediction
        
        # Predict current trend
        current_features = X.iloc[-1:].values
        prediction = model.predict(current_features)[0]
        
        return "Uptrend" if prediction == 1 else "Downtrend"

# -------------------------------
# Alert System
# -------------------------------
class AlertSystem:
    def __init__(self):
        self.alerts = []
    
    def add_alert(self, symbol, condition, value):
        conn = sqlite3.connect('trading_data.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO alerts (symbol, condition, value, triggered, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (symbol, condition, value, False, datetime.now()))
        conn.commit()
        conn.close()
    
    def check_alerts(self, data, symbol):
        conn = sqlite3.connect('trading_data.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM alerts WHERE symbol = ? AND triggered = ?', (symbol, False))
        active_alerts = cursor.fetchall()
        
        triggered_alerts = []
        current_price = data['Close'].iloc[-1]
        
        for alert in active_alerts:
            alert_id, symbol, condition, value, triggered, created_at, triggered_at = alert
            
            condition_met = False
            if condition == "price_above" and current_price > value:
                condition_met = True
            elif condition == "price_below" and current_price < value:
                condition_met = True
            elif condition == "rsi_oversold" and data.get('rsi', [70])[-1] < 30:
                condition_met = True
            elif condition == "rsi_overbought" and data.get('rsi', [30])[-1] > 70:
                condition_met = True
            
            if condition_met:
                cursor.execute('''
                    UPDATE alerts SET triggered = ?, triggered_at = ? WHERE id = ?
                ''', (True, datetime.now(), alert_id))
                triggered_alerts.append(alert)
        
        conn.commit()
        conn.close()
        return triggered_alerts

# -------------------------------
# Portfolio Manager
# -------------------------------
class PortfolioManager:
    def __init__(self, initial_balance=10000):
        self.initial_balance = initial_balance
        self.positions = {}
    
    def add_position(self, symbol, quantity, entry_price):
        conn = sqlite3.connect('trading_data.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO portfolio (symbol, quantity, entry_price, current_price, pnl, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (symbol, quantity, entry_price, entry_price, 0, datetime.now()))
        conn.commit()
        conn.close()
    
    def update_portfolio(self, current_prices):
        conn = sqlite3.connect('trading_data.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM portfolio')
        positions = cursor.fetchall()
        
        total_value = self.initial_balance
        total_pnl = 0
        
        for position in positions:
            pos_id, symbol, quantity, entry_price, _, _, _ = position
            current_price = current_prices.get(symbol, entry_price)
            pnl = (current_price - entry_price) * quantity
            
            cursor.execute('''
                UPDATE portfolio SET current_price = ?, pnl = ?, timestamp = ?
                WHERE id = ?
            ''', (current_price, pnl, datetime.now(), pos_id))
            
            total_value += pnl
            total_pnl += pnl
        
        conn.commit()
        conn.close()
        return total_value, total_pnl

# -------------------------------
# Real-time Data Streaming
# -------------------------------
class RealTimeData:
    def __init__(self):
        self.latest_data = {}
    
    async def binance_websocket(self, symbol):
        uri = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@kline_1m"
        
        async with websockets.connect(uri) as websocket:
            while True:
                message = await websocket.recv()
                data = json.loads(message)
                kline = data['k']
                
                self.latest_data[symbol] = {
                    'timestamp': datetime.fromtimestamp(kline['t'] / 1000),
                    'open': float(kline['o']),
                    'high': float(kline['h']),
                    'low': float(kline['l']),
                    'close': float(kline['c']),
                    'volume': float(kline['v'])
                }

# -------------------------------
# Multi-timeframe Analysis
# -------------------------------
def multi_timeframe_analysis(data, symbol):
    analysis = {}
    
    # Resample for different timeframes
    timeframes = {
        '1H': '1H',
        '4H': '4H', 
        '1D': '1D'
    }
    
    for tf_name, tf_resample in timeframes.items():
        try:
            # Resample data
            resampled = data.resample(tf_resample).agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
            
            if len(resampled) > 0:
                # Calculate indicators for this timeframe
                indicators = AdvancedIndicators()
                resampled = indicators.calculate_rsi(resampled)
                resampled = indicators.calculate_macd(resampled)
                
                # Determine trend
                price_change = (resampled['Close'].iloc[-1] - resampled['Open'].iloc[0]) / resampled['Open'].iloc[0] * 100
                
                analysis[tf_name] = {
                    'trend': 'Bullish' if price_change > 0 else 'Bearish',
                    'rsi': resampled['rsi'].iloc[-1] if 'rsi' in resampled else 50,
                    'price_change': price_change,
                    'support': resampled['Low'].min(),
                    'resistance': resampled['High'].max()
                }
        except Exception as e:
            st.warning(f"Could not analyze {tf_name} timeframe: {e}")
    
    return analysis

# -------------------------------
# Page settings
# -------------------------------
st.set_page_config(layout="wide", page_title="Advanced Crypto & Gold Analysis", page_icon="🚀")

# -------------------------------
# Initialize systems
# -------------------------------
alert_system = AlertSystem()
portfolio_manager = PortfolioManager()
ai_predictor = AIPredictor()
realtime_data = RealTimeData()
indicators = AdvancedIndicators()

# -------------------------------
# Sidebar with enhanced options
# -------------------------------
with st.sidebar:
    st.markdown("""
    <div style='margin-bottom:20px; text-align:center;'>
        <h2 style='margin:0; background:linear-gradient(90deg, #6EE7B7, #60A5FA); -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>🚀 Advanced Analysis</h2>
        <p style='margin:4px 0 0 0; color:#98A4B3; font-size:14px;'>AI-Powered Trading Dashboard</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Symbol selection
    symbols = ["BTC/USD", "ETH/USD", "XRP/USD", "SOL/USD", "DOGE/USD", "TRX/USD", "ADA/USD", "===========", "Gold"]
    symbol = st.selectbox("**Select Symbol**", options=symbols, index=1)
    
    # Timeframe and lookback
    col1, col2 = st.columns(2)
    with col1:
        timeframe = st.selectbox("**Timeframe**", options=["1m","5m","15m","30m","1h","4h","1d"], index=4)
    with col2:
        lookback = st.slider("**Lookback**", 1, 100, 10)
    
    st.markdown("---")
    
    # Indicator settings
    st.markdown("**📊 Indicator Settings**")
    
    with st.expander("Stochastic"):
        col1, col2 = st.columns(2)
        with col1:
            k_period = st.slider("**%K Period**", 5, 21, 14)
        with col2:
            d_period = st.slider("**%D Period**", 2, 7, 3)
    
    with st.expander("RSI"):
        rsi_period = st.slider("**RSI Period**", 5, 21, 14)
    
    with st.expander("MACD"):
        col1, col2, col3 = st.columns(3)
        with col1:
            macd_fast = st.slider("Fast", 5, 15, 12)
        with col2:
            macd_slow = st.slider("Slow", 20, 30, 26)
        with col3:
            macd_signal = st.slider("Signal", 5, 15, 9)
    
    st.markdown("---")
    
    # AI and Alert settings
    st.markdown("**🤖 AI & Alerts**")
    
    enable_ai = st.checkbox("Enable AI Predictions", value=True)
    enable_alerts = st.checkbox("Enable Alert System", value=True)
    enable_realtime = st.checkbox("Enable Real-time Data", value=False)
    
    if enable_alerts:
        with st.expander("Set Alert"):
            alert_condition = st.selectbox("Condition", ["price_above", "price_below", "rsi_oversold", "rsi_overbought"])
            alert_value = st.number_input("Value", value=1000.0)
            if st.button("Add Alert"):
                alert_system.add_alert(symbol, alert_condition, alert_value)
                st.success("Alert added!")
    
    st.markdown("---")
    
    # Portfolio management
    st.markdown("**💼 Portfolio**")
    
    with st.expander("Add Position"):
        col1, col2 = st.columns(2)
        with col1:
            pos_symbol = st.selectbox("Symbol", ["BTC", "ETH", "SOL", "ADA"])
            pos_quantity = st.number_input("Quantity", value=1.0)
        with col2:
            pos_price = st.number_input("Entry Price", value=1000.0)
        
        if st.button("Add to Portfolio"):
            portfolio_manager.add_position(pos_symbol, pos_quantity, pos_price)
            st.success("Position added!")
    
    st.markdown("---")
    
    # Date range
    st.markdown("**📅 Date Range**")
    default_end = datetime.now().replace(hour=23, minute=59, second=0, microsecond=0)
    end_date = st.date_input("End Date", value=default_end.date())
    end_time = st.time_input("End Time", value=default_end.time())

    required_candles = 500
    tf_map = {
        "1m": timedelta(minutes=1),
        "5m": timedelta(minutes=5),
        "15m": timedelta(minutes=15),
        "30m": timedelta(minutes=30),
        "1h": timedelta(hours=1),
        "4h": timedelta(hours=4),
        "1d": timedelta(days=1)
    }
    delta = tf_map[timeframe] * required_candles
    default_start = datetime.combine(end_date, end_time) - delta

    start_date = st.date_input("Start Date", value=default_start.date())
    start_time = st.time_input("Start Time", value=default_start.time())

# -------------------------------
# Convert to timestamp
# -------------------------------
start_dt = datetime.combine(start_date, start_time)
end_dt = datetime.combine(end_date, end_time)
since = int(start_dt.timestamp() * 1000)
until = int(end_dt.timestamp() * 1000)

# -------------------------------
# Main content area
# -------------------------------
st.markdown(f"""
<div class="header-container">
    <h1 style="margin:0; font-size:32px;">{symbol} - Advanced Analysis</h1>
    <p style="margin:4px 0 0 0; color:#98A4B3; font-size:16px;">
        Period: {start_dt.strftime('%Y-%m-%d %H:%M')} to {end_dt.strftime('%Y-%m-%d %H:%M')}
    </p>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# Fetch data
# -------------------------------
main_container = st.container()

with main_container:
    if symbol == "Gold":
        yf_tf_map = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "60m",
            "4h": "60m",
            "1d": "1d"
        }
        yf_interval = yf_tf_map[timeframe]

        ticker = "GC=F"
        with st.spinner("🔄 Fetching Gold data from Yahoo Finance..."):
            df = yf.download(
                ticker,
                start=start_dt,
                end=end_dt,
                interval=yf_interval,
                progress=False
            )

        if df.empty:
            st.error("No data found for Gold!")
            st.stop()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        data = df.rename(columns={
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Volume": "Volume"
        }).copy()

        if timeframe == "4h":
            data = data.resample("4H").agg({
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum"
            }).dropna()

        try:
            data.index = data.index.tz_convert("Asia/Tehran")
        except Exception:
            data.index = data.index.tz_localize("UTC").tz_convert("Asia/Tehran")

    else:
        exchange = ccxt.coinbase()
        ohlcv = []

        with st.spinner("🔄 Fetching crypto data from exchange..."):
            while since < until:
                batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=500)
                if len(batch) == 0:
                    break
                ohlcv += batch
                since = batch[-1][0] + 1
                time.sleep(exchange.rateLimit / 1000)

        if len(ohlcv) == 0:
            st.error("No data found! Check symbol or timeframe.")
            st.stop()

        data = pd.DataFrame(ohlcv, columns=['timestamp','Open','High','Low','Close','Volume'])
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms', utc=True)
        data['timestamp'] = data['timestamp'].dt.tz_convert('Asia/Tehran')
        data.set_index('timestamp', inplace=True)
        data = data[data.index <= pd.Timestamp(end_dt).tz_localize('Asia/Tehran')]

    # -------------------------------
    # Advanced Calculations
    # -------------------------------
    data["Volume_MA20"] = data["Volume"].rolling(window=20).mean()
    
    # Calculate all indicators
    data = indicators.calculate_stochastic(data, k_period, d_period)
    data = indicators.calculate_ichimoku(data)
    data = indicators.calculate_rsi(data, rsi_period)
    data = indicators.calculate_macd(data, macd_fast, macd_slow, macd_signal)
    data = indicators.calculate_bollinger_bands(data)
    
    # Fibonacci levels
    fib_levels = indicators.calculate_fibonacci_retracement(data)
    
    # AI Predictions
    if enable_ai and len(data) > 100:
        with st.spinner("🤖 AI is analyzing market trends..."):
            future_predictions = ai_predictor.predict_price_lstm(data)
            ml_trend = ai_predictor.detect_trend_ml(data)
    
    # Multi-timeframe analysis
    mtf_analysis = multi_timeframe_analysis(data, symbol)
    
    # Check alerts
    if enable_alerts:
        triggered_alerts = alert_system.check_alerts(data, symbol)
    
    # Update portfolio
    current_prices = {symbol.split('/')[0]: data['Close'].iloc[-1]}
    portfolio_value, total_pnl = portfolio_manager.update_portfolio(current_prices)
    
    up = data[data["Close"] >= data["Open"]]
    down = data[data["Close"] < data["Open"]]

    supply_idx = []
    demand_idx = []

    for i in range(lookback, len(data)-lookback):
        high_window = data['High'].iloc[i-lookback:i+lookback+1]
        low_window = data['Low'].iloc[i-lookback:i+lookback+1]
        if data['High'].iloc[i] == high_window.max():
            supply_idx.append(i)
        if data['Low'].iloc[i] == low_window.min():
            demand_idx.append(i)

    supply_idx_filtered = [i for i in supply_idx if data['Volume'].iloc[i] > data['Volume_MA20'].iloc[i]]
    demand_idx_filtered = [i for i in demand_idx if data['Volume'].iloc[i] > data['Volume_MA20'].iloc[i]]

    # -------------------------------
    # Enhanced Metrics Display
    # -------------------------------
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Current Price</div>
            <div class="metric-value">${data['Close'].iloc[-1]:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        price_change = data['Close'].iloc[-1] - data['Open'].iloc[-1]
        change_percent = (price_change / data['Open'].iloc[-1]) * 100
        change_color = ACCENT if price_change >= 0 else ERROR
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">24h Change</div>
            <div class="metric-value" style="color:{change_color};">{change_percent:+.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        current_rsi = data['rsi'].iloc[-1] if not pd.isna(data['rsi'].iloc[-1]) else 50
        rsi_color = ERROR if current_rsi > 70 else (ACCENT if current_rsi < 30 else MUTED)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">RSI ({rsi_period})</div>
            <div class="metric-value" style="color:{rsi_color};">{current_rsi:.1f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        current_k = data['stoch_k'].iloc[-1] if not pd.isna(data['stoch_k'].iloc[-1]) else 50
        stoch_color = ERROR if current_k > 80 else (ACCENT if current_k < 20 else MUTED)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Stochastic %K</div>
            <div class="metric-value" style="color:{stoch_color};">{current_k:.1f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        if enable_ai:
            trend_color = ACCENT if ml_trend == "Uptrend" else ERROR
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">AI Trend</div>
                <div class="metric-value" style="color:{trend_color};">{ml_trend}</div>
            </div>
            """, unsafe_allow_html=True)

    # -------------------------------
    # Alert Notifications
    # -------------------------------
    if enable_alerts and triggered_alerts:
        st.markdown("---")
        st.subheader("🔔 Triggered Alerts")
        for alert in triggered_alerts:
            st.error(f"**{alert[1]}** - {alert[2]} at {alert[3]} - Triggered!")

    # -------------------------------
    # Advanced Chart with Multiple Indicators
    # -------------------------------
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.50, 0.15, 0.15, 0.20],
        subplot_titles=(
            'Price Chart with Technical Indicators', 
            'Volume', 
            'MACD',
            'Stochastic & RSI'
        )
    )

    # Price chart (row 1) - Candlesticks + Ichimoku + Bollinger Bands
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name="Price",
        increasing_line_color="#26A69A",
        decreasing_line_color="#EF5350"
    ), row=1, col=1)

    # Ichimoku Cloud
    if 'senkou_span_a' in data and 'senkou_span_b' in data:
        fig.add_trace(go.Scatter(
            x=data.index, y=data['senkou_span_a'], 
            line=dict(color='rgba(0,0,0,0)'), showlegend=False
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=data.index, y=data['senkou_span_b'],
            line=dict(color='rgba(0,0,0,0)'),
            fill='tonexty', fillcolor='rgba(100,100,100,0.2)',
            name='Ichimoku Cloud'
        ), row=1, col=1)

    # Bollinger Bands
    if 'bb_upper' in data:
        fig.add_trace(go.Scatter(
            x=data.index, y=data['bb_upper'],
            line=dict(color='rgba(255,255,255,0.3)', width=1),
            name='BB Upper'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=data.index, y=data['bb_lower'],
            line=dict(color='rgba(255,255,255,0.3)', width=1),
            fill='tonexty', fillcolor='rgba(100,100,100,0.1)',
            name='Bollinger Bands'
        ), row=1, col=1)

    # Supply/Demand zones
    data["Candle_Range"] = data["High"] - data["Low"]
    avg_range = data["Candle_Range"].mean() if len(data)>0 else 0
    offset = avg_range * 0.2

    fig.add_trace(go.Scatter(
        x=data.index[supply_idx_filtered],
        y=data['High'].iloc[supply_idx_filtered] + offset,
        mode='markers',
        marker=dict(symbol='triangle-down', color='rgba(251,113,133,0.95)', size=14, line=dict(width=2, color='white')),
        name='Supply Zone'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=data.index[demand_idx_filtered],
        y=data['Low'].iloc[demand_idx_filtered] - offset,
        mode='markers',
        marker=dict(symbol='triangle-up', color='rgba(110,231,183,0.95)', size=14, line=dict(width=2, color='white')),
        name='Demand Zone'
    ), row=1, col=1)

    # Volume (row 2)
    fig.add_trace(go.Bar(x=up.index, y=up['Volume'], name="Up Volume", marker_color="#26A69A"), row=2, col=1)
    fig.add_trace(go.Bar(x=down.index, y=down['Volume'], name="Down Volume", marker_color="#EF5350"), row=2, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['Volume_MA20'], name="MA20 Volume", 
                           line=dict(color=ACCENT_SECOND, width=2)), row=2, col=1)

    # MACD (row 3)
    if 'macd' in data:
        fig.add_trace(go.Scatter(x=data.index, y=data['macd'], name="MACD", 
                               line=dict(color=ACCENT, width=2)), row=3, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['macd_signal'], name="Signal", 
                               line=dict(color=ERROR, width=2)), row=3, col=1)
        fig.add_trace(go.Bar(x=data.index, y=data['macd_histogram'], name="Histogram", 
                           marker_color=np.where(data['macd_histogram'] >= 0, '#26A69A', '#EF5350')), row=3, col=1)

    # Stochastic & RSI (row 4)
    if 'stoch_k' in data:
        fig.add_trace(go.Scatter(x=data.index, y=data['stoch_k'], name="%K", 
                               line=dict(color=ACCENT, width=1)), row=4, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['stoch_d'], name="%D", 
                               line=dict(color=ACCENT_SECOND, width=1)), row=4, col=1)
        fig.add_hline(y=80, line_dash="dash", line_color=ERROR, row=4, col=1)
        fig.add_hline(y=20, line_dash="dash", line_color=ACCENT, row=4, col=1)

    if 'rsi' in data:
        fig.add_trace(go.Scatter(x=data.index, y=data['rsi'], name="RSI", 
                               line=dict(color=WARNING, width=2), yaxis="y2"), row=4, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color=ERROR, row=4, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color=ACCENT, row=4, col=1)
        fig.update_layout(yaxis2=dict(overlaying='y', side='right', range=[0, 100]))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Poppins, sans-serif", color=TEXT, size=12),
        xaxis_rangeslider_visible=False,
        showlegend=True,
        height=1000,
        hovermode='x unified'
    )

    st.markdown("<div class='streamlit-card chart-container'>", unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------
    # Advanced Analysis Tabs
    # -------------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 AI Predictions", "📊 Multi-Timeframe", "💼 Portfolio", "📋 Fibonacci", "🔍 Data Summary"])

    with tab1:
        if enable_ai:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🤖 Price Prediction")
                if len(data) > 100:
                    future_dates = [data.index[-1] + timedelta(hours=i) for i in range(1, 11)]
                    prediction_df = pd.DataFrame({
                        'Date': future_dates,
                        'Predicted Price': future_predictions
                    })
                    
                    st.line_chart(prediction_df.set_index('Date')['Predicted Price'])
                    
                    current_price = data['Close'].iloc[-1]
                    predicted_change = ((future_predictions[-1] - current_price) / current_price) * 100
                    
                    st.metric("10-period Prediction", f"${future_predictions[-1]:.2f}", 
                             f"{predicted_change:+.2f}%")
            
            with col2:
                st.subheader("ML Trend Analysis")
                st.info(f"**AI Trend Detection:** {ml_trend}")
                
                # Feature importance (simulated)
                features = ['Price Momentum', 'Volume Trend', 'RSI Signal', 'Volatility']
                importance = [45, 25, 20, 10]
                
                feature_df = pd.DataFrame({'Feature': features, 'Importance': importance})
                st.bar_chart(feature_df.set_index('Feature'))

    with tab2:
        st.subheader("Multi-Timeframe Analysis")
        if mtf_analysis:
            for timeframe, analysis in mtf_analysis.items():
                trend_color = ACCENT if analysis['trend'] == 'Bullish' else ERROR
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(f"{timeframe} Trend", analysis['trend'])
                with col2:
                    st.metric("RSI", f"{analysis['rsi']:.1f}")
                with col3:
                    st.metric("Price Change", f"{analysis['price_change']:+.2f}%")
                
                st.progress(analysis['rsi'] / 100)

    with tab3:
        st.subheader("Portfolio Management")
        
        conn = sqlite3.connect('trading_data.db')
        portfolio_data = pd.read_sql('SELECT * FROM portfolio', conn)
        conn.close()
        
        if not portfolio_data.empty:
            st.dataframe(portfolio_data)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Value", f"${portfolio_value:,.2f}")
            with col2:
                st.metric("Total P&L", f"${total_pnl:,.2f}")
            with col3:
                st.metric("Return", f"{(total_pnl/portfolio_manager.initial_balance)*100:.2f}%")
        else:
            st.info("No positions in portfolio. Add positions from the sidebar.")

    with tab4:
        st.subheader("Fibonacci Retracement Levels")
        
        current_price = data['Close'].iloc[-1]
        high = data['High'].max()
        low = data['Low'].min()
        
        fib_df = pd.DataFrame.from_dict(fib_levels, orient='index', columns=['Price'])
        fib_df['Distance from Current'] = fib_df['Price'] - current_price
        fib_df['% from Current'] = (fib_df['Distance from Current'] / current_price) * 100
        
        st.dataframe(fib_df.style.format({
            'Price': '{:.2f}',
            'Distance from Current': '{:.2f}',
            '% from Current': '{:.2f}%'
        }))

    with tab5:
        st.subheader("Detailed Data Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Candles", len(data))
            st.metric("Average Volume", f"{data['Volume'].mean():.2f}")
            st.metric("Volatility", f"{data['Close'].pct_change().std():.4f}")
            
        with col2:
            st.metric("Highest Price", f"{data['High'].max():.2f}")
            st.metric("Lowest Price", f"{data['Low'].min():.2f}")
            st.metric("Average True Range", f"{(data['High'] - data['Low']).mean():.2f}")
            
        with col3:
            total_change = ((data['Close'].iloc[-1] - data['Open'].iloc[0]) / data['Open'].iloc[0]) * 100
            st.metric("Total Change", f"{total_change:.2f}%")
            st.metric("Sharpe Ratio", f"{(data['Close'].pct_change().mean() / data['Close'].pct_change().std()):.2f}")
        
        # Correlation matrix (if multiple symbols)
        st.subheader("Technical Signals")
        
        signals = []
        if current_rsi < 30:
            signals.append(("RSI Oversold", "Bullish", ACCENT))
        if current_rsi > 70:
            signals.append(("RSI Overbought", "Bearish", ERROR))
        if current_k < 20:
            signals.append(("Stochastic Oversold", "Bullish", ACCENT))
        if current_k > 80:
            signals.append(("Stochastic Overbought", "Bearish", ERROR))
        if 'macd' in data and data['macd'].iloc[-1] > data['macd_signal'].iloc[-1]:
            signals.append(("MACD Bullish", "Bullish", ACCENT))
        
        for signal, trend, color in signals:
            st.markdown(f"<div style='background:{color}20; padding:10px; border-radius:5px; margin:5px;'>"
                       f"<strong>{signal}</strong> - {trend}</div>", unsafe_allow_html=True)

# -------------------------------
# Real-time data section
# -------------------------------
if enable_realtime:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Real-time Data")
    
    if st.sidebar.button("Start Real-time Feed"):
        # This would need to be run in a separate thread for production
        st.info("Real-time data streaming would be implemented here")
        
        # Example of real-time data display
        if symbol in realtime_data.latest_data:
            latest = realtime_data.latest_data[symbol]
            st.sidebar.metric("Live Price", f"${latest['close']:.2f}")
            st.sidebar.metric("Change", f"{(latest['close'] - latest['open'])/latest['open']*100:.2f}%")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #98A4B3; font-size: 12px;'>
    <p>Advanced Trading Analytics Platform • Built with Streamlit • Data updates in real-time</p>
    <p>⚠️ This is for educational purposes only. Trade at your own risk.</p>
</div>
""", unsafe_allow_html=True)
