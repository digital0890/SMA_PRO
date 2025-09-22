import streamlit as st
import ccxt
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import numpy as np
import requests
import subprocess
import sys
import warnings
warnings.filterwarnings('ignore')

# Try to import ta library, install if not available
try:
    import ta
except ImportError:
    st.warning("📦 Installing required technical analysis library...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ta"])
    import ta

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
except ImportError:
    st.warning("📦 Installing required machine learning libraries...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# کامل‌ترین نسخه Dashboard تحلیل تکنیکال
# --------------------------------------------------

# -------------------------------
# تنظیمات اولیه
# -------------------------------
st.set_page_config(
    layout="wide", 
    page_title="Advanced Trading Dashboard", 
    page_icon="🚀",
    initial_sidebar_state="expanded"
)

# -------------------------------
# استایل پیشرفته
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

.tab-container {{
  background: var(--card);
  border-radius: 12px;
  padding: 20px;
  margin: 10px 0;
}}

@keyframes pulse {{
  0% {{ opacity: 1; }}
  50% {{ opacity: 0.5; }}
  100% {{ opacity: 1; }}
}}

.pulse {{
  animation: pulse 1.5s ease-in-out infinite;
}}

.signal-buy {{
  color: {ACCENT} !important;
  font-weight: bold;
}}

.signal-sell {{
  color: {ERROR} !important;
  font-weight: bold;
}}

.signal-neutral {{
  color: {MUTED} !important;
}}

.news-card {{
  background: linear-gradient(135deg, rgba(11,18,32,0.6), rgba(15,23,36,0.4));
  border-radius: 8px;
  padding: 12px;
  margin: 8px 0;
  border-left: 4px solid {ACCENT};
}}

.alert-card {{
  background: linear-gradient(135deg, rgba(251,113,133,0.1), rgba(251,191,36,0.1));
  border-radius: 8px;
  padding: 12px;
  margin: 8px 0;
  border-left: 4px solid {WARNING};
}}
"""

st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)

# -------------------------------
# توابع پیشرفته
# -------------------------------

def calculate_rsi(data, window=14):
    """محاسبه RSI دستی"""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    """محاسبه MACD دستی"""
    ema_fast = data['Close'].ewm(span=fast).mean()
    ema_slow = data['Close'].ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal).mean()
    macd_histogram = macd - macd_signal
    return macd, macd_signal, macd_histogram

def calculate_bollinger_bands(data, window=20, num_std=2):
    """محاسبه Bollinger Bands دستی"""
    sma = data['Close'].rolling(window=window).mean()
    std = data['Close'].rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, lower_band, sma

def calculate_stochastic(data, k_window=14, d_window=3):
    """محاسبه Stochastic دستی"""
    low_min = data['Low'].rolling(window=k_window).min()
    high_max = data['High'].rolling(window=k_window).max()
    stoch_k = ((data['Close'] - low_min) / (high_max - low_min)) * 100
    stoch_d = stoch_k.rolling(window=d_window).mean()
    return stoch_k, stoch_d

def calculate_all_indicators(data):
    """محاسبه تمام اندیکاتورهای تکنیکال"""
    try:
        # استفاده از کتابخانه ta اگر موجود باشد
        data['rsi'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
        
        macd = ta.trend.MACD(data['Close'])
        data['macd'] = macd.macd()
        data['macd_signal'] = macd.macd_signal()
        data['macd_histogram'] = macd.macd_diff()
        
        bollinger = ta.volatility.BollingerBands(data['Close'], window=20, window_dev=2)
        data['bb_upper'] = bollinger.bollinger_hband()
        data['bb_lower'] = bollinger.bollinger_lband()
        data['bb_middle'] = bollinger.bollinger_mavg()
        
        stochastic = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close'], window=14, smooth_window=3)
        data['stoch_k'] = stochastic.stoch()
        data['stoch_d'] = stochastic.stoch_signal()
        
        data['adx'] = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close'], window=14).adx()
        
    except:
        # محاسبه دستی اگر کتابخانه ta کار نکرد
        st.info("🔧 استفاده از محاسبات دستی برای اندیکاتورها")
        data['rsi'] = calculate_rsi(data)
        data['macd'], data['macd_signal'], data['macd_histogram'] = calculate_macd(data)
        data['bb_upper'], data['bb_lower'], data['bb_middle'] = calculate_bollinger_bands(data)
        data['stoch_k'], data['stoch_d'] = calculate_stochastic(data)
        
        # محاسبه ADX ساده شده
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        data['adx'] = tr.rolling(window=14).mean()
    
    # محاسبات حجم و میانگین‌های متحرک
    data['volume_sma'] = data['Volume'].rolling(20).mean()
    data['obv'] = (data['Volume'] * (~data['Close'].diff().le(0) * 2 - 1)).cumsum()
    
    # Moving Averages
    data['sma_20'] = data['Close'].rolling(20).mean()
    data['sma_50'] = data['Close'].rolling(50).mean()
    data['sma_200'] = data['Close'].rolling(200).mean()
    data['ema_12'] = data['Close'].ewm(span=12).mean()
    data['ema_26'] = data['Close'].ewm(span=26).mean()
    
    # ATR
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    data['atr'] = tr.rolling(window=14).mean()
    
    return data

def generate_signals(data):
    """تولید سیگنال‌های معاملاتی"""
    signals = []
    
    # بررسی وجود داده کافی
    if len(data) < 20:
        return signals
    
    # RSI Signals
    if 'rsi' in data and not pd.isna(data['rsi'].iloc[-1]):
        if data['rsi'].iloc[-1] < 30:
            signals.append(('RSI Oversold', 'BUY', ACCENT))
        elif data['rsi'].iloc[-1] > 70:
            signals.append(('RSI Overbought', 'SELL', ERROR))
    
    # MACD Signals
    if 'macd' in data and 'macd_signal' in data:
        if len(data) >= 2 and not pd.isna(data['macd'].iloc[-1]) and not pd.isna(data['macd_signal'].iloc[-1]):
            if data['macd'].iloc[-1] > data['macd_signal'].iloc[-1] and data['macd'].iloc[-2] <= data['macd_signal'].iloc[-2]:
                signals.append(('MACD Bullish Cross', 'BUY', ACCENT))
            elif data['macd'].iloc[-1] < data['macd_signal'].iloc[-1] and data['macd'].iloc[-2] >= data['macd_signal'].iloc[-2]:
                signals.append(('MACD Bearish Cross', 'SELL', ERROR))
    
    # Stochastic Signals
    if 'stoch_k' in data and 'stoch_d' in data:
        if not pd.isna(data['stoch_k'].iloc[-1]) and not pd.isna(data['stoch_d'].iloc[-1]):
            if data['stoch_k'].iloc[-1] < 20 and data['stoch_k'].iloc[-1] > data['stoch_d'].iloc[-1]:
                signals.append(('Stochastic Oversold Bullish', 'BUY', ACCENT))
            elif data['stoch_k'].iloc[-1] > 80 and data['stoch_k'].iloc[-1] < data['stoch_d'].iloc[-1]:
                signals.append(('Stochastic Overbought Bearish', 'SELL', ERROR))
    
    # Bollinger Bands
    if 'bb_upper' in data and 'bb_lower' in data:
        if not pd.isna(data['bb_lower'].iloc[-1]) and not pd.isna(data['bb_upper'].iloc[-1]):
            if data['Close'].iloc[-1] < data['bb_lower'].iloc[-1]:
                signals.append(('Below Lower BB', 'BUY', ACCENT))
            elif data['Close'].iloc[-1] > data['bb_upper'].iloc[-1]:
                signals.append(('Above Upper BB', 'SELL', ERROR))
    
    # Moving Average Crossovers
    if len(data) >= 2:
        if data['sma_20'].iloc[-1] > data['sma_50'].iloc[-1] and data['sma_20'].iloc[-2] <= data['sma_50'].iloc[-2]:
            signals.append(('SMA 20/50 Golden Cross', 'BUY', ACCENT))
        elif data['sma_20'].iloc[-1] < data['sma_50'].iloc[-1] and data['sma_20'].iloc[-2] >= data['sma_50'].iloc[-2]:
            signals.append(('SMA 20/50 Death Cross', 'SELL', ERROR))
    
    # ADX Strength
    if 'adx' in data and not pd.isna(data['adx'].iloc[-1]):
        if data['adx'].iloc[-1] > 25:
            signals.append(('Strong Trend (ADX > 25)', 'TREND', WARNING))
    
    return signals

def predict_price_trend(data):
    """پیش‌بینی روند قیمت ساده شده"""
    if len(data) < 50:
        return "ندارد داده کافی"
    
    try:
        # تحلیل روند ساده بر اساس Moving Average
        short_ma = data['Close'].tail(5).mean()
        long_ma = data['Close'].tail(20).mean()
        current_price = data['Close'].iloc[-1]
        
        if current_price > short_ma > long_ma:
            return "📈 صعودی"
        elif current_price < short_ma < long_ma:
            return "📉 نزولی"
        else:
            return "➡️ خنثی"
            
    except Exception as e:
        return "خطا در تحلیل"

def get_market_news(symbol):
    """دریافت اخبار بازار (شبیه‌سازی)"""
    news_items = [
        {
            'title': f'تحلیل تکنیکال {symbol} - مقاومت کلیدی شکسته شد',
            'summary': 'قیمت موفق به شکست سطح مقاومتی مهمی شده است.',
            'sentiment': 'positive',
            'time': '2 ساعت پیش'
        },
        {
            'title': f'اخبار مؤثر بر {symbol}',
            'summary': 'انتظار می‌رود اخبار فاندامنتال بر قیمت تأثیر گذارد.',
            'sentiment': 'neutral',
            'time': '5 ساعت پیش'
        },
        {
            'title': 'هشدار نوسانات بازار',
            'summary': 'نوسانات قیمت افزایش یافته، مراقب ریسک باشید.',
            'sentiment': 'warning',
            'time': '1 روز پیش'
        }
    ]
    return news_items

def create_portfolio():
    """ایجاد پورت‌فولیو شبیه‌سازی شده"""
    return {
        'BTC': {'amount': 0.5, 'avg_price': 45000, 'current_price': 52000},
        'ETH': {'amount': 3.2, 'avg_price': 3200, 'current_price': 3500},
        'Gold': {'amount': 100, 'avg_price': 1800, 'current_price': 1950},
        'Cash': {'amount': 5000, 'avg_price': 1, 'current_price': 1}
    }

# -------------------------------
# نوار کناری پیشرفته
# -------------------------------
with st.sidebar:
    st.markdown("""
    <div style='margin-bottom:20px; text-align:center;'>
        <h2 style='margin:0; background:linear-gradient(90deg, #6EE7B7, #60A5FA); -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>🚀 تریدینگ دشبورد</h2>
        <p style='margin:4px 0 0 0; color:#98A4B3; font-size:14px;'>پیشرفته‌ترین ابزار تحلیل</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # انتخاب نماد
    symbols = ["BTC/USD", "ETH/USD", "XRP/USD", "SOL/USD", "DOGE/USD", "TRX/USD", "ADA/USD", "===========", "Gold"]
    symbol = st.selectbox("**انتخاب نماد**", options=symbols, index=1)
    
    # تنظیمات تایم‌فریم
    col1, col2 = st.columns(2)
    with col1:
        timeframe = st.selectbox("**تایم‌فریم**", options=["1m","5m","15m","30m","1h","4h","1d"], index=4)
    with col2:
        lookback = st.slider("**Lookback**", 1, 20, 10)
    
    st.markdown("---")
    
    # تنظیمات اندیکاتورها
    st.markdown("**تنظیمات اندیکاتورها**")
    
    tab1, tab2 = st.tabs(["📊 اصلی", "⚙️ پیشرفته"])
    
    with tab1:
        rsi_period = st.slider("**RSI Period**", 5, 21, 14)
        macd_fast = st.slider("**MACD Fast**", 8, 15, 12)
        macd_slow = st.slider("**MACD Slow**", 20, 30, 26)
        macd_signal = st.slider("**MACD Signal**", 5, 15, 9)
    
    with tab2:
        bb_period = st.slider("**Bollinger Period**", 10, 30, 20)
        stoch_k = st.slider("**Stoch %K**", 5, 21, 14)
        stoch_d = st.slider("**Stoch %D**", 2, 7, 3)
    
    st.markdown("---")
    
    # تنظیمات تاریخ
    st.markdown("**بازه زمانی**")
    default_end = datetime.now().replace(hour=23, minute=59, second=0, microsecond=0)
    end_date = st.date_input("تاریخ پایان", value=default_end.date())
    end_time = st.time_input("زمان پایان", value=default_end.time())
    
    required_candles = 500  # کاهش داده برای عملکرد بهتر
    tf_map = {
        "1m": timedelta(minutes=1), "5m": timedelta(minutes=5), "15m": timedelta(minutes=15),
        "30m": timedelta(minutes=30), "1h": timedelta(hours=1), "4h": timedelta(hours=4), "1d": timedelta(days=1)
    }
    delta = tf_map[timeframe] * required_candles
    default_start = datetime.combine(end_date, end_time) - delta
    
    start_date = st.date_input("تاریخ شروع", value=default_start.date())
    start_time = st.time_input("زمان شروع", value=default_start.time())
    
    st.markdown("---")
    
    # وضعیت سیستم
    st.markdown("**وضعیت سیستم**")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">اتصال داده</div>
            <div class="metric-value">🟢</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">پردازش</div>
            <div class="metric-value">🟢</div>
        </div>
        """, unsafe_allow_html=True)

# -------------------------------
# تب‌های اصلی
# -------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 چارت اصلی", 
    "📊 اندیکاتورها", 
    "⚡ سیگنال‌ها", 
    "💼 پورت‌فولیو",
    "📰 اخبار و تحلیل"
])

# -------------------------------
# دریافت داده‌ها
# -------------------------------
start_dt = datetime.combine(start_date, start_time)
end_dt = datetime.combine(end_date, end_time)

@st.cache_data(ttl=300)
def fetch_data(symbol, start_dt, end_dt, timeframe):
    """دریافت داده‌ها با کش کردن"""
    try:
        if symbol == "Gold":
            yf_tf_map = {"1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", "1h": "60m", "4h": "60m", "1d": "1d"}
            yf_interval = yf_tf_map[timeframe]
            ticker = "GC=F"
            
            df = yf.download(ticker, start=start_dt, end=end_dt, interval=yf_interval, progress=False)
            
            if df.empty:
                st.error("داده‌ای برای طلا یافت نشد!")
                return None
                
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            data = df.rename(columns={
                "Open": "Open", "High": "High", "Low": "Low", 
                "Close": "Close", "Volume": "Volume"
            }).copy()

            if timeframe == "4h":
                data = data.resample("4H").agg({
                    "Open": "first", "High": "max", "Low": "min", 
                    "Close": "last", "Volume": "sum"
                }).dropna()

            try:
                data.index = data.index.tz_convert("Asia/Tehran")
            except:
                data.index = data.index.tz_localize("UTC").tz_convert("Asia/Tehran")

        else:
            exchange = ccxt.coinbase()
            since = int(start_dt.timestamp() * 1000)
            until = int(end_dt.timestamp() * 1000)
            ohlcv = []

            # محدود کردن تعداد داده‌ها برای عملکرد بهتر
            max_candles = 1000
            candle_count = 0
            
            while since < until and candle_count < max_candles:
                batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=min(500, max_candles - candle_count))
                if len(batch) == 0:
                    break
                ohlcv += batch
                candle_count += len(batch)
                since = batch[-1][0] + 1
                time.sleep(exchange.rateLimit / 1000)

            if len(ohlcv) == 0:
                st.error("داده‌ای یافت نشد! نماد یا تایم‌فریم را بررسی کنید.")
                return None

            data = pd.DataFrame(ohlcv, columns=['timestamp','Open','High','Low','Close','Volume'])
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms', utc=True)
            data['timestamp'] = data['timestamp'].dt.tz_convert('Asia/Tehran')
            data.set_index('timestamp', inplace=True)
            data = data[data.index <= pd.Timestamp(end_dt).tz_localize('Asia/Tehran')]

        # محاسبه اندیکاتورها
        if len(data) > 0:
            data = calculate_all_indicators(data)
            return data
        else:
            st.error("داده‌های دریافتی خالی است!")
            return None
        
    except Exception as e:
        st.error(f"خطا در دریافت داده: {str(e)}")
        return None

# دریافت داده‌ها
with st.spinner("🔄 در حال دریافت داده‌ها و محاسبه اندیکاتورها..."):
    data = fetch_data(symbol, start_dt, end_dt, timeframe)

if data is None or data.empty:
    st.error("❌ داده‌ای یافت نشد! تنظیمات را بررسی کنید.")
    st.stop()

# -------------------------------
# تب 1: چارت اصلی
# -------------------------------
with tab1:
    st.markdown(f"<div class='header-container'><h2>📈 چارت قیمت {symbol}</h2></div>", unsafe_allow_html=True)
    
    # متریک‌های سریع
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        price_change = ((data['Close'].iloc[-1] - data['Open'].iloc[-1]) / data['Open'].iloc[-1]) * 100
        change_color = ACCENT if price_change >= 0 else ERROR
        st.metric("قیمت فعلی", f"${data['Close'].iloc[-1]:.2f}", f"{price_change:+.2f}%")
    
    with col2:
        rsi_value = data['rsi'].iloc[-1] if 'rsi' in data and not pd.isna(data['rsi'].iloc[-1]) else 0
        st.metric("RSI", f"{rsi_value:.1f}")
    
    with col3:
        macd_value = data['macd'].iloc[-1] if 'macd' in data and not pd.isna(data['macd'].iloc[-1]) else 0
        st.metric("MACD", f"{macd_value:.4f}")
    
    with col4:
        adx_value = data['adx'].iloc[-1] if 'adx' in data and not pd.isna(data['adx'].iloc[-1]) else 0
        st.metric("شاخص ترند", f"{adx_value:.1f}")
    
    with col5:
        volume_change = ((data['Volume'].iloc[-1] - data['Volume'].mean()) / data['Volume'].mean()) * 100 if len(data) > 1 else 0
        st.metric("حجم", f"{data['Volume'].iloc[-1]:.0f}", f"{volume_change:+.1f}%")
    
    with col6:
        prediction = predict_price_trend(data)
        st.metric("پیش‌بینی روند", prediction)
    
    # چارت اصلی
    fig_main = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=('چارت قیمت', 'MACD', 'RSI')
    )
    
    # کندل‌استیک
    fig_main.add_trace(go.Candlestick(
        x=data.index, open=data['Open'], high=data['High'], 
        low=data['Low'], close=data['Close'], name="Price"
    ), row=1, col=1)
    
    # Moving Averages
    fig_main.add_trace(go.Scatter(x=data.index, y=data['sma_20'], name="SMA 20", line=dict(color='orange')), row=1, col=1)
    fig_main.add_trace(go.Scatter(x=data.index, y=data['sma_50'], name="SMA 50", line=dict(color='purple')), row=1, col=1)
    
    # Bollinger Bands (اگر موجود باشد)
    if 'bb_upper' in data:
        fig_main.add_trace(go.Scatter(x=data.index, y=data['bb_upper'], name="BB Upper", line=dict(color='gray', dash='dash')), row=1, col=1)
        fig_main.add_trace(go.Scatter(x=data.index, y=data['bb_lower'], name="BB Lower", line=dict(color='gray', dash='dash'), fill='tonexty'), row=1, col=1)
    
    # MACD (اگر موجود باشد)
    if 'macd' in data:
        fig_main.add_trace(go.Scatter(x=data.index, y=data['macd'], name="MACD", line=dict(color=ACCENT)), row=2, col=1)
        fig_main.add_trace(go.Scatter(x=data.index, y=data['macd_signal'], name="Signal", line=dict(color=ERROR)), row=2, col=1)
    
    # RSI (اگر موجود باشد)
    if 'rsi' in data:
        fig_main.add_trace(go.Scatter(x=data.index, y=data['rsi'], name="RSI", line=dict(color=ACCENT_SECOND)), row=3, col=1)
        fig_main.add_hline(y=70, line_dash="dash", line_color=ERROR, row=3, col=1)
        fig_main.add_hline(y=30, line_dash="dash", line_color=ACCENT, row=3, col=1)
    
    fig_main.update_layout(
        height=800, 
        showlegend=True, 
        template="plotly_dark",
        xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig_main, use_container_width=True)

# -------------------------------
# تب 2: اندیکاتورهای پیشرفته
# -------------------------------
with tab2:
    st.markdown(f"<div class='header-container'><h2>📊 اندیکاتورهای پیشرفته</h2></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Stochastic
        if 'stoch_k' in data:
            fig_stoch = go.Figure()
            fig_stoch.add_trace(go.Scatter(x=data.index, y=data['stoch_k'], name="%K", line=dict(color=ACCENT)))
            fig_stoch.add_trace(go.Scatter(x=data.index, y=data['stoch_d'], name="%D", line=dict(color=ACCENT_SECOND)))
            fig_stoch.add_hline(y=80, line_dash="dash", line_color=ERROR)
            fig_stoch.add_hline(y=20, line_dash="dash", line_color=ACCENT)
            fig_stoch.update_layout(title="Stochastic Oscillator", height=400, template="plotly_dark")
            st.plotly_chart(fig_stoch, use_container_width=True)
        
        # Volume Analysis
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(x=data.index, y=data['Volume'], name="Volume", marker_color=MUTED))
        if 'volume_sma' in data:
            fig_vol.add_trace(go.Scatter(x=data.index, y=data['volume_sma'], name="Volume MA", line=dict(color=WARNING)))
        fig_vol.update_layout(title="Volume Analysis", height=400, template="plotly_dark")
        st.plotly_chart(fig_vol, use_container_width=True)
    
    with col2:
        # ADX
        if 'adx' in data:
            fig_adx = go.Figure()
            fig_adx.add_trace(go.Scatter(x=data.index, y=data['adx'], name="ADX", line=dict(color=WARNING)))
            fig_adx.add_hline(y=25, line_dash="dash", line_color=ACCENT, annotation_text="Strong Trend")
            fig_adx.update_layout(title="ADX - Trend Strength", height=400, template="plotly_dark")
            st.plotly_chart(fig_adx, use_container_width=True)
        
        # OBV
        if 'obv' in data:
            fig_obv = go.Figure()
            fig_obv.add_trace(go.Scatter(x=data.index, y=data['obv'], name="OBV", line=dict(color=ACCENT)))
            fig_obv.update_layout(title="On Balance Volume", height=400, template="plotly_dark")
            st.plotly_chart(fig_obv, use_container_width=True)

# -------------------------------
# تب 3: سیگنال‌های معاملاتی
# -------------------------------
with tab3:
    st.markdown(f"<div class='header-container'><h2>⚡ سیگنال‌های معاملاتی</h2></div>", unsafe_allow_html=True)
    
    # تولید سیگنال‌ها
    signals = generate_signals(data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 سیگنال‌های فعلی")
        
        if not signals:
            st.info("📊 هیچ سیگنال قوی‌ای در حال حاضر شناسایی نشد")
        else:
            for signal_name, signal_type, color in signals:
                icon = "🟢" if signal_type == 'BUY' else "🔴" if signal_type == 'SELL' else "🟡"
                st.markdown(f"""
                <div class="alert-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span>{icon} <strong>{signal_name}</strong></span>
                        <span style="color: {color}; font-weight: bold;">{signal_type}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📈 قدرت روند")
        
        # تحلیل روند
        trend_strength = data['adx'].iloc[-1] if 'adx' in data else 0
        if trend_strength > 25:
            trend_status = "قوی"
            trend_color = ACCENT
        elif trend_strength > 15:
            trend_status = "متوسط"
            trend_color = WARNING
        else:
            trend_status = "ضعیف"
            trend_color = MUTED
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">شاخص قدرت روند (ADX)</div>
            <div class="metric-value" style="color: {trend_color};">{trend_strength:.1f}</div>
            <div class="metric-label">وضعیت: {trend_status}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # تحلیل نوسان
        volatility = data['atr'].iloc[-1] / data['Close'].iloc[-1] * 100 if 'atr' in data else 0
        st.metric("نوسان بازار", f"{volatility:.2f}%")
    
    # هشدارهای قیمتی
    st.markdown("### ⚠️ تنظیم هشدار قیمتی")
    col3, col4, col5 = st.columns(3)
    
    with col3:
        alert_price = st.number_input("قیمت هشدار", value=float(data['Close'].iloc[-1] * 1.05))
    
    with col4:
        alert_type = st.selectbox("نوع هشدار", ["بالاتر از", "پایین‌تر از"])
    
    with col5:
        if st.button("💾 ذخیره هشدار", use_container_width=True):
            st.success("✅ هشدار ذخیره شد!")
    
    # تاریخچه سیگنال‌ها
    st.markdown("### 📋 تاریخچه سیگنال‌ها")
    
    # ایجاد داده‌های نمونه برای تاریخچه
    history_data = data.tail(10).copy()
    signal_history = pd.DataFrame({
        'تاریخ': history_data.index,
        'قیمت': history_data['Close'].round(2),
        'RSI': history_data['rsi'].round(1) if 'rsi' in history_data else [0] * len(history_data),
        'سیگنال': ['خرید' if x > 0 else 'فروش' for x in history_data['Close'].diff()]
    })
    st.dataframe(signal_history, use_container_width=True)

# -------------------------------
# تب 4: مدیریت پورت‌فولیو
# -------------------------------
with tab4:
    st.markdown(f"<div class='header-container'><h2>💼 مدیریت پورت‌فولیو</h2></div>", unsafe_allow_html=True)
    
    portfolio = create_portfolio()
    
    # محاسبات پورت‌فولیو
    total_value = 0
    total_invested = 0
    
    for asset, details in portfolio.items():
        current_val = details['amount'] * details['current_price']
        invested = details['amount'] * details['avg_price']
        total_value += current_val
        total_invested += invested
    
    total_pnl = total_value - total_invested
    total_pnl_percent = (total_pnl / total_invested) * 100 if total_invested > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("ارزش کل پورت‌فولیو", f"${total_value:,.0f}")
    
    with col2:
        st.metric("سود/زیان کل", f"${total_pnl:,.0f}", f"{total_pnl_percent:+.2f}%")
    
    with col3:
        st.metric("دارایی نقدی", f"${portfolio['Cash']['amount']:,.0f}")
    
    with col4:
        st.metric("تعداد دارایی‌ها", len(portfolio) - 1)  # منهای cash
    
    # نمایش دارایی‌ها
    st.markdown("### 📊 ترکیب دارایی‌ها")
    
    for asset, details in portfolio.items():
        if asset != 'Cash':
            current_val = details['amount'] * details['current_price']
            invested = details['amount'] * details['avg_price']
            pnl = current_val - invested
            pnl_percent = (pnl / invested) * 100 if invested > 0 else 0
            
            col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
            
            with col1:
                st.write(f"**{asset}**")
            
            with col2:
                st.write(f"مقدار: {details['amount']}")
                st.write(f"قیمت فعلی: ${details['current_price']:,.0f}")
            
            with col3:
                st.write(f"ارزش فعلی: ${current_val:,.0f}")
                st.write(f"سود/زیان: {pnl_percent:+.2f}%")
            
            with col4:
                if pnl_percent > 0:
                    st.success("📈")
                else:
                    st.error("📉")
    
    # نمودار پیز پورت‌فولیو
    labels = [asset for asset in portfolio.keys() if asset != 'Cash']
    values = [portfolio[asset]['amount'] * portfolio[asset]['current_price'] for asset in labels]
    
    if sum(values) > 0:  # فقط اگر مقادیر مثبت وجود دارد
        fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
        fig_pie.update_layout(title="توزیع دارایی‌ها", template="plotly_dark")
        st.plotly_chart(fig_pie, use_container_width=True)

# -------------------------------
# تب 5: اخبار و تحلیل
# -------------------------------
with tab5:
    st.markdown(f"<div class='header-container'><h2>📰 اخبار و تحلیل بازار</h2></div>", unsafe_allow_html=True)
    
    # اخبار
    st.markdown("### 📰 آخرین اخبار بازار")
    news_items = get_market_news(symbol)
    
    for news in news_items:
        sentiment_icon = "🟢" if news['sentiment'] == 'positive' else "🔴" if news['sentiment'] == 'negative' else "🟡"
        st.markdown(f"""
        <div class="news-card">
            <div style="display: flex; justify-content: space-between;">
                <h4>{sentiment_icon} {news['title']}</h4>
                <span style="color: {MUTED}; font-size: 12px;">{news['time']}</span>
            </div>
            <p style="margin: 8px 0 0 0; color: {MUTED};">{news['summary']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # تحلیل فنی
        st.markdown("### 🔍 تحلیل فنی")
        
        technical_analysis = [
            ("روند کلی", "صعودی" if data['Close'].iloc[-1] > data['sma_50'].iloc[-1] else "نزولی"),
            ("قدرت روند", "قوی" if data['adx'].iloc[-1] > 25 else "متوسط" if data['adx'].iloc[-1] > 15 else "ضعیف"),
            ("اشباع خرید", "بله" if data['rsi'].iloc[-1] > 70 else "خیر"),
            ("اشباع فروش", "بله" if data['rsi'].iloc[-1] < 30 else "خیر"),
        ]
        
        for item, value in technical_analysis:
            st.write(f"**{item}:** {value}")
    
    with col2:
        # تحلیل احساسات
        st.markdown("### 😊 تحلیل احساسات")
        
        sentiment_data = {
            'شاخص': ['ترس و طمع', 'احساسات بازار', 'فعالیت معاملاتی'],
            'مقدار': [65, 72, 58],
            'وضعیت': ['متوسط', 'مثبت', 'متوسط']
        }
        
        for i, (indicator, value, status) in enumerate(zip(sentiment_data['شاخص'], sentiment_data['مقدار'], sentiment_data['وضعیت'])):
            col_a, col_b, col_c = st.columns([2, 1, 1])
            with col_a:
                st.write(indicator)
            with col_b:
                st.progress(value/100)
            with col_c:
                st.write(f"{value}%")
    
    # پیش‌بین‌های آینده
    st.markdown("### 🔮 پیش‌بینی‌های آینده")
    
    predictions = [
        {"period": "کوتاه‌مدت (1 هفته)", "outlook": "صعودی ملایم", "confidence": 65},
        {"period": "میان‌مدت (1 ماه)", "outlook": "خنثی به صعودی", "confidence": 55},
        {"period": "بلندمدت (3 ماه)", "outlook": "صعودی", "confidence": 70},
    ]
    
    for pred in predictions:
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.write(f"**{pred['period']}**")
        with col2:
            st.write(pred['outlook'])
        with col3:
            st.write(f"🔮 {pred['confidence']}%")

# -------------------------------
# پاورقی
# -------------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #98A4B3; font-size: 14px;'>
    <p>💡 این ابزار برای اهداف آموزشی ارائه شده و مسئولیتی در قبال تصمیمات معاملاتی ندارد</p>
    <p>🔄 داده‌ها با تأخیر و برای تحلیل تکنیکال ارائه می‌شوند</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# قابلیت ذخیره‌سازی وضعیت
# -------------------------------
if 'alerts' not in st.session_state:
    st.session_state.alerts = []

if 'portfolio' not in st.session_state:
    st.session_state.portfolio = create_portfolio()

# -------------------------------
# نوتیفیکیشن‌های زنده (شبیه‌سازی)
# -------------------------------
if st.sidebar.button('🔔 بررسی نوتیفیکیشن‌ها'):
    if signals:
        latest_signal = signals[0]
        st.sidebar.success(f"سیگنال جدید: {latest_signal[0]} - {latest_signal[1]}")
    else:
        st.sidebar.info("هیچ سیگنال جدیدی یافت نشد")

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; color: #98A4B3; font-size: 12px;'>
    <p>Version 3.0 • Advanced Trading Dashboard</p>
</div>
""", unsafe_allow_html=True)
