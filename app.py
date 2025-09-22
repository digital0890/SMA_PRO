import streamlit as st
import ccxt
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# --------------------------------------------------
# کامل‌ترین نسخه Dashboard تحلیل تکنیکال - نسخه بدون نیاز به نصب
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
# توابع محاسبه اندیکاتورها - بدون نیاز به کتابخانه اضافی
# -------------------------------

def calculate_rsi(data, window=14):
    """محاسبه RSI دستی"""
    delta = data['Close'].diff()
    
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    """محاسبه MACD دستی"""
    ema_fast = data['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = data['Close'].ewm(span=slow, adjust=False).mean()
    
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
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

def calculate_atr(data, window=14):
    """محاسبه Average True Range دستی"""
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    
    tr = np.maximum.reduce([high_low, high_close, low_close])
    atr = tr.rolling(window=window).mean()
    
    return atr

def calculate_obv(data):
    """محاسبه On Balance Volume دستی"""
    obv = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()
    return obv

def calculate_all_indicators(data):
    """محاسبه تمام اندیکاتورهای تکنیکال به صورت دستی"""
    # RSI
    data['rsi'] = calculate_rsi(data)
    
    # MACD
    data['macd'], data['macd_signal'], data['macd_histogram'] = calculate_macd(data)
    
    # Bollinger Bands
    data['bb_upper'], data['bb_lower'], data['bb_middle'] = calculate_bollinger_bands(data)
    
    # Stochastic
    data['stoch_k'], data['stoch_d'] = calculate_stochastic(data)
    
    # ATR
    data['atr'] = calculate_atr(data)
    
    # OBV
    data['obv'] = calculate_obv(data)
    
    # Moving Averages
    data['sma_5'] = data['Close'].rolling(5).mean()
    data['sma_20'] = data['Close'].rolling(20).mean()
    data['sma_50'] = data['Close'].rolling(50).mean()
    data['ema_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['ema_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    
    # Volume SMA
    data['volume_sma'] = data['Volume'].rolling(20).mean()
    
    # ADX ساده شده (بر اساس ATR)
    data['adx'] = data['atr'] / data['Close'] * 100  # شاخص ساده شده نوسان
    
    return data

def generate_signals(data):
    """تولید سیگنال‌های معاملاتی"""
    signals = []
    
    if len(data) < 20:
        return signals
    
    try:
        # RSI Signals
        if not pd.isna(data['rsi'].iloc[-1]):
            if data['rsi'].iloc[-1] < 30:
                signals.append(('RSI Oversold', 'BUY', ACCENT))
            elif data['rsi'].iloc[-1] > 70:
                signals.append(('RSI Overbought', 'SELL', ERROR))
        
        # MACD Signals
        if len(data) >= 2:
            if not pd.isna(data['macd'].iloc[-1]) and not pd.isna(data['macd_signal'].iloc[-1]):
                if data['macd'].iloc[-1] > data['macd_signal'].iloc[-1] and data['macd'].iloc[-2] <= data['macd_signal'].iloc[-2]:
                    signals.append(('MACD Bullish Cross', 'BUY', ACCENT))
                elif data['macd'].iloc[-1] < data['macd_signal'].iloc[-1] and data['macd'].iloc[-2] >= data['macd_signal'].iloc[-2]:
                    signals.append(('MACD Bearish Cross', 'SELL', ERROR))
        
        # Stochastic Signals
        if not pd.isna(data['stoch_k'].iloc[-1]) and not pd.isna(data['stoch_d'].iloc[-1]):
            if data['stoch_k'].iloc[-1] < 20 and data['stoch_k'].iloc[-1] > data['stoch_d'].iloc[-1]:
                signals.append(('Stochastic Oversold', 'BUY', ACCENT))
            elif data['stoch_k'].iloc[-1] > 80 and data['stoch_k'].iloc[-1] < data['stoch_d'].iloc[-1]:
                signals.append(('Stochastic Overbought', 'SELL', ERROR))
        
        # Bollinger Bands Signals
        if not pd.isna(data['bb_lower'].iloc[-1]) and not pd.isna(data['bb_upper'].iloc[-1]):
            if data['Close'].iloc[-1] < data['bb_lower'].iloc[-1]:
                signals.append(('Below Lower BB', 'BUY', ACCENT))
            elif data['Close'].iloc[-1] > data['bb_upper'].iloc[-1]:
                signals.append(('Above Upper BB', 'SELL', ERROR))
        
        # Moving Average Signals
        if data['sma_5'].iloc[-1] > data['sma_20'].iloc[-1] and data['sma_5'].iloc[-2] <= data['sma_20'].iloc[-2]:
            signals.append(('SMA 5/20 Golden Cross', 'BUY', ACCENT))
        elif data['sma_5'].iloc[-1] < data['sma_20'].iloc[-1] and data['sma_5'].iloc[-2] >= data['sma_20'].iloc[-2]:
            signals.append(('SMA 5/20 Death Cross', 'SELL', ERROR))
            
    except Exception as e:
        st.warning(f"خطا در تولید سیگنال‌ها: {str(e)}")
    
    return signals

def predict_price_trend(data):
    """پیش‌بینی روند قیمت ساده شده"""
    if len(data) < 10:
        return "داده ناکافی"
    
    try:
        # تحلیل روند بر اساس Moving Average
        short_ma = data['Close'].tail(5).mean()
        long_ma = data['Close'].tail(20).mean()
        current_price = data['Close'].iloc[-1]
        
        if current_price > short_ma > long_ma:
            return "📈 صعودی"
        elif current_price < short_ma < long_ma:
            return "📉 نزولی"
        else:
            return "➡️ خنثی"
    except:
        return "خطا در تحلیل"

def get_market_news(symbol):
    """دریافت اخبار بازار (شبیه‌سازی)"""
    base_symbol = symbol.split('/')[0] if '/' in symbol else symbol
    news_items = [
        {
            'title': f'تحلیل تکنیکال {base_symbol}',
            'summary': 'روند کنونی نیازمند بررسی دقیق تر است.',
            'sentiment': 'neutral',
            'time': 'امروز'
        },
        {
            'title': f'وضعیت بازار {base_symbol}',
            'summary': 'نوسانات معمول بازار در جریان است.',
            'sentiment': 'neutral',
            'time': 'امروز'
        }
    ]
    return news_items

def create_portfolio():
    """ایجاد پورت‌فولیو شبیه‌سازی شده"""
    return {
        'BTC': {'amount': 0.1, 'avg_price': 45000, 'current_price': 52000},
        'ETH': {'amount': 2.0, 'avg_price': 3200, 'current_price': 3500},
        'Cash': {'amount': 5000, 'avg_price': 1, 'current_price': 1}
    }

# -------------------------------
# نوار کناری
# -------------------------------
with st.sidebar:
    st.markdown("""
    <div style='margin-bottom:20px; text-align:center;'>
        <h2 style='margin:0; background:linear-gradient(90deg, #6EE7B7, #60A5FA); -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>🚀 تریدینگ دشبورد</h2>
        <p style='margin:4px 0 0 0; color:#98A4B3; font-size:14px;'>ابزار تحلیل تکنیکال</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # انتخاب نماد
    symbols = ["BTC/USD", "ETH/USD", "ADA/USD", "SOL/USD", "DOGE/USD", "Gold"]
    symbol = st.selectbox("**انتخاب نماد**", options=symbols, index=1)
    
    # تنظیمات تایم‌فریم
    timeframe = st.selectbox("**تایم‌فریم**", options=["1h", "4h", "1d", "1w"], index=2)
    
    st.markdown("---")
    
    # تنظیمات تاریخ
    st.markdown("**بازه زمانی**")
    
    # تنظیم تاریخ پیش‌فرض
    end_date = st.date_input("تاریخ پایان", value=datetime.now().date())
    days_back = st.slider("**تعداد روزهای گذشته**", 30, 365, 90)
    
    start_date = end_date - timedelta(days=days_back)
    
    st.markdown(f"**تاریخ شروع:** {start_date.strftime('%Y-%m-%d')}")
    
    st.markdown("---")
    
    # اطلاعات سیستم
    st.markdown("**وضعیت سیستم**")
    st.markdown("🟢 آنلاین")
    st.markdown(f"🔧 نسخه: 2.0")

# -------------------------------
# دریافت داده‌ها
# -------------------------------
@st.cache_data(ttl=3600)  # کش برای 1 ساعت
def fetch_data(symbol, start_date, end_date, timeframe):
    """دریافت داده‌ها از API"""
    try:
        if symbol == "Gold":
            # داده‌های طلا از Yahoo Finance
            ticker = "GC=F"
            df = yf.download(ticker, start=start_date, end=end_date + timedelta(days=1), interval="1d")
            
            if df.empty:
                return None
                
            data = df.rename(columns={
                "Open": "Open", "High": "High", "Low": "Low", 
                "Close": "Close", "Volume": "Volume"
            }).copy()
            
        else:
            # داده‌های ارز دیجیتال از Coinbase
            exchange = ccxt.coinbase()
            symbol_ccxt = symbol
            
            # تبدیل تایم‌فریم به فرمت CCxt
            tf_map = {"1h": "1h", "4h": "4h", "1d": "1d", "1w": "1w"}
            timeframe_ccxt = tf_map.get(timeframe, "1d")
            
            # تبدیل تاریخ به timestamp
            since = int(datetime.combine(start_date, datetime.min.time()).timestamp() * 1000)
            until = int(datetime.combine(end_date, datetime.max.time()).timestamp() * 1000)
            
            ohlcv = []
            current_since = since
            
            while current_since < until:
                try:
                    batch = exchange.fetch_ohlcv(symbol_ccxt, timeframe_ccxt, since=current_since, limit=300)
                    if not batch:
                        break
                    ohlcv.extend(batch)
                    current_since = batch[-1][0] + 1
                    time.sleep(0.1)  # جلوگیری از rate limit
                except Exception as e:
                    break
            
            if not ohlcv:
                return None
                
            data = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
            data.set_index('timestamp', inplace=True)
        
        return data
        
    except Exception as e:
        st.error(f"خطا در دریافت داده: {str(e)}")
        return None

# -------------------------------
# رابط اصلی
# -------------------------------
st.markdown(f"""
<div class='header-container'>
    <h1 style='margin:0; font-size:32px;'>تحلیل تکنیکال {symbol}</h1>
    <p style='margin:4px 0 0 0; color:#98A4B3; font-size:16px;'>
        بازه زمانی: {start_date.strftime('%Y-%m-%d')} تا {end_date.strftime('%Y-%m-%d')} | تایم‌فریم: {timeframe}
    </p>
</div>
""", unsafe_allow_html=True)

# دریافت داده‌ها
with st.spinner("📊 در حال دریافت داده‌ها..."):
    data = fetch_data(symbol, start_date, end_date, timeframe)

if data is None or data.empty:
    st.error("❌ داده‌ای یافت نشد. لطفاً تنظیمات را تغییر دهید.")
    st.stop()

# محاسبه اندیکاتورها
with st.spinner("🔬 در حال محاسبه اندیکاتورها..."):
    data = calculate_all_indicators(data)

# -------------------------------
# تب‌های اصلی
# -------------------------------
tab1, tab2, tab3 = st.tabs(["📈 چارت اصلی", "📊 اندیکاتورها", "⚡ سیگنال‌ها"])

with tab1:
    # متریک‌های سریع
    col1, col2, col3, col4 = st.columns(4)
    
    current_price = data['Close'].iloc[-1]
    price_change = ((current_price - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100
    
    with col1:
        st.metric("قیمت فعلی", f"${current_price:.2f}")
    
    with col2:
        st.metric("تغییر کل", f"{price_change:+.2f}%")
    
    with col3:
        rsi_value = data['rsi'].iloc[-1] if not pd.isna(data['rsi'].iloc[-1]) else 50
        st.metric("RSI", f"{rsi_value:.1f}")
    
    with col4:
        trend = predict_price_trend(data)
        st.metric("پیش‌بینی", trend)
    
    # چارت اصلی
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=('چارت قیمت', 'حجم معاملات')
    )
    
    # کندل‌استیک
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name="Price"
    ), row=1, col=1)
    
    # Moving Averages
    fig.add_trace(go.Scatter(x=data.index, y=data['sma_20'], name="SMA 20", line=dict(color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['sma_50'], name="SMA 50", line=dict(color='purple')), row=1, col=1)
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(x=data.index, y=data['bb_upper'], name="BB Upper", line=dict(color='gray', dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['bb_lower'], name="BB Lower", line=dict(color='gray', dash='dash'), fill='tonexty'), row=1, col=1)
    
    # حجم معاملات
    colors = ['green' if data['Close'].iloc[i] >= data['Open'].iloc[i] else 'red' for i in range(len(data))]
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name="Volume", marker_color=colors), row=2, col=1)
    
    fig.update_layout(height=600, template="plotly_dark", showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("📊 اندیکاتورهای تکنیکال")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # RSI
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=data.index, y=data['rsi'], name="RSI", line=dict(color=ACCENT)))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color=ERROR)
        fig_rsi.add_hline(y=30, line_dash="dash", line_color=ACCENT)
        fig_rsi.update_layout(title="RSI Indicator", height=300, template="plotly_dark")
        st.plotly_chart(fig_rsi, use_container_width=True)
        
        # MACD
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=data.index, y=data['macd'], name="MACD", line=dict(color=ACCENT)))
        fig_macd.add_trace(go.Scatter(x=data.index, y=data['macd_signal'], name="Signal", line=dict(color=ERROR)))
        fig_macd.add_trace(go.Bar(x=data.index, y=data['macd_histogram'], name="Histogram", marker_color=MUTED))
        fig_macd.update_layout(title="MACD", height=300, template="plotly_dark")
        st.plotly_chart(fig_macd, use_container_width=True)
    
    with col2:
        # Stochastic
        fig_stoch = go.Figure()
        fig_stoch.add_trace(go.Scatter(x=data.index, y=data['stoch_k'], name="%K", line=dict(color=ACCENT)))
        fig_stoch.add_trace(go.Scatter(x=data.index, y=data['stoch_d'], name="%D", line=dict(color=ACCENT_SECOND)))
        fig_stoch.add_hline(y=80, line_dash="dash", line_color=ERROR)
        fig_stoch.add_hline(y=20, line_dash="dash", line_color=ACCENT)
        fig_stoch.update_layout(title="Stochastic", height=300, template="plotly_dark")
        st.plotly_chart(fig_stoch, use_container_width=True)
        
        # Volume Analysis
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(x=data.index, y=data['Volume'], name="Volume", marker_color=MUTED))
        fig_vol.add_trace(go.Scatter(x=data.index, y=data['volume_sma'], name="Volume MA", line=dict(color=WARNING)))
        fig_vol.update_layout(title="Volume Analysis", height=300, template="plotly_dark")
        st.plotly_chart(fig_vol, use_container_width=True)

with tab3:
    st.subheader("⚡ سیگنال‌های معاملاتی")
    
    signals = generate_signals(data)
    
    if signals:
        st.success(f"🎯 {len(signals)} سیگنال شناسایی شد:")
        
        for signal_name, signal_type, color in signals:
            icon = "🟢" if signal_type == 'BUY' else "🔴"
            st.markdown(f"""
            <div class='alert-card'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <span>{icon} <strong>{signal_name}</strong></span>
                    <span style='color: {color}; font-weight: bold;'>{signal_type}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("📊 هیچ سیگنال واضحی شناسایی نشد. بازار در وضعیت خنثی قرار دارد.")
    
    # تحلیل بازار
    st.subheader("📈 تحلیل بازار")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        volatility = (data['High'] - data['Low']).mean() / data['Close'].mean() * 100
        st.metric("نوسان بازار", f"{volatility:.2f}%")
    
    with col2:
        avg_volume = data['Volume'].mean()
        st.metric("حجم متوسط", f"{avg_volume:,.0f}")
    
    with col3:
        trend_strength = "قوی" if abs(price_change) > 10 else "متوسط" if abs(price_change) > 5 else "ضعیف"
        st.metric("قدرت روند", trend_strength)

# -------------------------------
# پاورقی
# -------------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #98A4B3; font-size: 14px;'>
    <p>💡 این ابزار برای اهداف آموزشی ارائه شده است</p>
    <p>🔄 داده‌ها با تأخیر و از منابع عمومی دریافت می‌شوند</p>
</div>
""", unsafe_allow_html=True)
