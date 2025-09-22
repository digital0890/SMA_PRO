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
# کامل‌ترین نسخه Dashboard تحلیل تکنیکال - نسخه اصلاح شده
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
# توابع محاسبه اندیکاتورها - نسخه ایمن
# -------------------------------

def calculate_rsi(data, window=14):
    """محاسبه RSI دستی"""
    try:
        delta = data['Close'].diff()
        
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        
        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50)  # مقدار پیش‌فرض 50 برای داده‌های ناکافی
    except:
        return pd.Series([50] * len(data), index=data.index)

def calculate_macd(data, fast=12, slow=26, signal=9):
    """محاسبه MACD دستی"""
    try:
        ema_fast = data['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = data['Close'].ewm(span=slow, adjust=False).mean()
        
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_histogram = macd - macd_signal
        
        return macd, macd_signal, macd_histogram
    except:
        zeros = pd.Series([0] * len(data), index=data.index)
        return zeros, zeros, zeros

def calculate_bollinger_bands(data, window=20, num_std=2):
    """محاسبه Bollinger Bands دستی"""
    try:
        sma = data['Close'].rolling(window=window, min_periods=1).mean()
        std = data['Close'].rolling(window=window, min_periods=1).std()
        
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        
        return upper_band.fillna(method='bfill'), lower_band.fillna(method='bfill'), sma
    except:
        price = data['Close']
        return price, price, price

def calculate_stochastic(data, k_window=14, d_window=3):
    """محاسبه Stochastic دستی"""
    try:
        low_min = data['Low'].rolling(window=k_window, min_periods=1).min()
        high_max = data['High'].rolling(window=k_window, min_periods=1).max()
        
        stoch_k = ((data['Close'] - low_min) / (high_max - low_min)) * 100
        stoch_d = stoch_k.rolling(window=d_window, min_periods=1).mean()
        
        return stoch_k.fillna(50), stoch_d.fillna(50)
    except:
        fifties = pd.Series([50] * len(data), index=data.index)
        return fifties, fifties

def calculate_atr(data, window=14):
    """محاسبه Average True Range دستی - نسخه ایمن"""
    try:
        high_low = data['High'] - data['Low']
        high_close_prev = np.abs(data['High'] - data['Close'].shift(1))
        low_close_prev = np.abs(data['Low'] - data['Close'].shift(1))
        
        # جایگزینی مقادیر NaN با high_low
        high_close_prev = high_close_prev.fillna(high_low)
        low_close_prev = low_close_prev.fillna(high_low)
        
        tr = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        atr = tr.rolling(window=window, min_periods=1).mean()
        
        return atr.fillna(method='bfill')
    except Exception as e:
        # اگر خطایی رخ داد، یک سری ساده برگردان
        return pd.Series([0] * len(data), index=data.index)

def calculate_obv(data):
    """محاسبه On Balance Volume دستی"""
    try:
        price_diff = data['Close'].diff()
        obv = (np.sign(price_diff) * data['Volume']).fillna(0).cumsum()
        return obv
    except:
        return pd.Series([0] * len(data), index=data.index)

def calculate_all_indicators(data):
    """محاسبه تمام اندیکاتورهای تکنیکال به صورت ایمن"""
    try:
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
        data['sma_5'] = data['Close'].rolling(5, min_periods=1).mean()
        data['sma_20'] = data['Close'].rolling(20, min_periods=1).mean()
        data['sma_50'] = data['Close'].rolling(50, min_periods=1).mean()
        
        # Volume SMA
        data['volume_sma'] = data['Volume'].rolling(20, min_periods=1).mean()
        
        # شاخص ساده شده نوسان (جایگزین ADX)
        data['volatility'] = (data['High'] - data['Low']) / data['Close'] * 100
        
        return data
        
    except Exception as e:
        st.error(f"خطا در محاسبه اندیکاتورها: {str(e)}")
        # در صورت خطا، داده اصلی را بازگردان
        return data

def generate_signals(data):
    """تولید سیگنال‌های معاملاتی"""
    signals = []
    
    if len(data) < 20:
        return signals
    
    try:
        # RSI Signals
        current_rsi = data['rsi'].iloc[-1] if not pd.isna(data['rsi'].iloc[-1]) else 50
        if current_rsi < 30:
            signals.append(('RSI Oversold', 'BUY', ACCENT))
        elif current_rsi > 70:
            signals.append(('RSI Overbought', 'SELL', ERROR))
        
        # MACD Signals
        if len(data) >= 2:
            current_macd = data['macd'].iloc[-1] if not pd.isna(data['macd'].iloc[-1]) else 0
            current_signal = data['macd_signal'].iloc[-1] if not pd.isna(data['macd_signal'].iloc[-1]) else 0
            prev_macd = data['macd'].iloc[-2] if not pd.isna(data['macd'].iloc[-2]) else 0
            prev_signal = data['macd_signal'].iloc[-2] if not pd.isna(data['macd_signal'].iloc[-2]) else 0
            
            if current_macd > current_signal and prev_macd <= prev_signal:
                signals.append(('MACD Bullish Cross', 'BUY', ACCENT))
            elif current_macd < current_signal and prev_macd >= prev_signal:
                signals.append(('MACD Bearish Cross', 'SELL', ERROR))
        
        # Stochastic Signals
        current_stoch_k = data['stoch_k'].iloc[-1] if not pd.isna(data['stoch_k'].iloc[-1]) else 50
        current_stoch_d = data['stoch_d'].iloc[-1] if not pd.isna(data['stoch_d'].iloc[-1]) else 50
        
        if current_stoch_k < 20 and current_stoch_k > current_stoch_d:
            signals.append(('Stochastic Oversold', 'BUY', ACCENT))
        elif current_stoch_k > 80 and current_stoch_k < current_stoch_d:
            signals.append(('Stochastic Overbought', 'SELL', ERROR))
        
        # Bollinger Bands Signals
        current_close = data['Close'].iloc[-1]
        current_bb_lower = data['bb_lower'].iloc[-1] if not pd.isna(data['bb_lower'].iloc[-1]) else current_close
        current_bb_upper = data['bb_upper'].iloc[-1] if not pd.isna(data['bb_upper'].iloc[-1]) else current_close
        
        if current_close < current_bb_lower:
            signals.append(('Below Lower BB', 'BUY', ACCENT))
        elif current_close > current_bb_upper:
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
    symbols = ["BTC/USD", "ETH/USD", "ADA/USD", "SOL/USD", "Gold"]
    symbol = st.selectbox("**انتخاب نماد**", options=symbols, index=1)
    
    # تنظیمات تایم‌فریم
    timeframe = st.selectbox("**تایم‌فریم**", options=["1d", "1h", "4h"], index=0)
    
    st.markdown("---")
    
    # تنظیمات تاریخ
    st.markdown("**بازه زمانی**")
    
    end_date = st.date_input("تاریخ پایان", value=datetime.now().date())
    days_back = st.slider("**تعداد روزهای گذشته**", 30, 365, 90)
    
    start_date = end_date - timedelta(days=days_back)
    
    st.markdown(f"**تاریخ شروع:** {start_date.strftime('%Y-%m-%d')}")
    
    st.markdown("---")
    st.markdown("**وضعیت سیستم**")
    st.markdown("🟢 آنلاین")

# -------------------------------
# دریافت داده‌ها
# -------------------------------
@st.cache_data(ttl=3600)
def fetch_data(symbol, start_date, end_date, timeframe):
    """دریافت داده‌ها از API"""
    try:
        if symbol == "Gold":
            # داده‌های طلا از Yahoo Finance
            ticker = "GC=F"
            df = yf.download(ticker, start=start_date, end=end_date + timedelta(days=1))
            
            if df.empty:
                st.error("داده‌ای برای طلا یافت نشد!")
                return None
                
            data = df.rename(columns={
                "Open": "Open", "High": "High", "Low": "Low", 
                "Close": "Close", "Volume": "Volume"
            }).copy()
            
        else:
            # داده‌های ارز دیجیتال از Binance (رایگان)
            exchange = ccxt.binance()
            symbol_ccxt = symbol
            
            # تبدیل تایم‌فریم
            tf_map = {"1h": "1h", "4h": "4h", "1d": "1d"}
            timeframe_ccxt = tf_map.get(timeframe, "1d")
            
            # تبدیل تاریخ به timestamp
            since = int(datetime.combine(start_date, datetime.min.time()).timestamp() * 1000)
            
            try:
                ohlcv = exchange.fetch_ohlcv(symbol_ccxt, timeframe_ccxt, since=since, limit=1000)
            except:
                # اگر خطا داد، از Coinbase امتحان کن
                exchange = ccxt.coinbase()
                ohlcv = exchange.fetch_ohlcv(symbol_ccxt, timeframe_ccxt, since=since, limit=1000)
            
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
# نمایش اطلاعات اصلی
# -------------------------------
col1, col2, col3, col4 = st.columns(4)

current_price = data['Close'].iloc[-1]
price_change_pct = ((current_price - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100
price_change_color = ACCENT if price_change_pct >= 0 else ERROR

with col1:
    st.metric("قیمت فعلی", f"${current_price:.2f}", f"{price_change_pct:+.2f}%")

with col2:
    rsi_value = data['rsi'].iloc[-1] if not pd.isna(data['rsi'].iloc[-1]) else 50
    rsi_color = ACCENT if rsi_value < 30 else ERROR if rsi_value > 70 else MUTED
    st.metric("RSI", f"{rsi_value:.1f}")

with col3:
    volume = data['Volume'].iloc[-1]
    st.metric("حجم", f"{volume:,.0f}")

with col4:
    trend = predict_price_trend(data)
    st.metric("پیش‌بینی روند", trend)

# -------------------------------
# چارت اصلی
# -------------------------------
st.markdown("### 📈 چارت قیمت و اندیکاتورها")

# ایجاد تب‌های مختلف برای چارت‌ها
tab1, tab2, tab3 = st.tabs(["چارت قیمت", "اندیکاتورها", "سیگنال‌ها"])

with tab1:
    # چارت قیمت با Bollinger Bands
    fig_price = go.Figure()
    
    # کندل‌استیک
    fig_price.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name="Price"
    ))
    
    # Bollinger Bands
    fig_price.add_trace(go.Scatter(x=data.index, y=data['bb_upper'], 
                                 name="BB Upper", line=dict(color='gray', dash='dash')))
    fig_price.add_trace(go.Scatter(x=data.index, y=data['bb_lower'], 
                                 name="BB Lower", line=dict(color='gray', dash='dash'),
                                 fill='tonexty', fillcolor='rgba(128,128,128,0.1)'))
    
    fig_price.update_layout(height=500, template="plotly_dark", title="چارت قیمت با Bollinger Bands")
    st.plotly_chart(fig_price, use_container_width=True)

with tab2:
    # اندیکاتورها در ساب‌پلات
    fig_indicators = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('RSI', 'MACD', 'Stochastic')
    )
    
    # RSI
    fig_indicators.add_trace(go.Scatter(x=data.index, y=data['rsi'], name="RSI", 
                                      line=dict(color=ACCENT)), row=1, col=1)
    fig_indicators.add_hline(y=70, line_dash="dash", line_color=ERROR, row=1, col=1)
    fig_indicators.add_hline(y=30, line_dash="dash", line_color=ACCENT, row=1, col=1)
    
    # MACD
    fig_indicators.add_trace(go.Scatter(x=data.index, y=data['macd'], name="MACD", 
                                      line=dict(color=ACCENT)), row=2, col=1)
    fig_indicators.add_trace(go.Scatter(x=data.index, y=data['macd_signal'], name="Signal", 
                                      line=dict(color=ERROR)), row=2, col=1)
    fig_indicators.add_trace(go.Bar(x=data.index, y=data['macd_histogram'], name="Histogram", 
                                  marker_color=MUTED), row=2, col=1)
    
    # Stochastic
    fig_indicators.add_trace(go.Scatter(x=data.index, y=data['stoch_k'], name="%K", 
                                      line=dict(color=ACCENT)), row=3, col=1)
    fig_indicators.add_trace(go.Scatter(x=data.index, y=data['stoch_d'], name="%D", 
                                      line=dict(color=ACCENT_SECOND)), row=3, col=1)
    fig_indicators.add_hline(y=80, line_dash="dash", line_color=ERROR, row=3, col=1)
    fig_indicators.add_hline(y=20, line_dash="dash", line_color=ACCENT, row=3, col=1)
    
    fig_indicators.update_layout(height=600, template="plotly_dark", showlegend=True)
    st.plotly_chart(fig_indicators, use_container_width=True)

with tab3:
    # سیگنال‌ها
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
    
    # اطلاعات اضافی
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("میانگین حجم", f"{data['Volume'].mean():,.0f}")
        st.metric("بالاترین قیمت", f"${data['High'].max():.2f}")
    
    with col2:
        st.metric("نوسان روزانه", f"{(data['High'] - data['Low']).mean()/data['Close'].mean()*100:.2f}%")
        st.metric("پایین‌ترین قیمت", f"${data['Low'].min():.2f}")

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
