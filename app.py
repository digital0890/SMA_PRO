import streamlit as st
import ccxt
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta

# --------------------------------------------------
# Dark Modern Themed Streamlit App (Enhanced)
# --------------------------------------------------

# -------------------------------
# Enhanced Styling & Theme
# -------------------------------
FONT_IMPORT = "@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@400;500;600;700&display=swap');"

DARK_BG = "#0f1724"
CARD_BG = "#0b1220"
ACCENT = "#6EE7B7"
ACCENT_SECOND = "#60A5FA"
TEXT = "#E6EEF5"
MUTED = "#98A4B3"
ERROR = "#FB7185"
WARNING = "#FBBF24"
SUCCESS = "#34D399"
PURPLE = "#A78BFA"

CUSTOM_CSS = f"""
{FONT_IMPORT}
:root{{
  --bg: {DARK_BG};
  --card: {CARD_BG};
  --accent: {ACCENT};
  --accent-2: {ACCENT_SECOND};
  --text: {TEXT};
  --muted: {MUTED};
  --error: {ERROR};
  --warning: {WARNING};
  --success: {SUCCESS};
  --purple: {PURPLE};
  --shadow-sm: 0 2px 8px rgba(2,6,12,0.4);
  --shadow-md: 0 6px 18px rgba(2,6,12,0.6);
  --shadow-lg: 0 12px 24px rgba(2,6,12,0.8);
  --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}}

[data-testid='stAppViewContainer'] {{
  background: linear-gradient(180deg, rgba(5,10,16,1) 0%, rgba(10,14,20,1) 100%);
  color: var(--text);
  font-family: 'Inter', 'Poppins', system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
  text-shadow: 0 1px 2px rgba(0,0,0,0.3);
}}

[data-testid='stSidebar'] {{
  background: linear-gradient(180deg, rgba(6,10,16,0.95), rgba(8,12,20,0.95));
  border-right: 1px solid rgba(255,255,255,0.05);
  padding: 24px 18px 30px 22px;
  box-shadow: var(--shadow-md);
  backdrop-filter: blur(10px);
}}

[data-testid='stSidebar'] > div:first-child {{
  padding: 0;
}}

.streamlit-card {{
  background: var(--card);
  border-radius: 16px;
  padding: 24px;
  box-shadow: var(--shadow-md);
  border: 1px solid rgba(255,255,255,0.05);
  transition: var(--transition);
  position: relative;
  overflow: hidden;
}}

.streamlit-card::before {{
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 4px;
  background: linear-gradient(90deg, var(--accent), var(--accent-2), var(--purple));
  opacity: 0.7;
}}

.streamlit-card:hover {{
  box-shadow: var(--shadow-lg);
  transform: translateY(-2px);
  border-color: rgba(255,255,255,0.08);
}}

.chart-container {{
  margin-top: 24px;
  border-radius: 16px;
  overflow: hidden;
  position: relative;
}}

.plotly-graph-div {{ 
  background: transparent !important; 
  border-radius: 12px;
}}

/* Enhanced buttons */
button {{
  border-radius: 8px;
  font-weight: 500;
  transition: var(--transition);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  font-size: 0.85rem;
}}

button:hover {{
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}}

button:active {{
  transform: translateY(0);
}}

/* Enhanced sliders */
input[type="range"] {{
  -webkit-appearance: none;
  appearance: none;
  height: 6px;
  background: rgba(255,255,255,0.1);
  border-radius: 3px;
  outline: none;
}}

input[type="range"]::-webkit-slider-thumb {{
  -webkit-appearance: none;
  appearance: none;
  width: 18px;
  height: 18px;
  background: var(--accent);
  border-radius: 50%;
  cursor: pointer;
  transition: var(--transition);
}}

input[type="range"]::-webkit-slider-thumb:hover {{
  transform: scale(1.2);
  box-shadow: 0 0 10px rgba(110,231,183,0.5);
}}

/* Enhanced inputs */
input, select, textarea {{
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.1);
  border-radius: 8px;
  padding: 10px 14px;
  color: var(--text);
  transition: var(--transition);
}}

input:focus, select:focus, textarea:focus {{
  border-color: var(--accent);
  box-shadow: 0 0 0 3px rgba(110,231,183,0.1);
  outline: none;
}}

/* Enhanced headers */
h1, h2, h3 {{
  font-weight: 600;
  margin-bottom: 16px;
  letter-spacing: -0.025em;
}}

h1 {{
  font-size: 2rem;
  background: linear-gradient(90deg, var(--accent), var(--accent-2));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: 24px;
}}

h2 {{
  font-size: 1.5rem;
  color: var(--text);
}}

h3 {{
  font-size: 1.25rem;
  color: var(--text);
}}

/* Enhanced labels */
label {{
  font-weight: 500;
  margin-bottom: 6px;
  display: block;
  color: var(--muted);
  font-size: 0.9rem;
}}

/* Enhanced select boxes */
div[data-testid="stSelectbox"] > div > div > div {{
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.1);
  border-radius: 8px;
  padding: 10px 14px;
}}

div[data-testid="stSelectbox"]:hover > div > div > div {{
  border-color: var(--accent);
}}

/* Enhanced spinners */
[data-testid="stStatusWidget"] > div {{
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.1);
  border-radius: 8px;
  padding: 16px;
}}

/* Enhanced error messages */
div[data-testid="stException"] {{
  background: rgba(251,113,133,0.1);
  border: 1px solid rgba(251,113,133,0.3);
  border-radius: 8px;
  padding: 16px;
  color: var(--error);
  font-weight: 500;
}}

/* Hide footer */
footer {{
  visibility: hidden;
}}

/* Fade-in animation */
@keyframes fadeIn {{
  from {{ opacity: 0; transform: translateY(10px); }}
  to {{ opacity: 1; transform: translateY(0); }}
}}

.streamlit-card {{
  animation: fadeIn 0.5s ease-out forwards;
}}

/* Enhanced scrollbar */
::-webkit-scrollbar {{
  width: 10px;
}}

::-webkit-scrollbar-track {{
  background: rgba(255,255,255,0.05);
  border-radius: 5px;
}}

::-webkit-scrollbar-thumb {{
  background: rgba(255,255,255,0.1);
  border-radius: 5px;
}}

::-webkit-scrollbar-thumb:hover {{
  background: rgba(255,255,255,0.2);
}}

/* Enhanced metric cards */
.metric-card {{
  background: rgba(255,255,255,0.03);
  border-radius: 12px;
  padding: 16px;
  text-align: center;
  border: 1px solid rgba(255,255,255,0.05);
  transition: var(--transition);
}}

.metric-card:hover {{
  background: rgba(255,255,255,0.05);
  border-color: rgba(255,255,255,0.1);
  transform: translateY(-2px);
}}

.metric-value {{
  font-size: 1.5rem;
  font-weight: 600;
  margin: 8px 0 0 0;
}}

.metric-label {{
  font-size: 0.85rem;
  color: var(--muted);
  margin: 0;
}}
"""

st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)

# -------------------------------
# Page settings
# -------------------------------
st.set_page_config(layout="wide", page_title="Crypto & Gold Supply/Demand Analysis")

# -------------------------------
# Enhanced Sidebar
# -------------------------------
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; margin-bottom: 30px; padding: 20px; border-radius: 16px; background: rgba(255,255,255,0.03);'>
        <h1 style='margin: 0; font-size: 1.8rem;'>📊 تحلیل عرضه و تقاضا</h1>
        <p style='margin: 8px 0 0 0; color: var(--muted); font-size: 0.9rem;'>ارزهای دیجیتال و طلا</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='streamlit-card'>", unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top: 0;'>⚙️ تنظیمات</h3>", unsafe_allow_html=True)
    
    symbols = ["BTC/USD", "ETH/USD", "BNB/USD", "XRP/USD", "ADA/USD", "Gold"]
    symbol = st.selectbox("نماد را انتخاب کنید", options=symbols, index=1)
    
    timeframe = st.selectbox("تایم‌فریم", options=["1m","5m","15m","30m","1h","4h","1d"], index=4)
    lookback = st.slider("بازه نگاه به عقب (برای نقاط عرضه/تقاضا)", 1, 10, 3)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='streamlit-card'>", unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top: 0;'>📅 محدوده زمانی</h3>", unsafe_allow_html=True)
    
    default_end = datetime.now().replace(hour=23, minute=59, second=0, microsecond=0)
    end_date = st.date_input("تاریخ پایان", value=default_end.date())
    end_time = st.time_input("زمان پایان", value=default_end.time())
    
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
    
    start_date = st.date_input("تاریخ شروع", value=default_start.date())
    start_time = st.time_input("زمان شروع", value=default_start.time())
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style='margin-top: 30px; text-align: center; padding: 15px; border-radius: 12px; background: rgba(255,255,255,0.03);'>
        <p style='margin: 0; color: var(--muted); font-size: 0.85rem;'>نسخه 1.0 • ساخته شده با Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------
# Convert to timestamp
# -------------------------------
start_dt = datetime.combine(start_date, start_time)
end_dt = datetime.combine(end_date, end_time)
since = int(start_dt.timestamp() * 1000)
until = int(end_dt.timestamp() * 1000)

# -------------------------------
# Fetch data
# -------------------------------
main_container = st.container()

with main_container:
    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    
    # Enhanced header
    st.markdown(f"""
    <div class='streamlit-card' style='text-align: center; margin-bottom: 24px;'>
        <h1 style='margin: 0 0 8px 0; font-size: 2.2rem;'>تحلیل {symbol}</h1>
        <p style='margin: 0; color: var(--muted);'>تایم‌فریم: {timeframe} • بازه نگاه به عقب: {lookback}</p>
    </div>
    """, unsafe_allow_html=True)

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
        with st.spinner("در حال دریافت داده‌های طلا از Yahoo Finance..."):
            df = yf.download(
                ticker,
                start=start_dt,
                end=end_dt,
                interval=yf_interval,
                progress=False
            )

        if df.empty:
            st.error("داده‌ای برای طلا یافت نشد!")
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

        with st.spinner("در حال دریافت داده‌های ارز دیجیتال از صرافی..."):
            while since < until:
                batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=500)
                if len(batch) == 0:
                    break
                ohlcv += batch
                since = batch[-1][0] + 1
                time.sleep(exchange.rateLimit / 1000)

        if len(ohlcv) == 0:
            st.error("داده‌ای یافت نشد! نماد یا تایم‌فریم را بررسی کنید.")
            st.stop()

        data = pd.DataFrame(ohlcv, columns=['timestamp','Open','High','Low','Close','Volume'])
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms', utc=True)
        data['timestamp'] = data['timestamp'].dt.tz_convert('Asia/Tehran')
        data.set_index('timestamp', inplace=True)
        data = data[data.index <= pd.Timestamp(end_dt).tz_localize('Asia/Tehran')]

    # -------------------------------
    # Calculations
    # -------------------------------
    data["Volume_MA20"] = data["Volume"].rolling(window=20).mean()
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
    # Enhanced Stats Cards
    # -------------------------------
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_price = data["Close"].iloc[-1]
        st.markdown(f"""
        <div class='metric-card'>
            <p class='metric-label'>قیمت فعلی</p>
            <p class='metric-value' style='color: var(--accent);'>{current_price:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        change = (data["Close"].iloc[-1] - data["Close"].iloc[-2]) / data["Close"].iloc[-2] * 100
        change_color = ACCENT if change >= 0 else ERROR
        st.markdown(f"""
        <div class='metric-card'>
            <p class='metric-label'>تغییر 24h</p>
            <p class='metric-value' style='color: {change_color};'>{change:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        volume = data["Volume"].iloc[-1]
        st.markdown(f"""
        <div class='metric-card'>
            <p class='metric-label'>حجم معاملات</p>
            <p class='metric-value' style='color: var(--accent-2);'>{volume:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class='metric-card'>
            <p class='metric-label'>نقاط عرضه</p>
            <p class='metric-value' style='color: var(--error);'>{len(supply_idx_filtered)}</p>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------
    # Enhanced Chart
    # -------------------------------
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.72,0.28]
    )

    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name="Price",
        increasing_line_color=ACCENT,
        decreasing_line_color=ERROR,
        increasing_fillcolor=ACCENT,
        decreasing_fillcolor=ERROR
    ), row=1, col=1)

    data["Candle_Range"] = data["High"] - data["Low"]
    avg_range = data["Candle_Range"].mean() if len(data)>0 else 0
    offset = avg_range * 0.2

    fig.add_trace(go.Scatter(
        x=data.index[supply_idx_filtered],
        y=data['High'].iloc[supply_idx_filtered] + offset,
        mode='markers',
        marker=dict(symbol='triangle-down', color='rgba(251,113,133,0.95)', size=12),
        name='Supply'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=data.index[demand_idx_filtered],
        y=data['Low'].iloc[demand_idx_filtered] - offset,
        mode='markers',
        marker=dict(symbol='triangle-up', color='rgba(110,231,183,0.95)', size=12),
        name='Demand'
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=up.index,
        y=up['Volume'],
        name="Up Volume",
        marker_color="rgba(110,231,183,0.9)",
        opacity=0.9
    ), row=2, col=1)

    fig.add_trace(go.Bar(
        x=down.index,
        y=down['Volume'],
        name="Down Volume",
        marker_color="rgba(251,113,133,0.9)",
        opacity=0.9
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Volume_MA20'],
        mode="lines",
        name="MA20 Volume",
        line=dict(color="rgba(96,165,250,0.95)", width=2, dash='dash')
    ), row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Poppins, sans-serif", color=TEXT, size=12),
        xaxis_rangeslider_visible=False,
        showlegend=True,
        height=820,
        barmode="overlay",
        hovermode='x unified',
        legend=dict(
            orientation='h', 
            yanchor='bottom', 
            y=1.02, 
            xanchor='right', 
            x=1,
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(255,255,255,0.1)',
            borderwidth=1,
            borderpad=8
        ),
        margin=dict(l=40, r=24, t=40, b=40),
        transition={'duration': 400, 'easing': 'cubic-in-out'},
        hoverlabel=dict(
            bgcolor="rgba(15,23,36,0.9)",
            bordercolor="rgba(255,255,255,0.1)",
            font=dict(size=12)
        )
    )

    fig.update_xaxes(
        showgrid=False, 
        zeroline=False, 
        showline=True, 
        linewidth=0.6, 
        linecolor="#1f2937",
        tickfont=dict(size=11, color=MUTED)
    )

    fig.update_yaxes(
        showgrid=True, 
        gridwidth=0.4, 
        gridcolor='rgba(255,255,255,0.03)', 
        zeroline=False, 
        showline=False,
        tickfont=dict(size=11, color=MUTED)
    )

    st.markdown("<div class='streamlit-card chart-container'>", unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# End of file
