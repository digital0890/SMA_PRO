import streamlit as st
import ccxt
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta

# --------------------------------------------------
# Dark Modern Themed Streamlit App
# --------------------------------------------------
# -------------------------------
# Styling & Theme (Professional Dark)
# -------------------------------
FONT_IMPORT = "@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@400;500;600;700&display=swap');"
DARK_BG = "#0d1321"
CARD_BG = "#151b2d"
ACCENT = "#2dd4bf"
ACCENT_SECOND = "#4f46e5"
TEXT = "#f3f4f6"
MUTED = "#a1a1aa"
ERROR = "#ef4444"
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
  --border: rgba(255,255,255,0.1);
  --shadow: rgba(0,0,0,0.35);
}}
[data-testid='stAppViewContainer'] {{
  background: linear-gradient(180deg, #0d1321 0%, #1f2937 100%);
  color: var(--text);
  font-family: 'Inter', 'Poppins', system-ui, sans-serif;
  padding: 2rem;
  min-height: 100vh;
  overflow-x: hidden;
}}
[data-testid='stSidebar'] {{
  background: linear-gradient(180deg, #151b2d 0%, #1e293b 100%);
  border-right: 1px solid var(--border);
  padding: 1.75rem;
  box-shadow: 0 4px 20px var(--shadow);
  width: 300px;
  border-radius: 8px;
}}
.streamlit-card {{
  background: var(--card);
  border-radius: 14px;
  padding: 2rem;
  box-shadow: 0 8px 24px var(--shadow);
  border: 1px solid var(--border);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
}}
.streamlit-card:hover {{
  transform: translateY(-4px);
  box-shadow: 0 12px 32px var(--shadow);
}}
.stSelectbox, .stSlider, .stDateInput, .stTimeInput {{
  background: rgba(255,255,255,0.08);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 0.8rem;
  transition: all 0.3s ease;
}}
.stSelectbox:hover, .stSlider:hover, .stDateInput:hover, .stTimeInput:hover {{
  background: rgba(255,255,255,0.12);
  border-color: var(--accent-2);
}}
.stSelectbox > div > div > select,
.stDateInput > div > div > input,
.stTimeInput > div > div > input {{
  color: var(--text);
  background: transparent;
  border: none;
  outline: none;
  font-size: 0.95rem;
}}
.stSlider > div > div > div > div {{
  background: var(--accent);
  border-radius: 10px;
}}
.plotly-graph-div {{
  background: transparent !important;
  border-radius: 14px;
  border: 1px solid var(--border);
  overflow: hidden;
  box-shadow: 0 4px 16px var(--shadow);
}}
h3 {{
  font-family: 'Poppins', sans-serif;
  font-weight: 600;
  font-size: 1.7rem;
  color: var(--text);
  margin-bottom: 1.5rem;
  letter-spacing: -0.02em;
  border-bottom: 1px solid var(--border);
  padding-bottom: 0.5rem;
}}
button, .stButton > button {{
  background: var(--accent-2);
  color: var(--text);
  border: none;
  border-radius: 10px;
  padding: 0.7rem 1.4rem;
  font-family: 'Inter', sans-serif;
  font-weight: 500;
  font-size: 0.95rem;
  transition: all 0.3s ease;
}}
button:hover, .stButton > button:hover {{
  background: #4338ca;
  transform: translateY(-2px);
  box-shadow: 0 4px 12px var(--shadow);
}}
.chart-container {{
  margin-top: 2rem;
  animation: slideIn 0.5s ease-in-out;
}}
@keyframes slideIn {{
  from {{ opacity: 0; transform: translateY(15px); }}
  to {{ opacity: 1; transform: translateY(0); }}
}}
.stSpinner > div {{
  color: var(--accent);
  font-family: 'Inter', sans-serif;
}}
@media (max-width: 768px) {{
  [data-testid='stAppViewContainer'] {{
    padding: 1rem;
  }}
  [data-testid='stSidebar'] {{
    width: 100%;
    padding: 1.25rem;
    border-radius: 0;
  }}
  .streamlit-card {{
    padding: 1.5rem;
  }}
  h3 {{
    font-size: 1.4rem;
  }}
}}
@media (max-width: 480px) {{
  .streamlit-card {{
    padding: 1rem;
  }}
  button, .stButton > button {{
    padding: 0.6rem 1rem;
    font-size: 0.9rem;
  }}
}}
"""
st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)

# -------------------------------
# Page settings
# -------------------------------
st.set_page_config(layout="wide", page_title="Crypto & Gold Supply/Demand Analysis")

# -------------------------------
# Sidebar
# -------------------------------
with st.sidebar:
    st.markdown("<div style='margin-bottom:16px;'><h3 style='margin:0'>⚙️ تنظیمات</h3></div>", unsafe_allow_html=True)
    symbols = ["BTC/USD", "ETH/USD", "BNB/USD", "XRP/USD", "ADA/USD", "Gold"]
    symbol = st.selectbox("Select Symbol", options=symbols, index=1)
    timeframe = st.selectbox("Timeframe", options=["1m","5m","15m","30m","1h","4h","1d"], index=4)
    lookback = st.slider("Lookback (for Supply/Demand points)", 1, 10, 3)
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
# Fetch data
# -------------------------------
main_container = st.container()
with main_container:
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
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
        with st.spinner("Fetching Gold data from Yahoo Finance..."):
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
        with st.spinner("Fetching crypto data from exchange..."):
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
    # Chart (no subplot titles, no blue box)
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
        marker=dict(symbol='triangle-down', color='rgba(239,68,68,0.95)', size=12),
        name='Supply'
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=data.index[demand_idx_filtered],
        y=data['Low'].iloc[demand_idx_filtered] - offset,
        mode='markers',
        marker=dict(symbol='triangle-up', color='rgba(45,212,191,0.95)', size=12),
        name='Demand'
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=up.index,
        y=up['Volume'],
        name="Up Volume",
        marker_color="rgba(45,212,191,0.9)",
        opacity=0.9
    ), row=2, col=1)
    fig.add_trace(go.Bar(
        x=down.index,
        y=down['Volume'],
        name="Down Volume",
        marker_color="rgba(239,68,68,0.9)",
        opacity=0.9
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Volume_MA20'],
        mode="lines",
        name="MA20 Volume",
        line=dict(color="rgba(79,70,229,0.95)", width=2, dash='dash')
    ), row=2, col=1)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Poppins, sans-serif", color=TEXT),
        xaxis_rangeslider_visible=False,
        showlegend=True,
        height=820,
        barmode="overlay",
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=40, r=24, t=40, b=40),
        transition={'duration': 400, 'easing': 'cubic-in-out'}
    )
    fig.update_xaxes(showgrid=False, zeroline=False, showline=True, linewidth=0.6, linecolor="#1f2937")
    fig.update_yaxes(showgrid=True, gridwidth=0.4, gridcolor='rgba(255,255,255,0.03)', zeroline=False, showline=False)
    st.markdown("<div class='streamlit-card chart-container'>", unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
