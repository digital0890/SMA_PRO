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
# Styling & Theme (Dark Modern)
# -------------------------------
FONT_IMPORT = "@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Poppins:wght@400;600&display=swap');"

DARK_BG = "#0f1724"
CARD_BG = "#0b1220"
ACCENT = "#6EE7B7"
ACCENT_SECOND = "#60A5FA"
TEXT = "#E6EEF5"
MUTED = "#98A4B3"
ERROR = "#FB7185"

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
  padding: 18px;
  box-shadow: 0 6px 18px rgba(2,6,12,0.6);
  border: 1px solid rgba(255,255,255,0.03);
}}

.plotly-graph-div {{ background: transparent !important; }}
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
    st.markdown("<div style='margin-bottom:12px;'><h3 style='margin:0'>⚙️ Settings</h3></div>", unsafe_allow_html=True)

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
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

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
        increasing_line_color="rgba(60,180,120,0.95)",  # سبز تیره‌تر
        decreasing_line_color="rgba(200,60,90,0.95)",   # قرمز تیره‌تر
        increasing_fillcolor="rgba(60,180,120,0.95)",
        decreasing_fillcolor="rgba(200,60,90,0.95)"
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

# End of file
