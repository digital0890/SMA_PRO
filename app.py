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
# Keep core logic untouched; enhance UI/UX, typography,
# spacing, colors, accessibility and responsive layout.
# --------------------------------------------------

# -------------------------------
# Styling & Theme (Dark Modern)
# -------------------------------
FONT_IMPORT = "@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Poppins:wght@400;600&display=swap');"

DARK_BG = "#0f1724"           # deep navy-black
CARD_BG = "#0b1220"           # slightly lighter card background
ACCENT = "#6EE7B7"            # mint green accent
ACCENT_SECOND = "#60A5FA"     # blue accent for secondary highlights
TEXT = "#E6EEF5"              # soft off-white for main text
MUTED = "#98A4B3"             # muted text
ERROR = "#FB7185"             # accessible error

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

/* Page background and base font */
[data-testid='stAppViewContainer'] {{
  background: linear-gradient(180deg, rgba(5,10,16,1) 0%, rgba(10,14,20,1) 100%);
  color: var(--text);
  font-family: 'Inter', 'Poppins', system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
}}

/* Sidebar styling */
[data-testid='stSidebar'] {{
  background: linear-gradient(180deg, rgba(6,10,16,0.9), rgba(8,12,20,0.9));
  border-right: 1px solid rgba(255,255,255,0.03);
  padding: 18px 14px 24px 18px;
}}

/* Cards and containers */
.streamlit-card {{
  background: var(--card);
  border-radius: 12px;
  padding: 18px;
  box-shadow: 0 6px 18px rgba(2,6,12,0.6);
  border: 1px solid rgba(255,255,255,0.03);
}}

h1, h2, h3, h4 {{ color: var(--text); margin: 0 0 8px 0; }}

/* Headings */
.app-title {{
  font-family: 'Poppins', 'Inter', sans-serif;
  font-weight: 600;
  letter-spacing: -0.2px;
  color: var(--text);
}}

/* Inputs */
.stButton>button, button[kind="primary"] {{
  background: linear-gradient(90deg, var(--accent), var(--accent-2));
  color: #071226;
  border-radius: 10px;
  padding: 8px 14px;
  border: none;
  box-shadow: 0 6px 18px rgba(33,150,110,0.12);
  transition: transform 0.12s ease, box-shadow 0.12s ease;
}}
.stButton>button:hover {{ transform: translateY(-2px); box-shadow: 0 10px 30px rgba(33,150,110,0.12); }}

/* Inputs focus states for accessibility */
input, select, textarea {{
  outline-color: rgba(96,165,250,0.35) !important;
  box-shadow: 0 0 0 3px rgba(96,165,250,0.06) inset !important;
}}

/* Small helpers */
.kpi {{ display:inline-block; padding:8px 12px; border-radius:10px; background: rgba(255,255,255,0.02); color: var(--text); font-weight:600; }}

/* Responsive tweaks */
@media (max-width: 640px) {{
  .streamlit-card {{ padding: 12px; border-radius: 10px; }}
  .chart-container {{ height: 560px !important; }}
}}

/* Accessibility: high contrast focus */
:focus {{ outline: 3px solid rgba(110,231,183,0.18); outline-offset: 2px; }}

/* Improve plotly container background */
.plotly-graph-div {{ background: transparent !important; }}
"""

# inject CSS
st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)

# -------------------------------
# Page settings (unchanged behavior)
# -------------------------------
st.set_page_config(layout="wide", page_title="Crypto & Gold Supply/Demand Analysis")

# Title (centered with a subtle subtitle)
st.markdown("<div style='text-align:center; margin-bottom:8px;'>"
            "<h1 class='app-title' style='font-size:28px;'>📈 Crypto & Gold — Supply & Demand Analysis</h1>"
            "<div style='color:var(--muted); font-size:13px; margin-top:6px;'>Dark modern theme • Smooth UX • Accessible colors</div>"
            "</div>", unsafe_allow_html=True)
st.markdown("<hr style='opacity:0.06'/>", unsafe_allow_html=True)

# -------------------------------
# Sidebar: settings (kept logically the same)
# -------------------------------
with st.sidebar:
    st.markdown("<div style='margin-bottom:12px;'><h3 style='margin:0'>⚙️ Settings</h3></div>", unsafe_allow_html=True)

    # انتخاب ارز از لیست
    symbols = ["BTC/USD", "ETH/USD", "BNB/USD", "XRP/USD", "ADA/USD", "Gold"]
    symbol = st.selectbox("Select Symbol", options=symbols, index=1)

    timeframe = st.selectbox("Timeframe", options=["1m","5m","15m","30m","1h","4h","1d"], index=4)
    lookback = st.slider("Lookback (for Supply/Demand points)", 1, 10, 3)

    # Default end datetime: today 23:59 (kept same behaviour)
    default_end = datetime.now().replace(hour=23, minute=59, second=0, microsecond=0)
    end_date = st.date_input("End Date", value=default_end.date())
    end_time = st.time_input("End Time", value=default_end.time())

    required_candles = 500

    # Auto calculate start date based on timeframe and required candles
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

    # Extra UX helpers
    st.markdown("<div style='margin-top:12px; color:var(--muted); font-size:13px;'>Adjust timeframe & range to fetch 500+ candles for stable indicators.</div>", unsafe_allow_html=True)

# -------------------------------
# Convert to timestamp in ms (unchanged)
# -------------------------------
start_dt = datetime.combine(start_date, start_time)
end_dt = datetime.combine(end_date, end_time)
since = int(start_dt.timestamp() * 1000)
until = int(end_dt.timestamp() * 1000)

# -------------------------------
# Fetch data (kept behavior identical, with UX spinners/messages)
# -------------------------------

# Create a top-level container to hold main content nicely
main_container = st.container()

with main_container:
    # summary row
    col1, col2 = st.columns([3,1])
    with col1:
        st.markdown("<div class='streamlit-card'>", unsafe_allow_html=True)
        st.markdown(f"<div style='display:flex; gap:12px; align-items:center;'>"
                    f"<div class='kpi'>Symbol: <strong>{symbol}</strong></div>"
                    f"<div class='kpi'>Timeframe: <strong>{timeframe}</strong></div>"
                    f"<div class='kpi'>Lookback: <strong>{lookback}</strong></div>"
                    f"</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        # lightweight action area
        st.markdown("<div style='text-align:right'>", unsafe_allow_html=True)
        if st.button("Refresh Data"):
            st.experimental_rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # Fetching data (Gold vs Crypto)
    if symbol == "Gold":
        yf_tf_map = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "60m",
            "4h": "60m",   # resample later
            "1d": "1d"
        }
        yf_interval = yf_tf_map[timeframe]

        ticker = "GC=F"   # Gold Futures
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

        # flatten multiindex columns if present
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

        # ensure timezone alignment
        try:
            data.index = data.index.tz_convert("Asia/Tehran")
        except Exception:
            data.index = data.index.tz_localize("UTC").tz_convert("Asia/Tehran")

    else:
        exchange = ccxt.coinbase()
        ohlcv = []

        with st.spinner("Fetching crypto data from exchange..."):
            # fetch in batches until we reach 'until' (kept logic identical)
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

        # Create DataFrame for crypto
        data = pd.DataFrame(ohlcv, columns=['timestamp','Open','High','Low','Close','Volume'])
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms', utc=True)
        data['timestamp'] = data['timestamp'].dt.tz_convert('Asia/Tehran')
        data.set_index('timestamp', inplace=True)
        data = data[data.index <= pd.Timestamp(end_dt).tz_localize('Asia/Tehran')]

    # -------------------------------
    # Calculations (preserve exact logic)
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
    # Display number of identified points (styled)
    # -------------------------------
    st.markdown("<div class='streamlit-card' style='margin-bottom:12px;'>", unsafe_allow_html=True)
    st.markdown(f"<div style='display:flex; gap:14px; flex-wrap:wrap; align-items:center;'>"
                f"<div style='font-weight:700; color:var(--text);'>🔴 Supply Points: <span style='color:var(--accent-2);'>{len(supply_idx_filtered)}</span></div>"
                f"<div style='font-weight:700; color:var(--text);'>🟢 Demand Points: <span style='color:var(--accent);'>{len(demand_idx_filtered)}</span></div>"
                f"</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------
    # Plot chart with polished Plotly theme
    # -------------------------------
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05,
                        row_heights=[0.72,0.28],
                        subplot_titles=(f"{symbol} Candlestick Chart", "Volume"))

    # Candlestick
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

    # compute candle range for offset
    data["Candle_Range"] = data["High"] - data["Low"]
    avg_range = data["Candle_Range"].mean() if len(data)>0 else 0
    offset = avg_range * 0.2   # 20% of avg candle

    # Supply points
    fig.add_trace(go.Scatter(
        x=data.index[supply_idx_filtered],
        y=data['High'].iloc[supply_idx_filtered] + offset,
        mode='markers',
        marker=dict(symbol='triangle-down', color='rgba(251,113,133,0.95)', size=12),
        name='Supply'
    ), row=1, col=1)

    # Demand points
    fig.add_trace(go.Scatter(
        x=data.index[demand_idx_filtered],
        y=data['Low'].iloc[demand_idx_filtered] - offset,
        mode='markers',
        marker=dict(symbol='triangle-up', color='rgba(110,231,183,0.95)', size=12),
        name='Demand'
    ), row=1, col=1)

    # Up & down volume bars
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

    # MA20 Volume
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Volume_MA20'],
        mode="lines",
        name="MA20 Volume",
        line=dict(color="rgba(96,165,250,0.95)", width=2, dash='dash')
    ), row=2, col=1)

    # layout polish
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
        margin=dict(l=40, r=24, t=80, b=40),
        transition={'duration': 400, 'easing': 'cubic-in-out'}
    )

    # improve axes styling for accessibility
    fig.update_xaxes(showgrid=False, zeroline=False, showline=True, linewidth=0.6, linecolor="#1f2937")
    fig.update_yaxes(showgrid=True, gridwidth=0.4, gridcolor='rgba(255,255,255,0.03)', zeroline=False, showline=False)

    # -------------------------------
    # Render chart inside a styled card
    # -------------------------------
    st.markdown("<div class='streamlit-card chart-container'>", unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------
    # Minimal notes & accessibility tips
    # -------------------------------
    with st.expander("Design & Accessibility Notes (click to expand)"):
        st.markdown("- Color palette chosen for high contrast while reducing eye strain on dark backgrounds.")
        st.markdown("- Focus states and larger tap targets improve keyboard & mobile accessibility.")
        st.markdown("- Fonts: Inter & Poppins for clarity and modern feel.")
        st.markdown("- Animations are subtle and do not change app functionality.")

    # Footer spacing
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

# End of file
