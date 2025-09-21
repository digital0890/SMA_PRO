import ccxt
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import streamlit as st
import time

# تنظیمات صفحه
st.set_page_config(
    page_title="تحلیل تکنیکال ETH/USD",
    page_icon="📈",
    layout="wide"
)

# عنوان برنامه
st.title("تحلیل تکنیکال ETH/USD با شناسایی نقاط Supply و Demand")

# نوار کناری برای تنظیمات
st.sidebar.header("تنظیمات تحلیل")

# انتخاب جفت ارز
symbol_options = {
    "ETH/USD": "ETH/USD",
    "BTC/USD": "BTC/USD",
    "ADA/USD": "ADA/USD"
}
selected_symbol = st.sidebar.selectbox("جفت ارز", list(symbol_options.keys()), index=0)

# انتخاب تایم فریم
timeframe_options = {
    "1h": "1h",
    "4h": "4h", 
    "1d": "1d"
}
selected_timeframe = st.sidebar.selectbox("تایم فریم", list(timeframe_options.keys()), index=0)

# انتخاب تعداد کندل‌ها
limit = st.sidebar.slider("تعداد کندل‌ها", min_value=100, max_value=1000, value=500, step=50)

# دکمه بروزرسانی داده‌ها
if st.sidebar.button("بروزرسانی داده‌ها"):
    st.rerun()

# نمایش اسپیندر هنگام دریافت داده
with st.spinner("در حال دریافت داده‌ها از صرافی..."):
    try:
        # -------------------------------
        # دریافت داده با CCXT
        # -------------------------------
        exchange = ccxt.coinbase()
        ohlcv = exchange.fetch_ohlcv(selected_symbol, selected_timeframe, limit=limit)

        # ساخت DataFrame
        data = pd.DataFrame(ohlcv, columns=['timestamp','Open','High','Low','Close','Volume'])

        # تبدیل timestamp به datetime با تایم‌زون تهران
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms', utc=True)
        data['timestamp'] = data['timestamp'].dt.tz_convert('Asia/Tehran')
        data.set_index('timestamp', inplace=True)

        # -------------------------------
        # محاسبات
        # -------------------------------
        # میانگین متحرک حجم (MA20)
        data["Volume_MA20"] = data["Volume"].rolling(window=20).mean()

        # تفکیک کندل‌های صعودی و نزولی
        up = data[data["Close"] >= data["Open"]]
        down = data[data["Close"] < data["Open"]]

        # شناسایی نقاط Supply و Demand (نسخه ساده)
        lookback = 3  # تعداد کندل‌های قبل و بعد برای مقایسه
        supply_idx = []
        demand_idx = []

        for i in range(lookback, len(data)-lookback):
            high_window = data['High'].iloc[i-lookback:i+lookback+1]
            low_window = data['Low'].iloc[i-lookback:i+lookback+1]

            if data['High'].iloc[i] == max(high_window):
                supply_idx.append(i)
            if data['Low'].iloc[i] == min(low_window):
                demand_idx.append(i)

        # -------------------------------
        # فیلتر نقاط Supply و Demand بر اساس Volume > Volume_MA20
        # -------------------------------
        supply_idx_filtered = [i for i in supply_idx if data['Volume'].iloc[i] > data['Volume_MA20'].iloc[i]]
        demand_idx_filtered = [i for i in demand_idx if data['Volume'].iloc[i] > data['Volume_MA20'].iloc[i]]

        # -------------------------------
        # رسم نمودار
        # -------------------------------
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.05,
                            row_heights=[0.7, 0.3],
                            subplot_titles=(f"نمودار کندل‌استیک ({selected_symbol})", "حجم معاملات"))

        # کندل‌استیک
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="قیمت"
        ), row=1, col=1)

        # نقاط Supply فیلتر شده (مثلث قرمز بالا)
        fig.add_trace(go.Scatter(
            x=data.index[supply_idx_filtered],
            y=data['High'].iloc[supply_idx_filtered] + 5,
            mode='markers',
            marker=dict(symbol='triangle-up', color='red', size=12),
            name='Supply'
        ), row=1, col=1)

        # نقاط Demand فیلتر شده (مثلث سبز پایین)
        fig.add_trace(go.Scatter(
            x=data.index[demand_idx_filtered],
            y=data['Low'].iloc[demand_idx_filtered] - 5,
            mode='markers',
            marker=dict(symbol='triangle-down', color='green', size=12),
            name='Demand'
        ), row=1, col=1)

        # حجم صعودی
        fig.add_trace(go.Bar(
            x=up.index,
            y=up['Volume'],
            name="حجم صعودی",
            marker_color="green",
            opacity=0.8
        ), row=2, col=1)

        # حجم نزولی
        fig.add_trace(go.Bar(
            x=down.index,
            y=down['Volume'],
            name="حجم نزولی",
            marker_color="red",
            opacity=0.8
        ), row=2, col=1)

        # MA20 حجم
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Volume_MA20'],
            mode="lines",
            name="MA20 حجم",
            line=dict(color="orange", width=2)
        ), row=2, col=1)

        # استایل نهایی
        fig.update_layout(
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            showlegend=True,
            height=800,
            barmode="overlay"
        )

        fig.update_layout(
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            showlegend=True,
            height=800,
            barmode="overlay",
            # اضافه کردن خط عمودی متحرک
            xaxis=dict(
                showspikes=True,
                spikemode='across',
                spikesnap='cursor',
                spikedash='solid',
                spikecolor='white',
                spikethickness=1
            ),
            # اضافه کردن خط عمودی به نمودار حجم نیز
            xaxis2=dict(
                showspikes=True,
                spikemode='across',
                spikesnap='cursor',
                spikedash='solid',
                spikecolor='white',
                spikethickness=1
            )
        )

        # اضافه کردن خطوط افقی متحرک نیز (اختیاری)
        fig.update_yaxes(
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
            spikedash='solid',
            spikecolor='white',
            spikethickness=1
        )

        fig.update_yaxes(
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
            spikedash='solid',
            spikecolor='white',
            spikethickness=1,
            row=2, col=1
        )

        # نمایش نمودار در Streamlit
        st.plotly_chart(fig, use_container_width=True)
        
        # نمایش آمار کلی
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("تعداد نقاط Supply", len(supply_idx_filtered))
        with col2:
            st.metric("تعداد نقاط Demand", len(demand_idx_filtered))
        with col3:
            last_price = data['Close'].iloc[-1]
            st.metric("آخرین قیمت", f"{last_price:.2f}")
            
        # نمایش داده‌های خام
        with st.expander("مشاهده داده‌های خام"):
            st.dataframe(data.tail(20))

    except Exception as e:
        st.error(f"خطا در دریافت داده: {str(e)}")