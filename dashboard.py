import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
import yfinance as yf
from datetime import datetime
import numpy as np
from groq import Groq

st.set_page_config(page_title="Aether Analyzer", layout="wide", page_icon="📈")
st.title("🧬 Aether Analyzer — Pro Hybrid AI")
st.caption("yfinance data + Groq AI • Professional & sellable")

groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

@st.cache_data
def get_data(ticker: str, years: int = 10):
    try:
        data = yf.download(ticker, period=f"{years}y", auto_adjust=True)
        data = data.dropna()
        
        # Manual indicators (no pandas_ta)
        data["SMA_50"] = data["Close"].rolling(50).mean()
        delta = data["Close"].diff(1)
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = abs(delta.clip(upper=0)).rolling(14).mean()
        data["RSI_14"] = 100 - (100 / (1 + gain / loss))
        
        st.success(f"✅ Loaded {len(data)} days for {ticker}")
        return data
    except Exception:
        st.error(f"Could not load {ticker}")
        return pd.DataFrame()

def get_groq_sentiment(ticker):
    try:
        news = yf.Ticker(ticker).news[:8] if hasattr(yf.Ticker(ticker), 'news') else []
        news_text = "\n".join([item.get('title', '') for item in news])
        
        prompt = f"""You are a professional stock analyst. Analyze the current sentiment for {ticker}.
Recent news headlines:
{news_text}

Give:
1. Overall sentiment score (-1.0 very bearish → +1.0 very bullish)
2. One-sentence explanation
3. Recommendation (Strong Buy / Buy / Hold / Sell / Strong Sell)

Be concise."""
        
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=400
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Sentiment analysis unavailable: {e}"

def run_backtest(data):
    if len(data) < 200:
        return None
    data = data.copy()
    data['Return'] = data['Close'].pct_change()
    data['Signal'] = 0
    data['Signal'] = np.where(data['RSI_14'] < 30, 1, data['Signal'])
    data['Signal'] = np.where(data['RSI_14'] > 70, -1, data['Signal'])
    
    data['Strategy_Return'] = data['Return'] * data['Signal'].shift(1)
    data['Cumulative_Strategy'] = (1 + data['Strategy_Return']).cumprod()
    data['Cumulative_BuyHold'] = (1 + data['Return']).cumprod()
    
    total_return = data['Cumulative_Strategy'].iloc[-1] - 1
    buyhold_return = data['Cumulative_BuyHold'].iloc[-1] - 1
    sharpe = data['Strategy_Return'].mean() / data['Strategy_Return'].std() * np.sqrt(252) if data['Strategy_Return'].std() != 0 else 0
    max_dd = (data['Cumulative_Strategy'] / data['Cumulative_Strategy'].cummax() - 1).min()
    
    return {
        'total_return': total_return,
        'buyhold_return': buyhold_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'equity_curve': data[['Cumulative_Strategy', 'Cumulative_BuyHold']]
    }

# Sidebar
st.sidebar.header("Controls")
ticker_input = st.sidebar.text_input("Enter tickers (comma separated)", value="AAPL, NVDA, TSLA, F, SNX")
tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

years = st.sidebar.slider("History (years)", 1, 10, 10)

if st.sidebar.button("🔄 Refresh All Data"):
    st.cache_data.clear()
    st.rerun()

with st.spinner("📥 Loading market data..."):
    data_dict = {t: get_data(t, years) for t in tickers if t}

st.success("✅ All data loaded")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Portfolio Overview", "🔍 Technical Analysis", "💬 Groq LLM Insights", "📈 Advanced Backtesting", "💬 Chat with Your Portfolio"])

with tab1:
    st.subheader("Portfolio Overview")
    if data_dict:
        rows = []
        for t, df in data_dict.items():
            if len(df) > 1:
                latest = df.iloc[-1]
                prev = df.iloc[-2]
                change = ((latest['Close']/prev['Close'])-1)*100
                rsi = latest.get('RSI_14', 50)
                # Safe check for RSI
                if pd.isna(rsi) or rsi > 70:
                    signal = "🔴 SELL"
                elif rsi < 30:
                    signal = "🟢 STRONG BUY"
                else:
                    signal = "🟡 HOLD"
                rows.append([t, latest['Close'], change, rsi, signal])
        
        comparison_df = pd.DataFrame(rows, columns=["Ticker", "Price", "% Change", "RSI", "Signal"])
        st.dataframe(comparison_df.style.format({"Price": "${:.2f}", "% Change": "{:.1f}%"}), use_container_width=True)

        st.subheader("Correlation & Risk Heatmap")
        prices = pd.DataFrame({t: df['Close'] for t, df in data_dict.items() if not df.empty})
        if len(prices.columns) > 1:
            corr = prices.corr()
            fig = ff.create_annotated_heatmap(z=corr.values, x=list(corr.columns), y=list(corr.index), annotation_text=corr.round(2).values, colorscale="RdBu", showscale=True)
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Technical Indicators")
    ticker_choice = st.selectbox("Choose ticker", tickers)
    if ticker_choice in data_dict and len(data_dict[ticker_choice]) > 0:
        st.dataframe(data_dict[ticker_choice][['Close', 'SMA_50', 'RSI_14', 'MACD', 'BB_upper', 'BB_lower']].tail(10), use_container_width=True)

with tab3:
    st.subheader("💬 Groq LLM Insights + Sentiment")
    ticker_choice = st.selectbox("Choose ticker for sentiment", tickers, key="sentiment")
    if st.button("🔥 Run Full AI Analysis (Groq)"):
        with st.spinner("Analyzing recent news with Groq..."):
            sentiment = get_groq_sentiment(ticker_choice)
            st.markdown(sentiment)

with tab4:
    st.subheader("📈 Advanced Backtesting")
    ticker_choice = st.selectbox("Choose ticker to backtest", tickers, key="backtest")
    if st.button("Run Advanced Backtest"):
        result = run_backtest(data_dict[ticker_choice])
        if result:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Strategy Total Return", f"{result['total_return']:.1%}", delta=f"{result['total_return'] - result['buyhold_return']:.1%} vs Buy&Hold")
            col2.metric("Buy & Hold Return", f"{result['buyhold_return']:.1%}")
            col3.metric("Sharpe Ratio", f"{result['sharpe']:.2f}")
            col4.metric("Max Drawdown", f"{result['max_dd']:.1%}")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=result['equity_curve'].index, y=result['equity_curve']['Cumulative_Strategy'], name="Strategy", line=dict(color="green", width=3)))
            fig.add_trace(go.Scatter(x=result['equity_curve'].index, y=result['equity_curve']['Cumulative_BuyHold'], name="Buy & Hold", line=dict(color="gray", width=2)))
            fig.update_layout(title=f"Equity Curve for {ticker_choice} — Strategy vs Buy & Hold", height=500)
            st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.subheader("💬 Chat with Your Portfolio")
    st.caption("Ask anything about your loaded tickers — powered by Groq (near-instant)")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about your portfolio (e.g. 'Why is SNX flashing a SELL signal?' or 'What is the heaviest sector exposure?')"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        context = "Current portfolio tickers: " + ", ".join(tickers) + "\n"
        for t, df in data_dict.items():
            if len(df) > 1:
                latest = df.iloc[-1]
                change = ((latest['Close']/df.iloc[-2]['Close'])-1)*100
                context += f"{t}: Price ${latest['Close']:.2f}, % Change {change:.1f}%, RSI {latest.get('RSI_14', 50):.1f}\n"

        full_prompt = f"""You have access to the following real-time portfolio data:
{context}

User question: {prompt}

Answer clearly, professionally, and data-driven."""

        with st.spinner("Thinking..."):
            try:
                completion = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": full_prompt}],
                    temperature=0.3,
                    max_tokens=500
                )
                response = completion.choices[0].message.content
            except Exception as e:
                response = f"Groq error: {e}"

        st.session_state.chat_history.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

st.sidebar.success(f"✅ Loaded {len(tickers)} tickers")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
def run_backtest(data):
    if len(data) < 200:
        return None
    data = data.copy()
    data['Return'] = data['Close'].pct_change()
    data['Signal'] = 0
    data['Signal'] = np.where(data['RSI_14'] < 30, 1, data['Signal'])
    data['Signal'] = np.where(data['RSI_14'] > 70, -1, data['Signal'])
    
    data['Strategy_Return'] = data['Return'] * data['Signal'].shift(1)
    data['Cumulative_Strategy'] = (1 + data['Strategy_Return']).cumprod()
    data['Cumulative_BuyHold'] = (1 + data['Return']).cumprod()
    
    total_return = data['Cumulative_Strategy'].iloc[-1] - 1
    buyhold_return = data['Cumulative_BuyHold'].iloc[-1] - 1
    sharpe = data['Strategy_Return'].mean() / data['Strategy_Return'].std() * np.sqrt(252) if data['Strategy_Return'].std() != 0 else 0
    max_dd = (data['Cumulative_Strategy'] / data['Cumulative_Strategy'].cummax() - 1).min()
    
    return {
        'total_return': total_return,
        'buyhold_return': buyhold_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'equity_curve': data[['Cumulative_Strategy', 'Cumulative_BuyHold']]
    }

# Sidebar
st.sidebar.header("Controls")
ticker_input = st.sidebar.text_input("Enter tickers (comma separated)", value="AAPL, NVDA, TSLA, F, SNX")
tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

years = st.sidebar.slider("History (years)", 1, 10, 10)

if st.sidebar.button("🔄 Refresh All Data"):
    st.cache_data.clear()
    st.rerun()

with st.spinner("📥 Loading market data..."):
    data_dict = {t: get_data(t, years) for t in tickers if t}

st.success("✅ All data loaded")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Portfolio Overview", "🔍 Technical Analysis", "💬 Groq LLM Insights", "📈 Advanced Backtesting", "💬 Chat with Your Portfolio"])

with tab1:
    st.subheader("Portfolio Overview")
    if data_dict:
        rows = []
        for t, df in data_dict.items():
            if len(df) > 1:
                latest = df.iloc[-1]
                prev = df.iloc[-2]
                change = ((latest['Close']/prev['Close'])-1)*100
                rsi = latest.get('RSI_14', 50)
                if rsi > 70: signal = "🔴 SELL"
                elif rsi < 30: signal = "🟢 STRONG BUY"
                else: signal = "🟡 HOLD"
                rows.append([t, latest['Close'], change, rsi, signal])
        
        comparison_df = pd.DataFrame(rows, columns=["Ticker", "Price", "% Change", "RSI", "Signal"])
        st.dataframe(comparison_df.style.format({"Price": "${:.2f}", "% Change": "{:.1f}%"}), use_container_width=True)

        st.subheader("Correlation & Risk Heatmap")
        prices = pd.DataFrame({t: df['Close'] for t, df in data_dict.items() if not df.empty})
        if len(prices.columns) > 1:
            corr = prices.corr()
            fig = ff.create_annotated_heatmap(z=corr.values, x=list(corr.columns), y=list(corr.index), annotation_text=corr.round(2).values, colorscale="RdBu", showscale=True)
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Technical Indicators")
    ticker_choice = st.selectbox("Choose ticker", tickers)
    if ticker_choice in data_dict and len(data_dict[ticker_choice]) > 0:
        st.dataframe(data_dict[ticker_choice][['Close', 'SMA_50', 'RSI_14', 'MACD', 'BB_upper', 'BB_lower']].tail(10), use_container_width=True)

with tab3:
    st.subheader("💬 Groq LLM Insights + Sentiment")
    ticker_choice = st.selectbox("Choose ticker for sentiment", tickers, key="sentiment")
    if st.button("🔥 Run Full AI Analysis (Groq)"):
        with st.spinner("Analyzing recent news with Groq..."):
            sentiment = get_groq_sentiment(ticker_choice)
            st.markdown(sentiment)

with tab4:
    st.subheader("📈 Advanced Backtesting")
    ticker_choice = st.selectbox("Choose ticker to backtest", tickers, key="backtest")
    if st.button("Run Advanced Backtest"):
        result = run_backtest(data_dict[ticker_choice])
        if result:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Strategy Total Return", f"{result['total_return']:.1%}", delta=f"{result['total_return'] - result['buyhold_return']:.1%} vs Buy&Hold")
            col2.metric("Buy & Hold Return", f"{result['buyhold_return']:.1%}")
            col3.metric("Sharpe Ratio", f"{result['sharpe']:.2f}")
            col4.metric("Max Drawdown", f"{result['max_dd']:.1%}")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=result['equity_curve'].index, y=result['equity_curve']['Cumulative_Strategy'], name="Strategy", line=dict(color="green", width=3)))
            fig.add_trace(go.Scatter(x=result['equity_curve'].index, y=result['equity_curve']['Cumulative_BuyHold'], name="Buy & Hold", line=dict(color="gray", width=2)))
            fig.update_layout(title=f"Equity Curve for {ticker_choice} — Strategy vs Buy & Hold", height=500)
            st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.subheader("💬 Chat with Your Portfolio")
    st.caption("Ask anything about your loaded tickers — powered by Groq (near-instant)")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about your portfolio (e.g. 'Why is SNX flashing a SELL signal?' or 'What is the heaviest sector exposure?')"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        context = "Current portfolio tickers: " + ", ".join(tickers) + "\n"
        for t, df in data_dict.items():
            if len(df) > 1:
                latest = df.iloc[-1]
                change = ((latest['Close']/df.iloc[-2]['Close'])-1)*100
                context += f"{t}: Price ${latest['Close']:.2f}, % Change {change:.1f}%, RSI {latest.get('RSI_14', 50):.1f}\n"

        full_prompt = f"""You have access to the following real-time portfolio data:
{context}

User question: {prompt}

Answer clearly, professionally, and data-driven."""

        with st.spinner("Thinking..."):
            try:
                completion = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": full_prompt}],
                    temperature=0.3,
                    max_tokens=500
                )
                response = completion.choices[0].message.content
            except Exception as e:
                response = f"Groq error: {e}"

        st.session_state.chat_history.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

st.sidebar.success(f"✅ Loaded {len(tickers)} tickers")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
