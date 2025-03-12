import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import ccxt
import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import logging
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1. 자동 새로고침 (15초 간격)
st_autorefresh(interval=15000, limit=10000, key="realtime_refresh")

# 2. Coindesk 뉴스 스크래핑 및 감성 분석
def fetch_coindesk_headlines():
    url = "https://www.coindesk.com/price/bitcoin"
    headlines = []
    try:
        resp = requests.get(url)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            # Coindesk 페이지에서 <h3> 태그를 찾아 헤드라인 추출 (구조 변경 시 수정 필요)
            for h in soup.find_all("h3"):
                text = h.get_text(strip=True)
                if text:
                    headlines.append(text)
        else:
            logging.error(f"Coindesk 응답 코드: {resp.status_code}")
    except Exception as e:
        logging.error(f"Coindesk 뉴스 스크래핑 오류: {e}")
    return headlines

def analyze_news_sentiment(headlines):
    if not headlines:
        return 0
    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(headline)["compound"] for headline in headlines]
    return np.mean(scores)

# 3. Investtech 신호 스크래핑  
def fetch_investtech_signal():
    url = "https://www.investtech.com/main/market.php?CompanyID=99400001&product=241"
    try:
        resp = requests.get(url)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            # Investtech 페이지에서 신호를 포함한 요소를 추출 (실제 구조에 맞게 수정 필요)
            # 예시로 id="tradeSignal"인 div 내 텍스트를 사용한다고 가정
            signal_tag = soup.find("div", {"id": "tradeSignal"})
            if signal_tag:
                text = signal_tag.get_text(strip=True)
                if "롱" in text or "매수" in text:
                    return 1, text
                elif "숏" in text or "매도" in text:
                    return -1, text
                else:
                    return 0, text
            else:
                return 0, "신호 없음"
        else:
            logging.error(f"Investtech 응답 코드: {resp.status_code}")
            return 0, "응답 오류"
    except Exception as e:
        logging.error(f"Investtech 신호 스크래핑 오류: {e}")
        return 0, "오류"

# 4. Binance OHLCV 데이터 가져오기 (ccxt 사용)
def fetch_ohlcv(symbol="BTC/USDT", timeframe="1m", limit=300):
    exchange = ccxt.binance({"enableRateLimit": True})
    try:
        data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    except Exception as e:
        st.error(f"데이터 불러오기 오류: {e}")
        return None

# 5. 지표 계산: MA50, MA200, MACD
def add_indicators(df):
    df["MA50"] = df["close"].rolling(50).mean()
    df["MA200"] = df["close"].rolling(200).mean()
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
    return df

# 6. 단일 시간대 신호 판단 (예시 로직)
def get_signal(df):
    if pd.isna(df["MA50"].iloc[-1]) or pd.isna(df["MA200"].iloc[-1]):
        return 0
    ma50 = df["MA50"].iloc[-1]
    ma200 = df["MA200"].iloc[-1]
    macd_hist = df["MACD_hist"].iloc[-1]
    if ma50 > ma200 and macd_hist > 0:
        return 1
    elif ma50 < ma200 and macd_hist < 0:
        return -1
    else:
        return 0

# 7. 위험 관리: 진입가, 손절가, 익절가 계산 (1분봉 기준)
def compute_risk(entry_price, position, risk_factor=0.02):
    if position == "롱":
        stop_loss = entry_price * (1 - risk_factor)
        take_profit = entry_price * (1 + risk_factor * 2)
    elif position == "숏":
        stop_loss = entry_price * (1 + risk_factor)
        take_profit = entry_price * (1 - risk_factor * 2)
    else:
        stop_loss, take_profit = None, None
    return stop_loss, take_profit

# 8. 차트 생성 (Plotly)
def create_candle_chart(df, title):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["timestamp"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="캔들"
    ))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["MA50"],
                             mode="lines", name="MA50", line=dict(color="orange", width=1.5)))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["MA200"],
                             mode="lines", name="MA200", line=dict(color="blue", width=1.5)))
    fig.update_layout(title=title,
                      xaxis_title="시간",
                      yaxis_title="가격 (USDT)",
                      xaxis_rangeslider_visible=False,
                      height=600)
    return fig

def create_macd_chart(df, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["MACD"],
                             mode="lines", name="MACD", line=dict(color="purple", width=2)))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["MACD_signal"],
                             mode="lines", name="Signal", line=dict(color="orange", width=2)))
    fig.add_trace(go.Bar(x=df["timestamp"], y=df["MACD_hist"],
                         name="MACD 히스토그램", marker=dict(color="green")))
    fig.update_layout(title=title,
                      xaxis_title="시간",
                      yaxis_title="지표값",
                      height=400)
    return fig

# 9. 여러 시간대 신호 취합 및 종합 추천 (기술적 신호 + 뉴스 + Investtech)
def aggregate_signals(timeframes):
    signals = {}
    latest_prices = {}
    for tf in timeframes:
        df = fetch_ohlcv("BTC/USDT", timeframe=tf, limit=300)
        if df is not None:
            df = add_indicators(df)
            sig = get_signal(df)
            signals[tf] = sig
            latest_prices[tf] = df["close"].iloc[-1]
        else:
            signals[tf] = 0
            latest_prices[tf] = None
    return signals, latest_prices

# ----- Streamlit UI 구성 -----
st.title("실시간 비트코인 차트 & 뉴스/Investtech 기반 롱/숏 추천")
st.markdown("""
이 앱은 Binance의 BTC/USDT 데이터를 여러 시간대(1분, 15분, 30분, 1시간, 4시간, 1일)로  
실시간(15초 간격)으로 업데이트하며,  
Coindesk의 최신 뉴스 감성과 Investtech 차트 신호를 반영하여  
최종 롱/숏 진입 시점, 손절가, 익절가를 추천합니다.
""")

# --- (1) 뉴스 감성 분석 (Coindesk) ---
st.subheader("1. 최신 뉴스 감성 (Coindesk)")
news_headlines = fetch_coindesk_headlines()
news_sentiment = analyze_news_sentiment(news_headlines)
if news_sentiment > 0.05:
    news_influence = 1
    news_text = "긍정적 뉴스 → 롱 우세"
elif news_sentiment < -0.05:
    news_influence = -1
    news_text = "부정적 뉴스 → 숏 우세"
else:
    news_influence = 0
    news_text = "중립 뉴스 → 특별 추천 없음"
st.write(f"평균 감성 점수: {news_sentiment:.2f} / {news_text}")
if news_headlines:
    st.write("최신 뉴스 헤드라인 (일부):")
    for head in news_headlines[:5]:
        st.write(f"- {head}")
else:
    st.write("뉴스 데이터를 불러올 수 없습니다.")

# --- (2) Investtech 신호 스크래핑 ---
st.subheader("2. Investtech 차트 신호")
investtech_signal, investtech_text = fetch_investtech_signal()
if investtech_signal == 1:
    investtech_influence = 1
    investtech_msg = f"Investtech: 롱 신호 ({investtech_text})"
elif investtech_signal == -1:
    investtech_influence = -1
    investtech_msg = f"Investtech: 숏 신호 ({investtech_text})"
else:
    investtech_influence = 0
    investtech_msg = f"Investtech: 신호 없음 또는 중립 ({investtech_text})"
st.write(investtech_msg)

# --- (3) 실시간 Binance 차트 (여러 시간대) ---
st.subheader("3. 실시간 Binance 차트 (여러 시간대)")
timeframes = ["1m", "15m", "30m", "1h", "4h", "1d"]
tabs = st.tabs(["1분", "15분", "30분", "1시간", "4시간", "1일"])
tech_signals = {}
for i, tf in enumerate(timeframes):
    with tabs[i]:
        df_tf = fetch_ohlcv("BTC/USDT", timeframe=tf, limit=300)
        if df_tf is not None:
            df_tf = add_indicators(df_tf)
            candle_fig = create_candle_chart(df_tf, title=f"BTC/USDT {tf} 차트")
            st.plotly_chart(candle_fig, use_container_width=True)
            macd_fig = create_macd_chart(df_tf, title=f"BTC/USDT {tf} MACD")
            st.plotly_chart(macd_fig, use_container_width=True)
            current_tf_price = df_tf["close"].iloc[-1]
            sig = get_signal(df_tf)
            tech_signals[tf] = sig
            if sig == 1:
                st.write(f"**[{tf}] 신호:** 롱 / 가격: {current_tf_price:.2f} USDT")
            elif sig == -1:
                st.write(f"**[{tf}] 신호:** 숏 / 가격: {current_tf_price:.2f} USDT")
            else:
                st.write(f"**[{tf}] 신호:** 중립 / 가격: {current_tf_price:.2f} USDT")
        else:
            st.write(f"[{tf}] 데이터를 불러오지 못했습니다.")

# --- (4) 종합 신호 및 최종 추천 ---
st.subheader("4. 종합 롱/숏 추천 및 위험 관리")
agg_signals, latest_prices = aggregate_signals(timeframes)
st.write("각 시간대 기술적 신호:", agg_signals)

# 단순 다수결로 기술적 신호 합산
tech_total = sum(agg_signals.values())
# 최종 점수: 기술적 신호 + 뉴스 감성 + Investtech 신호 (각각 가중치 1로 적용)
final_score = tech_total + news_influence + investtech_influence
if final_score > 0:
    final_signal = "롱"
elif final_score < 0:
    final_signal = "숏"
else:
    final_signal = "중립"
st.write("최종 추천 신호 (종합):", final_signal)

# 1분봉 기준 진입 가격 및 위험 관리 (진입가, 손절가, 익절가)
df_1m = fetch_ohlcv("BTC/USDT", timeframe="1m", limit=300)
if df_1m is not None:
    df_1m = add_indicators(df_1m)
    entry_price = df_1m["close"].iloc[-1]
    stop_loss, take_profit = compute_risk(entry_price, final_signal, risk_factor=0.02)
    st.write(f"현재 가격 (1분봉 기준): {entry_price:.2f} USDT")
    if final_signal != "중립":
        st.write(f"예상 진입 포지션: {final_signal}")
        st.write(f"예상 손절가: {stop_loss:.2f} USDT")
        st.write(f"예상 익절가: {take_profit:.2f} USDT")
    else:
        st.write("특별한 진입 신호가 없습니다.")
else:
    st.write("1분봉 데이터를 불러오지 못했습니다.")

st.info("이 페이지는 15초마다 자동 새로고침됩니다. (API 호출 제한에 유의)")
