from fastapi import FastAPI
import joblib
import numpy as np
import requests
import pandas as pd
from feature_engineering import add_features

app = FastAPI()

model = joblib.load("ai_signal_model.json")

FEATURE_COLUMNS = [
    "rsi",
    "macd",
    "macd_signal",
    "macd_hist",
    "price_above_ema50",
    "ema_trend"
]


def fetch_live_data():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"

    params = {
        "vs_currency": "usd",
        "days": 7,
        "interval": "hourly"
    }

    response = requests.get(url, params=params)
    data = response.json()

    prices = data["prices"]
    volumes = data["total_volumes"]

    df = pd.DataFrame(prices, columns=["timestamp", "price"])
    df["volume"] = [v[1] for v in volumes]

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    df = add_features(df)

    return df


@app.get("/")
def home():
    return {"message": "AI Signal Engine Running 🚀"}


@app.get("/signal")
def get_signal():
    df = fetch_live_data()
    latest = df.iloc[-1]

    X_live = np.array([latest[FEATURE_COLUMNS]])

    probabilities = model.predict_proba(X_live)[0]
    classes = model.classes_

    prob_dict = dict(zip(classes, probabilities))

    buy_prob = prob_dict.get(1, 0)
    sell_prob = prob_dict.get(-1, 0)

    if buy_prob > 0.65:
        signal = "BUY"
        confidence = buy_prob
    elif sell_prob > 0.65:
        signal = "SELL"
        confidence = sell_prob
    else:
        signal = "NO TRADE"
        confidence = max(buy_prob, sell_prob)

    return {
        "signal": signal,
        "confidence": round(float(confidence), 3),
        "buy_probability": round(float(buy_prob), 3),
        "sell_probability": round(float(sell_prob), 3)
    }
