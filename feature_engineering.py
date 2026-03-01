import ta
import numpy as np

def add_features(df):
    # RSI
    df["rsi"] = ta.momentum.RSIIndicator(df["price"], window=14).rsi()

    # EMA
    df["ema50"] = ta.trend.EMAIndicator(df["price"], window=50).ema_indicator()
    df["ema200"] = ta.trend.EMAIndicator(df["price"], window=200).ema_indicator()

    # MACD
    macd = ta.trend.MACD(df["price"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    # Engineered features (stronger than raw values)
    df["price_above_ema50"] = (df["price"] > df["ema50"]).astype(int)
    df["ema_trend"] = (df["ema50"] > df["ema200"]).astype(int)

    df.dropna(inplace=True)
    return df


def create_target(df):
    # 3-hour forward return
    df["future_return"] = (df["price"].shift(-3) - df["price"]) / df["price"]

    df["target"] = 0
    df.loc[df["future_return"] > 0.004, "target"] = 1     # +0.4%
    df.loc[df["future_return"] < -0.004, "target"] = -1   # -0.4%

    df.dropna(inplace=True)
    return df
