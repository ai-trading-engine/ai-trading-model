import requests
import pandas as pd
import joblib
from xgboost import XGBClassifier
from feature_engineering import add_features, create_target

def fetch_coingecko_data(coin_id="bitcoin", days=180):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"

    params = {
        "vs_currency": "usd",
        "days": days,
        "interval": "hourly"
    }

    response = requests.get(url, params=params)
    data = response.json()

    prices = data["prices"]
    volumes = data["total_volumes"]

    df = pd.DataFrame(prices, columns=["timestamp", "price"])
    df["volume"] = [v[1] for v in volumes]

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    return df


# Fetch data
df = fetch_coingecko_data("bitcoin", 180)

# Add features
df = add_features(df)

# Create target
df = create_target(df)

# Remove neutral trades
df = df[df["target"] != 0]

features = [
    "rsi",
    "macd",
    "macd_signal",
    "macd_hist",
    "price_above_ema50",
    "ema_trend"
]

X = df[features]
y = df["target"]

model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    use_label_encoder=False,
    eval_metric="mlogloss"
)

model.fit(X, y)

joblib.dump(model, "ai_signal_model.json")

print("✅ Model trained and saved.")
