"""
HemoSmart - Standalone LSTM Forecast Script
------------------------------------------------
Runs in its OWN separate virtual environment (see setup instructions
below), completely isolated from the main HemoSmart/CrewAI
environment. This exists because TensorFlow's installed build
requires an older NumPy/protobuf combination than CrewAI's
dependencies require -- both cannot be reliably satisfied in one
environment. Isolating them removes the conflict entirely rather than
fighting version pins.

This script does ONE thing: load the trained LSTM model, run the
rolling-window forecast, and print the result as JSON to stdout. The
main environment calls this script as a subprocess and reads its
stdout -- it never imports TensorFlow directly.

SETUP (run once):
    python -m venv lstm_env
    lstm_env\\Scripts\\activate          (Windows)
    pip install tensorflow scikit-learn pandas numpy

USAGE (called automatically by agents/tools.py, or manually):
    lstm_env\\Scripts\\python.exe agents\\lstm_forecast_standalone.py --days 7

OUTPUT: a single line of JSON on stdout, e.g.:
    {"forecast_days": 7, "model_used": "LSTM", "predicted_units_per_day": [...], "average_daily_demand": 32.1}
or, on any failure:
    {"error": "..."}
"""

import sys
import os
import json
import argparse
import pickle

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
LSTM_SEQ_LENGTH = 30


def run_lstm_forecast(days: int) -> dict:
    lstm_path = os.path.join(MODEL_DIR, "lstm_model.h5")
    scaler_path = os.path.join(MODEL_DIR, "lstm_scaler.pkl")
    data_path = os.path.join(DATA_DIR, "blood_demand.csv")

    for path, name in [(lstm_path, "lstm_model.h5"), (scaler_path, "lstm_scaler.pkl"), (data_path, "blood_demand.csv")]:
        if not os.path.exists(path):
            return {"error": f"Required file not found: {name} (expected at {path})"}

    try:
        import pandas as pd
        import numpy as np
        from tensorflow.keras.models import load_model

        model = load_model(lstm_path, compile=False)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        df = pd.read_csv(data_path)
        df["ds"] = pd.to_datetime(df["ds"])
        last_date = df["ds"].max()

        scaled_data = scaler.transform(df[["y"]].values)
        window = scaled_data[-LSTM_SEQ_LENGTH:].reshape(1, LSTM_SEQ_LENGTH, 1)

        predictions_scaled = []
        current_window = window.copy()
        for _ in range(days):
            next_scaled = model.predict(current_window, verbose=0)[0][0]
            predictions_scaled.append(next_scaled)
            current_window = np.append(current_window[:, 1:, :], [[[next_scaled]]], axis=1)

        predictions_scaled = np.array(predictions_scaled).reshape(-1, 1)
        predictions_actual = scaler.inverse_transform(predictions_scaled).flatten()
        dates = [last_date + pd.Timedelta(days=i + 1) for i in range(days)]

        return {
            "forecast_days": days,
            "model_used": "LSTM",
            "predicted_units_per_day": [
                {"date": str(d.date()), "predicted_units": round(float(p), 1)}
                for d, p in zip(dates, predictions_actual)
            ],
            "average_daily_demand": round(float(predictions_actual.mean()), 1),
        }
    except Exception as exc:
        return {"error": f"LSTM forecast failed: {exc.__class__.__name__}: {exc}"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7)
    args = parser.parse_args()

    result = run_lstm_forecast(args.days)
    # Single line of JSON on stdout -- this is the ONLY thing the
    # calling process should read from stdout, so no other print()
    # statements belong in this script.
    print(json.dumps(result))