"""
HemoSmart - Demand Forecast Backtest
----------------------------------------
How good is the Prophet forecast the Planner agent relies on, compared
with what a blood bank would do without it?

Rolling-origin backtest on data/component_demand.csv (per component,
hospital-wide): at each origin every method sees only the history up to
that day and forecasts the next 7 days (the Planner's horizon). Origins
are every 14 days through the second half of 2025, which includes the
post-monsoon dengue platelet spike.

Methods:
  seasonal naive     each day = the same weekday last week
  weekday mean       each day = mean of that weekday over the last 4 weeks
                     (a typical manual blood-bank estimate)
  Prophet            the deployed model (weekly + yearly seasonality)
  LSTM               a small PyTorch LSTM, retrained at each origin
                     (28-day window + calendar features -> next 7 days)
  noise floor        the true expected demand from the generating
                     process (data/demand_process.py). Demand is Poisson
                     around it, so even a perfect forecast has this error.

Run (from the repo root):
    python -m evaluation.forecast_backtest

Writes evaluation/results/forecast_backtest.md and .json.
"""

import json
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd

from data.demand_process import expected_demand

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "evaluation", "results")
HORIZON = 7
ORIGINS = pd.date_range("2025-06-01", "2025-12-20", freq="14D")
DENGUE = (pd.Timestamp("2025-08-15"), pd.Timestamp("2025-10-15"))
METHODS = ("seasonal_naive", "weekday_mean", "prophet", "lstm", "noise_floor")
METHOD_LABELS = {
    "seasonal_naive": "Seasonal naive (last week)",
    "weekday_mean": "4-week weekday mean",
    "prophet": "Prophet (HemoSmart)",
    "lstm": "LSTM",
    "noise_floor": "Noise floor (true mean)",
}

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)


# ----------------------------------------------------------------------
# Methods: history (pd.Series indexed by date) -> next HORIZON days
# ----------------------------------------------------------------------
def seasonal_naive(history: pd.Series) -> np.ndarray:
    return np.array([history.iloc[-7 + (h % 7)] for h in range(HORIZON)], dtype=float)


def weekday_mean(history: pd.Series) -> np.ndarray:
    last = history.iloc[-28:]
    targets = pd.date_range(history.index[-1] + pd.Timedelta(days=1), periods=HORIZON)
    return np.array([last[last.index.dayofweek == d.dayofweek].mean() for d in targets])


def prophet_forecast(history: pd.Series, with_interval: bool = False):
    """Point forecast; with_interval=True also returns Prophet's default
    80% interval (yhat_lower, yhat_upper) -- the band the dashboard shows."""
    from models.train_prophet import _new_model
    model = _new_model().fit(pd.DataFrame({"ds": history.index, "y": history.values}))
    future = pd.DataFrame({"ds": pd.date_range(history.index[-1] + pd.Timedelta(days=1), periods=HORIZON)})
    fc = model.predict(future)
    point = np.clip(fc["yhat"].values, 0, None)
    if with_interval:
        return point, np.clip(fc["yhat_lower"].values, 0, None), np.clip(fc["yhat_upper"].values, 0, None)
    return point


WINDOW = 28


def _calendar(dates: pd.DatetimeIndex) -> np.ndarray:
    dow = 2 * np.pi * dates.dayofweek / 7
    doy = 2 * np.pi * dates.dayofyear / 365.25
    return np.stack([np.sin(dow), np.cos(dow), np.sin(doy), np.cos(doy)], axis=1)


def lstm_forecast(history: pd.Series, seed: int = 0) -> np.ndarray:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    mean, std = history.mean(), history.std() or 1.0
    y = ((history.values - mean) / std).astype(np.float32)
    cal = _calendar(history.index).astype(np.float32)
    future_cal = _calendar(pd.date_range(history.index[-1] + pd.Timedelta(days=1), periods=HORIZON)).astype(np.float32)
    all_cal = np.concatenate([cal, future_cal])

    def window(end):  # inputs for a forecast made at index `end` (exclusive)
        x = np.concatenate([y[end - WINDOW:end, None], cal[end - WINDOW:end]], axis=1)
        return x, all_cal[end:end + HORIZON].reshape(-1)

    xs, cs, ts = [], [], []
    for end in range(WINDOW, len(y) - HORIZON + 1):
        x, c = window(end)
        xs.append(x), cs.append(c), ts.append(y[end:end + HORIZON])
    xs, cs, ts = map(lambda a: torch.tensor(np.array(a)), (xs, cs, ts))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(5, 32, batch_first=True)
            self.head = nn.Sequential(nn.Linear(32 + 4 * HORIZON, 64), nn.ReLU(), nn.Linear(64, HORIZON))

        def forward(self, x, c):
            _, (h, _) = self.lstm(x)
            return self.head(torch.cat([h[-1], c], dim=1))

    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=0.005)
    for _ in range(150):
        opt.zero_grad()
        loss = nn.functional.mse_loss(net(xs, cs), ts)
        loss.backward()
        opt.step()

    x, c = window(len(y))
    with torch.no_grad():
        out = net(torch.tensor(x)[None], torch.tensor(c)[None])[0].numpy()
    return np.clip(out * std + mean, 0, None)


def noise_floor(history: pd.Series, component: str) -> np.ndarray:
    targets = pd.date_range(history.index[-1] + pd.Timedelta(days=1), periods=HORIZON)
    return expected_demand(targets)[component]


# ----------------------------------------------------------------------
# Backtest
# ----------------------------------------------------------------------
def run() -> dict:
    data = pd.read_csv(os.path.join(REPO_ROOT, "data", "component_demand.csv"), parse_dates=["ds"])
    totals = data.groupby(["component", "ds"])["units"].sum()
    errors = []  # one row per (component, origin, method, horizon day)
    coverage = []  # one row per (component, origin, horizon day): actual inside Prophet's 80% band?
    for component in totals.index.get_level_values(0).unique():
        series = totals.loc[component].astype(float)
        for origin in ORIGINS:
            history = series[series.index <= origin]
            actual = series[(series.index > origin)].iloc[:HORIZON].values
            if len(actual) < HORIZON:
                continue
            prophet_point, lower, upper = prophet_forecast(history, with_interval=True)
            for h in range(HORIZON):
                day = origin + pd.Timedelta(days=h + 1)
                coverage.append({"component": component, "dengue": bool(DENGUE[0] <= day <= DENGUE[1]),
                                 "inside": bool(lower[h] <= actual[h] <= upper[h]),
                                 "width": float(upper[h] - lower[h])})
            preds = {
                "seasonal_naive": seasonal_naive(history),
                "weekday_mean": weekday_mean(history),
                "prophet": prophet_point,
                "lstm": lstm_forecast(history),
                "noise_floor": noise_floor(history, component),
            }
            for m, p in preds.items():
                for h in range(HORIZON):
                    day = origin + pd.Timedelta(days=h + 1)
                    errors.append({"component": component, "origin": str(origin.date()), "method": m,
                                   "dengue": bool(DENGUE[0] <= day <= DENGUE[1]),
                                   "abs_err": abs(p[h] - actual[h]), "sq_err": (p[h] - actual[h]) ** 2,
                                   "signed_err": p[h] - actual[h]})
            print(f"{component:<9} {origin.date()} done", flush=True)

    df = pd.DataFrame(errors)

    def score(frame):
        g = frame.groupby(["component", "method"])
        return pd.DataFrame({"mae": g.abs_err.mean(), "rmse": np.sqrt(g.sq_err.mean()), "bias": g.signed_err.mean()})

    overall, dengue = score(df), score(df[df.dengue])
    to_dict = lambda s: {c: {m: {k: round(float(s.loc[(c, m), k]), 3) for k in s.columns}  # noqa: E731
                             for m in METHODS} for c in s.index.get_level_values(0).unique()}
    cov = pd.DataFrame(coverage)
    interval = {
        c: {"coverage": round(float(g.inside.mean()), 3),
            "coverage_dengue": round(float(g[g.dengue].inside.mean()), 3),
            "mean_width": round(float(g.width.mean()), 2)}
        for c, g in cov.groupby("component")
    }
    return {"overall": to_dict(overall), "dengue_window": to_dict(dengue), "interval": interval,
            "origins": [str(o.date()) for o in ORIGINS]}


def to_markdown(result: dict, meta: dict) -> str:
    lines = [
        "# HemoSmart — Demand Forecast Backtest",
        "",
        f"Rolling origin, {len(result['origins'])} origins every 14 days from {result['origins'][0]}, "
        f"{HORIZON}-day horizon (the Planner's), hospital-wide units/day per component. "
        f"Generated by `python -m evaluation.forecast_backtest` on {meta['generated_at']}.",
        "",
    ]
    for title, key in (("All origins", "overall"),
                       (f"Dengue window only ({DENGUE[0].date()} – {DENGUE[1].date()})", "dengue_window")):
        lines += [f"## {title}", "",
                  "| Component | " + " | ".join(METHOD_LABELS[m] for m in METHODS) + " |",
                  "|---|" + "---|" * len(METHODS)]
        for component, row in result[key].items():
            lines.append(f"| {component} MAE | " + " | ".join(f"{row[m]['mae']:.2f}" for m in METHODS) + " |")
        for component, row in result[key].items():
            lines.append(f"| {component} bias | " + " | ".join(f"{row[m]['bias']:+.2f}" for m in METHODS) + " |")
        lines.append("")
    lines += ["## Prophet's 80% interval — is the dashboard's band honest?", "",
              "| Component | Coverage (target 80%) | Coverage, dengue window | Mean band width (units/day) |",
              "|---|---|---|---|"]
    for c, r in result["interval"].items():
        lines.append(f"| {c} | {r['coverage'] * 100:.0f}% | {r['coverage_dengue'] * 100:.0f}% | {r['mean_width']:.1f} |")
    lines += ["", "Coverage = share of actual days that fell inside the band. Near 80% means the band can be read "
              "as stated; well below means it is too narrow.", ""]
    lines += [
        "MAE in units/day (lower is better). Bias = mean(forecast − actual): negative means under-forecasting, "
        "which for blood means a late donor call. The noise floor is the error of the true expected demand — "
        "Poisson variation no forecast can remove.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    result = run()
    meta = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"), "horizon": HORIZON}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "forecast_backtest.json"), "w") as f:
        json.dump({"meta": meta, **result}, f, indent=2)
    md = to_markdown(result, meta)
    with open(os.path.join(RESULTS_DIR, "forecast_backtest.md"), "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
