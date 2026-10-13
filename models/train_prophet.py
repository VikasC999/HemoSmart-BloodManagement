"""
HemoSmart - Blood Demand Forecasting (Prophet, per component)
-----------------------------------------------------------------
Trains one Prophet model per blood component on the simulated daily
demand history (data/component_demand.csv), then disaggregates each
component's forecast across blood groups by their historical share --
a top-down hierarchical forecast.

Why top-down rather than 24 separate models (8 groups x 3 components):
the rare groups (AB-, A-) see a fraction of a unit per day, which is
mostly zeros -- too sparse for a per-series model to learn seasonality
from. The component totals carry the real signal (weekday surgery
pattern, the post-monsoon dengue platelet spike), and group shares are
stable, so the split is where the structure actually is.

Runs entirely on CPU. Requires: data/component_demand.csv
(run data/create_dataset.py first)

Run:
    python models/train_prophet.py

Outputs:
    models/prophet_components.pkl   -> {component: fitted Prophet model}
    models/demand_profile.json      -> group shares + mean daily demand per
                                       group/component (par levels)
    models/prophet_metrics.txt      -> MAE / RMSE on a 30-day holdout
    models/prophet_forecast_plot.png
"""

import json
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

HOLDOUT_DAYS = 30
SHARE_WINDOW_DAYS = 365


def _new_model() -> Prophet:
    return Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False)


def main() -> None:
    data = pd.read_csv("data/component_demand.csv", parse_dates=["ds"])
    print("Loaded component demand:", data.shape)

    totals = data.groupby(["component", "ds"])["units"].sum().reset_index()
    models, metrics, profile = {}, [], {"group_share": {}, "mean_daily": {}}

    fig, axes = plt.subplots(len(totals["component"].unique()), 1, figsize=(10, 9), sharex=True)
    for ax, (component, series) in zip(axes, totals.groupby("component")):
        series = series[["ds", "units"]].rename(columns={"units": "y"}).sort_values("ds")

        # Holdout evaluation: train on all but the last 30 days
        train, test = series.iloc[:-HOLDOUT_DAYS], series.iloc[-HOLDOUT_DAYS:]
        holdout_model = _new_model().fit(train)
        predicted = holdout_model.predict(test[["ds"]])["yhat"].values
        mae = mean_absolute_error(test["y"], predicted)
        rmse = np.sqrt(mean_squared_error(test["y"], predicted))
        metrics.append(f"{component}: MAE {mae:.2f}, RMSE {rmse:.2f} (units/day, {HOLDOUT_DAYS}-day holdout)")
        print(metrics[-1])

        # Production model: refit on everything
        model = _new_model().fit(series)
        models[component] = model

        fit = model.predict(series[["ds"]])
        ax.plot(series["ds"], series["y"], ".", ms=2, alpha=0.4, label="actual")
        ax.plot(fit["ds"], fit["yhat"], lw=1, label="Prophet fit")
        ax.set_title(f"{component} daily demand")
        ax.legend(loc="upper left")

        recent = data[(data["component"] == component)
                      & (data["ds"] > data["ds"].max() - pd.Timedelta(days=SHARE_WINDOW_DAYS))]
        by_group = recent.groupby("blood_group")["units"].sum()
        profile["group_share"][component] = (by_group / by_group.sum()).round(5).to_dict()
        profile["mean_daily"][component] = (by_group / SHARE_WINDOW_DAYS).round(3).to_dict()

    plt.tight_layout()
    plt.savefig("models/prophet_forecast_plot.png", dpi=150, bbox_inches="tight")
    plt.close()

    profile["trained_through"] = str(data["ds"].max().date())

    with open("models/prophet_metrics.txt", "w") as f:
        f.write("\n".join(metrics) + "\n")
    with open("models/demand_profile.json", "w") as f:
        json.dump(profile, f, indent=2)
    with open("models/prophet_components.pkl", "wb") as f:
        pickle.dump(models, f)

    print("\nSaved models/prophet_components.pkl and models/demand_profile.json")


if __name__ == "__main__":
    main()
