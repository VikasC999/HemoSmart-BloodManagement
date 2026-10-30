"""
HemoSmart - Report Figures
------------------------------
Static PNGs for the Phase II report, drawn from the evaluation results
(evaluation/results/*.json), so every figure matches a table exactly.
The forecast figure refits Prophet itself (a few fits, ~20 s).

Run (from the repo root, after the evaluation scripts):
    python -m evaluation.figures

Writes docs/figures/fig_*.png.

Colour: the validated reference palette (dataviz skill), slots in fixed
order -- blue, orange, aqua -- with neutral grey for baselines and
reference marks. Lighter slots always carry direct labels.
"""

import json
import logging
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO_ROOT, "evaluation", "results")
OUT = os.path.join(REPO_ROOT, "docs", "figures")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#8a8984"      # baselines / reference marks
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_SOFT = "#cde2fb"

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 9,
        "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.edgecolor": GRID, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False, "legend.fontsize": 8,
    })


def _load(name):
    with open(os.path.join(RESULTS, f"{name}.json")) as f:
        return json.load(f)


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{name}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.relpath(path, REPO_ROOT))


def _bars(ax, x, heights, color, width, bottom=None, labels=True, fmt="{:.0f}", err=None):
    bars = ax.bar(x, heights, width, bottom=bottom, color=color, edgecolor=SURFACE, linewidth=2,
                  yerr=err, error_kw={"ecolor": INK_2, "elinewidth": 0.8, "capsize": 2})
    if labels:
        for b, h, base in zip(bars, heights, bottom if bottom is not None else [0] * len(heights)):
            if h > 0:
                ax.text(b.get_x() + b.get_width() / 2, base + h / 2, fmt.format(h), ha="center", va="center",
                        fontsize=8, color=INK if color in (AQUA, BLUE_SOFT) else "white")
    return bars


# ----------------------------------------------------------------------
def fig_simulation():
    s = _load("simulation")["summary"]
    policies = ["baseline", "allocation", "agents", "hemosmart"]
    names = ["Manual\nbaseline", "+ wastage-aware\nallocation", "+ autonomous\nagents", "HemoSmart\n(+ redistribution)"]
    x = np.arange(len(policies))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))

    expired = [s[p]["expired"]["mean"] for p in policies]
    moved = [s[p]["transferred"]["mean"] for p in policies]
    _bars(a1, x, expired, BLUE, 0.6, labels=False)
    _bars(a1, x, moved, ORANGE, 0.6, bottom=expired, labels=False)
    for xi, e, m in zip(x, expired, moved):  # segment labels only where a bar is split
        if m > 0:
            a1.text(xi, e / 2, f"{e:.0f}", ha="center", va="center", fontsize=8, color="white")
            a1.text(xi, e + m / 2, f"{m:.0f}", ha="center", va="center", fontsize=8, color="white")
    for xi, e, m in zip(x, expired, moved):
        a1.text(xi, e + m + 1.5, f"{e + m:.0f}", ha="center", va="bottom", fontsize=8, color=INK_2)
    a1.set_title("Units not used on site (90 days)")
    a1.set_ylabel("units, mean of 3 seeds")
    a1.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)],
              labels=["Expired on site", "Redistributed to partner"], loc="upper left")
    a1.set_ylim(0, max(e + m for e, m in zip(expired, moved)) * 1.35)  # headroom for the legend

    short = [s[p]["shortage_units"]["mean"] for p in policies]
    err = [s[p]["shortage_units"]["std"] for p in policies]
    _bars(a2, x, short, INK_2, 0.6, err=err)
    a2.set_title("Units short — unmet demand (90 days)")
    a2.set_ylim(0, max(v + d for v, d in zip(short, err)) * 1.15)
    for ax in (a1, a2):
        ax.set_xticks(x, names, fontsize=8)
        ax.grid(axis="x", visible=False)
    fig.text(0.01, -0.06, "Synthetic 90-day simulation from 2026-08-01, 3 seeds (error bars: ±1 std). "
             "Source: evaluation/results/simulation.md", fontsize=7, color=INK_2)
    fig.tight_layout()
    _save(fig, "fig_simulation")


def fig_sensitivity():
    rows = [r for r in _load("sensitivity")["rows"] if r["alert_cap"] == 16]
    w = [r["platelet_window_days"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for key, color, label in (("shortage_units", BLUE, "Units short"),
                              ("not_used_here", ORANGE, "Units not used here")):
        ys = [r[key] for r in rows]
        ax.plot(w, ys, color=color, lw=2, marker="o", ms=5)
        ax.annotate(label, (w[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK)
    ax.axvline(5, color=NEUTRAL, lw=1, ls="--")
    ax.set_ylim(20, 150)
    ax.text(5.1, 128, "chosen: 5 days\n= platelet shelf life", fontsize=8, color=INK_2, va="top")
    ax.set_title("Platelet outreach window: shortages vs over-collection")
    ax.set_xlabel("outreach window (days ahead a gap triggers donor alerts)")
    ax.set_ylabel("units over 90 days")
    ax.set_xlim(1.7, 8.3)
    ax.grid(axis="x", visible=False)
    fig.text(0.01, -0.04, "Alert cap 16, mean of 3 seeds. Source: evaluation/results/sensitivity.md",
             fontsize=7, color=INK_2)
    _save(fig, "fig_sensitivity")


def fig_prediction():
    s = _load("prediction_benchmark")["summary"]
    targets = ["PRBC", "Platelets", "FFP", "Any"]
    methods = [("static_rule", NEUTRAL, "Static threshold rule"), ("logistic", BLUE, "Logistic regression"),
               ("random_forest", ORANGE, "Random forest"), ("xgboost", AQUA, "XGBoost (HemoSmart)")]
    x = np.arange(len(targets))
    width = 0.19
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    for i, (key, color, label) in enumerate(methods):
        vals = [s[key][t]["f1"]["mean"] for t in targets]
        bars = ax.bar(x + (i - 1.5) * width, vals, width, color=color, edgecolor=SURFACE, linewidth=2, label=label)
        if key in ("static_rule", "xgboost"):  # selective labels: the comparison that matters
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v - 0.006, f"{v:.2f}", ha="center", va="top", fontsize=7,
                        color=INK if color == AQUA else "white", fontweight="bold")
    ceil = [s["ceiling"][t]["f1"]["mean"] for t in targets]
    ax.hlines(ceil, x - 2 * width, x + 2 * width, colors=INK, lw=1.2, linestyles="--", label="Ceiling (best possible)")
    ax.set_xticks(x, ["PRBC", "Platelets", "FFP", "Any transfusion"])
    ax.set_ylim(0.5, 0.95)
    ax.set_ylabel("F1 score (5-fold CV)")
    ax.set_title("Transfusion prediction vs a static threshold rule")
    ax.legend(ncol=3, loc="upper left", fontsize=7.5)
    ax.grid(axis="x", visible=False)
    fig.text(0.01, -0.04, "5,000 synthetic patients; labels are simulated clinician decisions. "
             "Source: evaluation/results/prediction_benchmark.md", fontsize=7, color=INK_2)
    _save(fig, "fig_prediction")


def fig_forecast():
    from evaluation.forecast_backtest import prophet_forecast, weekday_mean

    data = pd.read_csv(os.path.join(REPO_ROOT, "data", "component_demand.csv"), parse_dates=["ds"])
    series = data[data.component == "Platelets"].groupby("ds")["units"].sum().astype(float)
    origins = pd.date_range("2025-07-31", "2025-10-23", freq="7D")
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    for k, origin in enumerate(origins):
        hist = series[series.index <= origin]
        days = pd.date_range(origin + pd.Timedelta(days=1), periods=7)
        point, lo, hi = prophet_forecast(hist, with_interval=True)
        ax.fill_between(days, lo, hi, color=BLUE_SOFT, lw=0, label="Prophet 80% range" if k == 0 else None)
        ax.plot(days, point, color=BLUE, lw=2, label="Prophet 7-day forecast" if k == 0 else None)
        ax.plot(days, weekday_mean(hist), color=ORANGE, lw=1.5, ls="--",
                label="4-week weekday mean (manual)" if k == 0 else None)
    window = series[(series.index > origins[0]) & (series.index <= origins[-1] + pd.Timedelta(days=7))]
    ax.scatter(window.index, window.values, s=10, color=INK_2, zorder=3, label="Actual demand")
    ax.set_ylim(0, 28)
    ax.set_title("Platelet demand through the dengue season: weekly 7-day forecasts")
    ax.set_ylabel("platelet units / day (hospital-wide)")
    ax.legend(loc="upper left", ncol=2)
    ax.grid(axis="x", visible=False)
    fig.autofmt_xdate()
    fig.text(0.01, -0.06, "Each forecast uses only data up to its start date. Synthetic demand. "
             "Accuracy and band coverage: evaluation/results/forecast_backtest.md", fontsize=7, color=INK_2)
    _save(fig, "fig_forecast")


def fig_bandit():
    b = _load("bandit_regret")["fatigue_new"]  # the realistic donor model that decides the policy (D85)
    oracle = np.array(b["oracle"]["mean_curve"])
    days = np.arange(1, len(oracle) + 1)
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    # every policy with the production cooldown, so the comparison is like for like
    series = [("random+cooldown7", NEUTRAL, "Random calls"), ("epsilon_greedy+cooldown7", ORANGE, "ε-greedy"),
              ("greedy+cooldown7", AQUA, "Greedy"), ("thompson+cooldown7", BLUE, "Thompson sampling (HemoSmart)")]
    ends = []
    for key, color, label in series:
        regret = oracle - np.array(b[key]["mean_curve"])
        ax.plot(days, regret, color=color, lw=2, label=label)
        ends.append((regret[-1], label))
    # direct labels at the line ends, nudged apart so close lines stay legible
    ends.sort()
    last = -1e9
    for y, label in ends:
        y_text = max(y, last + 13)
        ax.text(days[-1] + 6, y_text, f"{label}  {y:.0f}", va="center", fontsize=8, color=INK)
        last = y_text
    ax.set_xlim(0, 365)
    ax.set_title("Donor selection: donations lost vs a clairvoyant oracle")
    ax.set_xlabel("day")
    ax.set_ylabel("cumulative regret (donations)")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)
    fig.text(0.01, -0.04, "400 donors + new joiners, contact fatigue, 7-day contact cooldown, 5 alerts/day, 20 seeds. "
             "Source: evaluation/results/bandit_regret.md", fontsize=7, color=INK_2)
    _save(fig, "fig_bandit")


# ----------------------------------------------------------------------
def _box(ax, x, y, w, h, title, body="", fill="#ffffff", edge=GRID, title_color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
                                facecolor=fill, edgecolor=edge, linewidth=1.2))
    ax.text(x + w / 2, y + h - 0.022, title, ha="center", va="top", fontsize=8.5, fontweight="bold",
            color=title_color)
    if body:
        ax.text(x + w / 2, y + h - 0.058, body, ha="center", va="top", fontsize=7, color=INK_2, linespacing=1.35)


def _arrow(ax, a, b, color=INK_2, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=9, color=color, lw=1, linestyle=ls))


def fig_architecture():
    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Clients
    _box(ax, 0.02, 0.86, 0.30, 0.11, "React dashboards", "5 roles · JWT login")
    _box(ax, 0.35, 0.86, 0.30, 0.11, "MCP clients", "any MCP-compatible AI client")
    _box(ax, 0.68, 0.86, 0.30, 0.11, "Hospital systems", "adapter interface · mock partner")

    # API
    _box(ax, 0.02, 0.70, 0.96, 0.11, "FastAPI backend",
         "RBAC on every route · audit log · /api/predict · /api/forecast · /api/cases · /api/autonomy · /mcp")
    for x in (0.17, 0.50, 0.83):
        _arrow(ax, (x, 0.86), (x, 0.81), style="<|-|>")

    # Layer 1: autonomous loop
    ax.add_patch(FancyBboxPatch((0.02, 0.27), 0.55, 0.39, boxstyle="round,pad=0.006,rounding_size=0.015",
                                facecolor="#f3f8fe", edgecolor=BLUE, linewidth=1.2))
    ax.text(0.04, 0.648, "Layer 1 — Autonomous supply loop", fontsize=8.5, fontweight="bold", color=INK, va="top")
    ax.text(0.04, 0.622, "hourly · deterministic Python agents · run in order 1 → 7", fontsize=7, color=INK_2,
            va="top")
    agents = ["1 Donor replies", "2 Inventory", "3 Forecast", "4 Planner",
              "5 Donor outreach", "6 Logistics", "7 Explanation"]
    w, h, gap = 0.122, 0.07, 0.012
    pos = [(0.035 + i * (w + gap), 0.505) for i in range(4)] + [(0.035 + i * (w + gap), 0.41) for i in range(3)]
    for name, (x, y) in zip(agents, pos):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.002,rounding_size=0.01",
                                    facecolor="#ffffff", edgecolor=BLUE, linewidth=1.1))
        ax.text(x + w / 2, y + h / 2, name, ha="center", va="center", fontsize=7, fontweight="bold", color=INK)
    _box(ax, 0.035, 0.29, 0.52, 0.09, "Shared blackboard + Orchestrator",
         "agents read earlier posts · each step commits · failures fall back",
         fill="#ffffff", edge=BLUE)
    _arrow(ax, (0.30, 0.70), (0.30, 0.66))

    # Layer 2: case pipeline
    ax.add_patch(FancyBboxPatch((0.60, 0.27), 0.38, 0.39, boxstyle="round,pad=0.006,rounding_size=0.015",
                                facecolor="#fef4ef", edgecolor=ORANGE, linewidth=1.2))
    ax.text(0.62, 0.648, "Layer 2 — Patient case pipeline", fontsize=8.5, fontweight="bold", color=INK, va="top")
    ax.text(0.62, 0.622, "per uploaded PDF · CrewAI LLM agents", fontsize=7, color=INK_2, va="top")
    crew = ["Extraction", "Prediction", "RAG", "Inventory", "Action", "Orchestrator"]
    for i, name in enumerate(crew):
        col, row = i % 3, i // 3
        _box(ax, 0.62 + col * 0.12, 0.50 - row * 0.095, 0.105, 0.075, name, fill="#ffffff", edge=ORANGE)
    ax.text(0.79, 0.33, "every agent's answer checked in code;\nfalls back to the verified value",
            ha="center", va="center", fontsize=7, color=INK_2)
    _arrow(ax, (0.79, 0.70), (0.79, 0.66))

    # Shared services
    services = [("XGBoost ×3 + SHAP", "PRBC / Platelets / FFP"), ("Prophet", "demand per component"),
                ("FAISS RAG", "WHO handbook, page cites"), ("Blood-bank domain", "units, ABO/RhD, FEFO"),
                ("LLM (Groq)", "language only")]
    for i, (name, body) in enumerate(services):
        _box(ax, 0.02 + i * 0.196, 0.13, 0.18, 0.10, name, body)
    _arrow(ax, (0.30, 0.27), (0.30, 0.23))
    _arrow(ax, (0.79, 0.27), (0.79, 0.23))

    _box(ax, 0.02, 0.01, 0.96, 0.08, "PostgreSQL  ·  Redis (cache)",
         "patients, predictions (+SHAP, citations), blood units, transfusions, donors, alerts, agent cycles, audit log")
    _arrow(ax, (0.5, 0.13), (0.5, 0.09))
    _save(fig, "fig_architecture")


def main():
    _style()
    fig_architecture()
    fig_simulation()
    fig_sensitivity()
    fig_prediction()
    fig_bandit()
    fig_forecast()


if __name__ == "__main__":
    main()
