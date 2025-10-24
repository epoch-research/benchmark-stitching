# run_eci_in_play_confidence_bars.py

# RUN DATA_LOADER.PY FIRST!
# it will produce scores_df_final.csv, which you need
# fit.py should be in the same directory

# Produces a plot with:
#  - model points (ECI vs time) filtered to >= 2023-01-01
#  - for each benchmark, a vertical bar at the average date of models whose ECI
#    is within the benchmark's 10–90% in-play range; the bar spans that ECI range.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from datetime import datetime, timedelta
from fit import fit_statistical_model

CSV_PATH = "outputs/scores_df_final.csv"
SVG_OUTPUT_PATH = "eci_in_play.svg"
ANNOTATION_MODELS = {
    "gpt-5-2025-08-07_medium",
    "claude-3-5-sonnet-20240620",
    "gpt-4-0314",
}

# here you can edit benchmarks, names are in scores_df_final.csv
BENCHMARKS_TO_SHOW = ("TriviaQA", "GPQA diamond", "FrontierMath-2025-02-28-Private")

# filtering out old models
DATE_MIN = pd.Timestamp("2023-01-01")

# here you can adjust in-play thresholds
P_LOW = 0.10
P_HIGH = 0.90

def _logit(p):
    p = np.clip(p, 1e-9, 1-1e-9)
    return np.log(p/(1-p))

def compute_eci_band(alpha_b, D_b, p_low, p_high):
    """Invert alpha(alpha(C-D)) = p  to C = D + (1/alpha) * logit(p). Returns (ECI_low, ECI_high)."""
    lo = D_b + (1.0/alpha_b) * _logit(p_low)
    hi = D_b + (1.0/alpha_b) * _logit(p_high)
    return (float(min(lo, hi)), float(max(lo, hi)))

def prepare_model_points(model_capabilities_df, raw_df, *, date_col="date"):
    """
    Ensure there's a single release date per model. If the fitted DF already has 'date'
    we keep it; else we derive from the raw CSV (first non-null date per model).
    """
    models = model_capabilities_df.copy()

    if date_col not in models.columns or models[date_col].isna().all():
        model_dates = (
            raw_df.dropna(subset=[date_col])
                 .sort_values(date_col)
                 .groupby("model", as_index=False)
                 .first()[["model", date_col]]
        )
        models = models.merge(model_dates, on="model", how="left")

    models["_dt"] = pd.to_datetime(models[date_col], errors="coerce")
    models = models.loc[models["_dt"] >= DATE_MIN].copy()
    return models

def average_date(xs_dt):
    """Return the average datetime of a pandas Series of datetimes."""
    if len(xs_dt) == 0:
        return None
    ns = xs_dt.view("int64")
    avg_ns = int(ns.mean())
    return pd.to_datetime(avg_ns)

def draw_confidence_bars(ax, bands_df, models_df, color_map, cap_days=10, lw=3.0):
    """
    For each benchmark:
      - find models with ECI in [eci_low, eci_high]
      - take average x-date of those models
      - draw a vertical line from eci_low to eci_high at that x, with short caps
    """
    cap_delta = pd.Timedelta(days=cap_days)

    proxies = []  # for legend
    labels = []

    for _, row in bands_df.iterrows():
        name = row["benchmark_name"]
        y0, y1 = float(row["eci_low"]), float(row["eci_high"])
        color = color_map[name]

        # models inside the band
        mask = (models_df["estimated_capability"] >= y0) & (models_df["estimated_capability"] <= y1)
        in_band = models_df.loc[mask & models_df["_dt"].notna(), ["_dt", "estimated_capability"]]

        x_mid = average_date(in_band["_dt"])
        if x_mid is None:
            proxy = mpatches.Patch(color=color, alpha=0.25)
            proxies.append(proxy)
            labels.append(f"{name} (no models in range)")
            continue

        # vertical line
        ax.vlines(x_mid, y0, y1, color=color, linewidth=lw, alpha=0.95, zorder=4)

        # caps at ends (short horizontal ticks)
        ax.hlines(y0, x_mid - cap_delta, x_mid + cap_delta, color=color, linewidth=lw, alpha=0.95, zorder=4)
        ax.hlines(y1, x_mid - cap_delta, x_mid + cap_delta, color=color, linewidth=lw, alpha=0.95, zorder=4)

        proxy = mpatches.Patch(color=color, alpha=0.25)
        proxies.append(proxy)
        labels.append(name)

    if proxies:
        ax.legend(handles=proxies, labels=labels, title="Benchmarks", framealpha=0.9, loc = "upper left")

def main():
    df = pd.read_csv(CSV_PATH)

    # fitting with benchmark anchoring - WinoGrande difficulty=0, slope=1
    _, model_caps_df, bench_params_df = fit_statistical_model(
        df,
        anchor_mode="benchmark",
        anchor_benchmark="Winogrande",
        anchor_difficulty=0.0,
        anchor_slope=1.0,
        regularization_strength=0.1
    )

    # compute in-play bands for selected benchmarks
    band_rows = []
    for _, r in bench_params_df.iterrows():
        name = r["benchmark_name"]
        if name not in BENCHMARKS_TO_SHOW:
            continue
        alpha_b = float(r["estimated_slope"])
        D_b     = float(r["estimated_difficulty"])
        eci_low, eci_high = compute_eci_band(alpha_b, D_b, P_LOW, P_HIGH)
        band_rows.append({
            "benchmark_name": name,
            "eci_low": eci_low,
            "eci_high": eci_high,
            "benchmark_release_date": r.get("benchmark_release_date", None)
        })
    bands_df = pd.DataFrame(band_rows)
    models = prepare_model_points(model_caps_df, df, date_col="date")
    fig, ax = plt.subplots(figsize=(11.5, 7))

    # Scatter the filtered models
    ax.scatter(models["_dt"], models["estimated_capability"], s=20, alpha=0.7, zorder=3)

    annotated = models.loc[
        models["model"].isin(ANNOTATION_MODELS) & models["_dt"].notna(),
        ["model", "_dt", "estimated_capability"]
    ]
    if not annotated.empty:
        # draw hollow markers for annotated points so labels are easier to read
        ax.scatter(
            annotated["_dt"],
            annotated["estimated_capability"],
            s=60,
            facecolors="none",
            edgecolors="black",
            linewidths=1.2,
            zorder=5,
        )
        for _, row in annotated.iterrows():
            ax.annotate(
                row["model"],
                xy=(row["_dt"], row["estimated_capability"]),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=8,
                fontweight="bold",
                zorder=6,
            )

    # nice date formatting
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

    # auto colors for any number of benchmarks, so you can type in many at a time
    cmap = plt.get_cmap("tab10")
    bench_names = bands_df["benchmark_name"].tolist()
    color_map = {name: cmap(i % cmap.N) for i, name in enumerate(bench_names)}

    # draw vertical “confidence bars” at avg x of models within each band
    draw_confidence_bars(ax, bands_df, models, color_map=color_map, cap_days=10, lw=3.0)

    ax.set_xlabel("Time (model release date)")
    ax.set_ylabel("ECI")
    ax.set_title("Benchmarks ‘in play’ (10–90%) as vertical bars at avg model date; models filtered to ≥ 2023-01-01")
    ax.grid(True, linestyle=":", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(SVG_OUTPUT_PATH, format="svg")
    plt.show()

if __name__ == "__main__":
    main()
