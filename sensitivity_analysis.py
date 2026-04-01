
"""
sensitivity_analysis.py

Purpose
-------
Generate Figure 3 and Figure 4 for the manuscript:
(1) multi-window sensitivity comparison
(2) local zoomed comparison of different PLTE window settings

This script is intended for repository release. The comments therefore explain
the analytical purpose of each block in clear English for reviewers and readers.

How to use
----------
1. Prepare a CSV file containing a timestamp column and one acceleration column.
2. Edit the USER SETTINGS section below.
3. Run:
       python sensitivity_analysis.py

Outputs
-------
- Fig. 3.png
- Fig. 4.png
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from joblib import Parallel, delayed
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import warnings




import warnings
from dataclasses import dataclass

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import minimize
from tqdm import tqdm

warnings.filterwarnings("ignore")


# =============================================================================
# USER SETTINGS
# =============================================================================
CSV_PATH = "1.2_A5.csv"
TIME_COL = "time"
ACCEL_COL = "x"

# PLTE sensitivity settings
WINDOW_MINUTES_LIST = [0.5, 1.0, 3.0, 5.0]
N_CHANGEPOINTS = 25
CP_RANGE = 0.8
TAU = 0.05
N_JOBS = -1
MIN_POINTS_PER_WINDOW = 10

# Figure 3 settings
FIG_WIDTH = 174 / 26 * 0.5
FIG3_PANEL_HEIGHT = 1.0
FIG3_XLIM = (0, 20)
FIG3_YLIM = (-0.5, 2.5)
FIG3_X_TICK = 5

# Figure 4 settings
FIG4_XLIM = (68, 73)
FIG4_YLIM = (7, 11)

# General display
FONT_SIZE = 7
PLOT_SCALE = 100.0
PALETTE = ["black", "steelblue", "seagreen", "darkorchid"]


# =============================================================================
# DATA STRUCTURE
# =============================================================================
@dataclass
class PLTEConfig:
    window_minutes: float
    n_changepoints: int = N_CHANGEPOINTS
    cp_range: float = CP_RANGE
    tau: float = TAU
    n_jobs: int = N_JOBS
    min_points_per_window: int = MIN_POINTS_PER_WINDOW


# =============================================================================
# DATA LOADING
# =============================================================================
def load_series(csv_path: str, time_col: str, accel_col: str) -> pd.DataFrame:
    """
    Read the input series while preserving the original timestamps.
    No interpolation or resampling is performed.
    """
    raw = pd.read_csv(csv_path)
    raw[time_col] = pd.to_datetime(raw[time_col])
    raw = raw.sort_values(time_col).reset_index(drop=True)

    df = pd.DataFrame({
        "ds": raw[time_col],
        "y": pd.to_numeric(raw[accel_col], errors="coerce"),
    }).dropna().reset_index(drop=True)

    t0 = df["ds"].iloc[0]
    df["relative_min"] = (df["ds"] - t0).dt.total_seconds() / 60.0
    return df


# =============================================================================
# PLTE CORE
# =============================================================================
def fit_single_window(
    df: pd.DataFrame,
    window_start: pd.Timestamp,
    config: PLTEConfig,
) -> pd.DataFrame | None:
    """
    Fit a Prophet-style piecewise-linear MAP trend within a single time window.
    """
    window_end = window_start + pd.Timedelta(minutes=config.window_minutes)
    train = df.loc[(df["ds"] >= window_start) & (df["ds"] < window_end)]

    if len(train) < config.min_points_per_window:
        return None

    ds_vals = train["ds"].values
    y = train["y"].to_numpy(dtype=float)
    n = len(y)

    t_sec = (ds_vals - ds_vals[0]).astype("timedelta64[s]").astype(float)
    t_scale = t_sec[-1] if t_sec[-1] > 0 else 1.0
    t = t_sec / t_scale

    y_mean = y.mean()
    y_range = y.max() - y.min()
    y_scale = y_range if y_range > 1e-12 else 1.0
    y_norm = (y - y_mean) / y_scale

    n_cp = min(config.n_changepoints, max(n // 4, 2))
    hist_end = max(int(np.ceil(config.cp_range * n)) - 1, 1)
    cp_idx = np.round(np.linspace(0, hist_end, n_cp)).astype(int)
    cp_idx = np.clip(cp_idx, 0, n - 1)

    s = t[cp_idx]
    a = (t[:, None] >= s[None, :]).astype(float)

    k0, m0 = np.polyfit(t, y_norm, 1)
    sigma0 = max(np.std(y_norm), 1e-4)
    x0 = np.concatenate([[k0, m0, sigma0], np.zeros(n_cp)])

    def neg_log_posterior(params: np.ndarray) -> float:
        k_, m_, sigma_ = params[0], params[1], params[2]
        delta_ = params[3:]
        gamma_ = -s * delta_

        trend_ = (k_ + a @ delta_) * t + (m_ + a @ gamma_)
        resid_ = y_norm - trend_

        nll = 0.5 * np.dot(resid_, resid_) / (sigma_ ** 2) + n * np.log(sigma_)
        reg_delta = np.sum(np.abs(delta_)) / config.tau
        reg_km = (k_ ** 2 + m_ ** 2) / 50.0
        reg_sigma = np.log(1.0 + (sigma_ / 5.0) ** 2)
        return float(nll + reg_delta + reg_km + reg_sigma)

    bounds = [(None, None), (None, None), (1e-8, None)] + [(None, None)] * n_cp
    result = minimize(
        neg_log_posterior,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
    )

    k_, m_ = result.x[0], result.x[1]
    delta_ = result.x[3:]
    gamma_ = -s * delta_
    yhat_norm = (k_ + a @ delta_) * t + (m_ + a @ gamma_)
    yhat = yhat_norm * y_scale + y_mean

    return pd.DataFrame({"ds": ds_vals, "yhat": yhat})


def run_plte(df: pd.DataFrame, config: PLTEConfig) -> pd.DataFrame:
    """
    Apply PLTE over sliding windows and average overlapping fitted values.
    """
    starts = pd.date_range(
        df["ds"].iloc[0],
        df["ds"].iloc[-1],
        freq=pd.Timedelta(minutes=config.window_minutes),
    )

    window_results = Parallel(n_jobs=config.n_jobs, backend="loky")(
        delayed(fit_single_window)(df, start, config)
        for start in tqdm(starts, desc=f"PLTE window={config.window_minutes} min")
    )

    valid_results = [res for res in window_results if res is not None]
    if not valid_results:
        raise RuntimeError(
            f"No valid PLTE window was fitted for {config.window_minutes} min."
        )

    merged = pd.concat(valid_results, ignore_index=True)
    averaged = merged.groupby("ds", as_index=False)["yhat"].mean()
    return averaged


# =============================================================================
# PLOTTING
# =============================================================================
def apply_plot_style() -> None:
    plt.rcParams.update({"font.size": FONT_SIZE, "font.family": "Times New Roman"})


def compute_sensitivity_results(df: pd.DataFrame) -> dict[float, pd.DataFrame]:
    """
    Run PLTE for multiple window sizes to evaluate trend-shape sensitivity.
    """
    outputs: dict[float, pd.DataFrame] = {}
    for window_minutes in WINDOW_MINUTES_LIST:
        config = PLTEConfig(window_minutes=window_minutes)
        avg = run_plte(df, config)
        avg["relative_min"] = (avg["ds"] - df["ds"].iloc[0]).dt.total_seconds() / 60.0
        outputs[window_minutes] = avg
    return outputs


def plot_figure_3(df: pd.DataFrame, sensitivity_results: dict[float, pd.DataFrame]) -> None:
    """
    Figure 3: top panel for direct comparison, lower panels for each window.
    """
    n_win = len(WINDOW_MINUTES_LIST)
    fig3, axes3 = plt.subplots(
        n_win + 1,
        1,
        figsize=(FIG_WIDTH, FIG3_PANEL_HEIGHT * (n_win + 1)),
        sharex=True,
    )

    if not hasattr(axes3, "__len__"):
        axes3 = [axes3]

    # Upper panel: all window settings overlaid.
    ax_all = axes3[0]
    ax_all.plot(
        df["relative_min"],
        df["y"] * PLOT_SCALE,
        color="black",
        alpha=0.2,
        lw=0.5,
        zorder=1,
        label="Raw signal",
    )

    for window_minutes, color in zip(WINDOW_MINUTES_LIST, PALETTE):
        result = sensitivity_results[window_minutes]
        ax_all.plot(
            result["relative_min"],
            result["yhat"] * PLOT_SCALE,
            color=color,
            alpha=0.8,
            lw=1.0,
            label=f"PLTE ({window_minutes:g} min)",
        )

    ax_all.set_ylabel(r"Acc. ($10mm/s^2$)")
    ax_all.set_ylim(*FIG3_YLIM)
    ax_all.grid(alpha=0.3, linestyle="-")
    ax_all.legend(loc="upper left", ncol=2, frameon=True, framealpha=0.6)
    ax_all.text(
        0.98, 0.95, "Comparison",
        transform=ax_all.transAxes, ha="right", va="top", fontweight="bold"
    )

    # Lower panels: each window shown separately.
    for i, (window_minutes, color) in enumerate(zip(WINDOW_MINUTES_LIST, PALETTE), start=1):
        ax = axes3[i]
        result = sensitivity_results[window_minutes]

        ax.plot(
            df["relative_min"],
            df["y"] * PLOT_SCALE,
            color="black",
            alpha=0.2,
            lw=0.5,
            label="Raw signal",
            zorder=1,
        )
        ax.plot(
            result["relative_min"],
            result["yhat"] * PLOT_SCALE,
            color=color,
            alpha=0.9,
            lw=1.2,
            label=f"PLTE ({window_minutes:g} min)",
            zorder=2,
        )

        ax.set_ylabel(r"Acc. ($10mm/s^2$)")
        ax.set_ylim(*FIG3_YLIM)
        ax.grid(alpha=0.2, linestyle="--")
        ax.legend(loc="upper left", frameon=False)
        ax.text(
            0.98, 0.95, f"Window: {window_minutes:g} min",
            transform=ax.transAxes, ha="right", va="top", fontweight="bold"
        )

    axes3[-1].set_xlabel("Time (min)")
    axes3[-1].set_xlim(*FIG3_XLIM)
    axes3[-1].xaxis.set_major_locator(ticker.MultipleLocator(FIG3_X_TICK))

    plt.tight_layout()
    plt.savefig("Fig. 3.png", dpi=300)
    plt.close(fig3)


def plot_figure_4(df: pd.DataFrame, sensitivity_results: dict[float, pd.DataFrame]) -> None:
    """
    Figure 4: local zoomed overlay used to show whether the trend expression is
    substantially altered under different window settings.
    """
    fig4, ax4 = plt.subplots(figsize=(FIG_WIDTH, 1.5))

    ax4.plot(
        df["relative_min"],
        df["y"] * PLOT_SCALE,
        color="black",
        alpha=0.15,
        lw=0.5,
        label="Raw signal",
        zorder=1,
    )

    for window_minutes, color in zip(WINDOW_MINUTES_LIST, PALETTE):
        result = sensitivity_results[window_minutes]
        ax4.plot(
            result["relative_min"],
            result["yhat"] * PLOT_SCALE,
            color=color,
            alpha=0.8,
            lw=1.0,
            label=f"PLTE ({window_minutes:g} min)",
        )

    ax4.set_ylabel(r"Acceleration ($10mm/s^2$)")
    ax4.set_xlabel("Time (min)")
    ax4.set_ylim(*FIG4_YLIM)
    ax4.set_xlim(*FIG4_XLIM)
    ax4.grid(alpha=0.3, linestyle="-")
    ax4.legend(loc="upper left", fontsize="small", frameon=True)

    plt.tight_layout()
    plt.savefig("Fig. 4.png", dpi=300)
    plt.close(fig4)


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    apply_plot_style()
    df = load_series(CSV_PATH, TIME_COL, ACCEL_COL)
    sensitivity_results = compute_sensitivity_results(df)

    plot_figure_3(df, sensitivity_results)
    plot_figure_4(df, sensitivity_results)
    print("Saved: Fig. 3.png, Fig. 4.png")


if __name__ == "__main__":
    main()
