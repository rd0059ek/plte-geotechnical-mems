
"""
compare_figures.py

Purpose
-------
Generate Figure 1 and Figure 2 for the manuscript:
(1) full-series comparison of Raw / Kalman / PLTE
(2) local zoomed comparison over the target interval

This script is written for repository release and reviewer inspection.
The comments therefore focus on reproducibility and manuscript intent,
rather than internal notebook-style trial-and-error notes.

How to use
----------
1. Prepare a CSV file containing a timestamp column and one acceleration column.
2. Edit the USER SETTINGS section below.
3. Run:
       python compare_figures.py

Outputs
-------
- Fig. 1.png
- Fig. 2.png
"""

from __future__ import annotations

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

# PLTE configuration
WINDOW_MINUTES = 1.0
N_CHANGEPOINTS = 25
CP_RANGE = 0.8
TAU = 0.05
N_JOBS = -1
MIN_POINTS_PER_WINDOW = 10

# Kalman configuration
KALMAN_QR_RATIO = 1e-4
KALMAN_INITIAL_P = 5.0

# Figure 1 settings: manuscript-wide comparison
FIG1_XLIM = (0, 75)
FIG1_YLIM = (-0.1, 1.0)
FIG1_X_SCALE = 10.0
FIG1_X_TICK = 15

# Figure 2 settings: local zoom
FIG2_XLIM = (0, 20)
FIG2_YLIM = (-0.5, 2.5)
FIG2_X_SCALE = 100.0
FIG2_X_TICK = 5
FIG2_Y_TICK = 0.5

# General display
FIG_WIDTH = 174 / 26
FIG_HEIGHT = 3
FONT_SIZE = 7


# =============================================================================
# DATA STRUCTURE
# =============================================================================
@dataclass
class PLTEConfig:
    window_minutes: float = WINDOW_MINUTES
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
    Read the input series and convert it to the standard two-column format:
    ds = timestamp, y = signal value.

    The original timestamps are preserved. No interpolation or resampling is
    applied here because the manuscript explicitly discusses direct processing
    on the original time axis.
    """
    raw = pd.read_csv(csv_path)
    raw[time_col] = pd.to_datetime(raw[time_col])
    raw = raw.sort_values(time_col).reset_index(drop=True)

    df = pd.DataFrame({
        "ds": raw[time_col],
        "y": pd.to_numeric(raw[accel_col], errors="coerce"),
    }).dropna().reset_index(drop=True)

    t0 = df["ds"].iloc[0]
    df["minutes"] = (df["ds"] - t0).dt.total_seconds() / 60.0
    return df


# =============================================================================
# KALMAN BASELINE
# =============================================================================
def kalman_1d(y: np.ndarray, qr_ratio: float = 1e-4, initial_p: float = 5.0) -> np.ndarray:
    """
    Simple 1D Kalman filter under a random-walk state model.

    This baseline is used as a representative recursive smoother. The goal is
    not to claim global superiority of one method, but to show how recursive
    smoothing and PLTE differ in trend-expression behavior.
    """
    n = len(y)
    x_est = np.zeros(n, dtype=float)
    x_est[0] = y[0]

    r = np.var(y[: min(100, n)]) if n > 1 else 1.0
    q = r * qr_ratio
    p = initial_p

    for i in range(1, n):
        p_pred = p + q
        k = p_pred / (p_pred + r)
        x_est[i] = x_est[i - 1] + k * (y[i] - x_est[i - 1])
        p = (1 - k) * p_pred

    return x_est


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

    The implementation keeps the original irregular timestamps inside each
    window. The fitted outputs from overlapping windows are averaged later.
    """
    window_end = window_start + pd.Timedelta(minutes=config.window_minutes)
    train = df.loc[(df["ds"] >= window_start) & (df["ds"] < window_end)]

    if len(train) < config.min_points_per_window:
        return None

    ds_vals = train["ds"].values
    y = train["y"].to_numpy(dtype=float)
    n = len(y)

    # Normalize time to [0, 1] inside the current window.
    t_sec = (ds_vals - ds_vals[0]).astype("timedelta64[s]").astype(float)
    t_scale = t_sec[-1] if t_sec[-1] > 0 else 1.0
    t = t_sec / t_scale

    # Normalize signal amplitude for numerical stability.
    y_mean = y.mean()
    y_range = y.max() - y.min()
    y_scale = y_range if y_range > 1e-12 else 1.0
    y_norm = (y - y_mean) / y_scale

    # Build changepoint design matrix.
    n_cp = min(config.n_changepoints, max(n // 4, 2))
    hist_end = max(int(np.ceil(config.cp_range * n)) - 1, 1)
    cp_idx = np.round(np.linspace(0, hist_end, n_cp)).astype(int)
    cp_idx = np.clip(cp_idx, 0, n - 1)

    s = t[cp_idx]
    a = (t[:, None] >= s[None, :]).astype(float)

    # Initial values from a simple least-squares line.
    k0, m0 = np.polyfit(t, y_norm, 1)
    sigma0 = max(np.std(y_norm), 1e-4)
    x0 = np.concatenate([[k0, m0, sigma0], np.zeros(n_cp)])

    def neg_log_posterior(params: np.ndarray) -> float:
        k_, m_, sigma_ = params[0], params[1], params[2]
        delta_ = params[3:]
        gamma_ = -s * delta_

        trend_ = (k_ + a @ delta_) * t + (m_ + a @ gamma_)
        resid_ = y_norm - trend_

        # Gaussian data term.
        nll = 0.5 * np.dot(resid_, resid_) / (sigma_ ** 2) + n * np.log(sigma_)

        # Sparse changepoint prior.
        reg_delta = np.sum(np.abs(delta_)) / config.tau

        # Weak regularization on slope/intercept and scale.
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
    Apply PLTE over sliding time windows and average overlapping fitted values.
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
        raise RuntimeError("No valid PLTE window was fitted. Please check the input series.")

    merged = pd.concat(valid_results, ignore_index=True)
    averaged = merged.groupby("ds", as_index=False)["yhat"].mean()
    averaged = averaged.rename(columns={"yhat": "yhat_plte"})
    return averaged


# =============================================================================
# PLOTTING
# =============================================================================
def prepare_plot_frame(df: pd.DataFrame, plte_avg: pd.DataFrame) -> pd.DataFrame:
    plot = df.merge(plte_avg, on="ds", how="left")
    plot["y_kalman"] = kalman_1d(
        plot["y"].to_numpy(),
        qr_ratio=KALMAN_QR_RATIO,
        initial_p=KALMAN_INITIAL_P,
    )
    return plot


def apply_plot_style() -> None:
    plt.rcParams.update({"font.size": FONT_SIZE, "font.family": "Times New Roman"})


def plot_figure_1(plot: pd.DataFrame) -> None:
    """
    Figure 1: full-series comparison.
    """
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    ax.plot(plot["minutes"], plot["y"] * FIG1_X_SCALE,
            color="gray", lw=0.3, alpha=0.4, label="Raw signal")
    ax.plot(plot["minutes"], plot["y_kalman"] * FIG1_X_SCALE,
            color="steelblue", lw=1.2, alpha=0.9, label=f"Kalman (Q/R={KALMAN_QR_RATIO:g})")
    ax.plot(plot["minutes"], plot["yhat_plte"] * FIG1_X_SCALE,
            color="black", lw=1.2, label=f"PLTE ({WINDOW_MINUTES:g} min)")

    ax.set_ylabel(r"Acceleration ($mm/s^2$)")
    ax.set_xlabel("Time (min)")
    ax.set_ylim(*FIG1_YLIM)
    ax.set_xlim(*FIG1_XLIM)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(FIG1_X_TICK))
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("Fig. 1.png", dpi=300)
    plt.close(fig)


def plot_figure_2(plot: pd.DataFrame) -> None:
    """
    Figure 2: local zoom for the interval discussed in the manuscript.
    """
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    ax.plot(plot["minutes"], plot["y"] * FIG2_X_SCALE,
            color="gray", lw=0.3, alpha=0.4, label="Raw signal")
    ax.plot(plot["minutes"], plot["y_kalman"] * FIG2_X_SCALE,
            color="steelblue", lw=1.2, alpha=0.9, label=f"Kalman (Q/R={KALMAN_QR_RATIO:g})")
    ax.plot(plot["minutes"], plot["yhat_plte"] * FIG2_X_SCALE,
            color="black", lw=1.2, label=f"PLTE ({WINDOW_MINUTES:g} min)")

    ax.set_ylabel(r"Acceleration ($10\ mm/s^2$)")
    ax.set_xlabel("Time (min)")
    ax.set_ylim(*FIG2_YLIM)
    ax.set_xlim(*FIG2_XLIM)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(FIG2_X_TICK))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(FIG2_Y_TICK))
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("Fig. 2.png", dpi=300)
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    apply_plot_style()

    config = PLTEConfig()
    df = load_series(CSV_PATH, TIME_COL, ACCEL_COL)
    plte_avg = run_plte(df, config)
    plot = prepare_plot_frame(df, plte_avg)

    plot_figure_1(plot)
    plot_figure_2(plot)
    print("Saved: Fig. 1.png, Fig. 2.png")


if __name__ == "__main__":
    main()
