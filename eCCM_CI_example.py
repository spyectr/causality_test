"""
eCCM_CI_example.py — Minimal pedagogical test of eCCM lag convention.

Two mode knobs are available:
  RNN_MODE = "ricker":   causal-chain U/X/Y and CI X/Y follow log-Ricker dynamics
  RNN_MODE = "logistic": causal-chain U/X/Y and CI X/Y follow logistic dynamics

  CI_MODE = "ricker":    CI source U is an autonomous Ricker process
  CI_MODE = "logistic":  CI source U is an autonomous logistic process
  CI_MODE = "ou":        CI source U is an OU process

Two scenarios:
  A. Causal chain:   U -> X (lag 0) -> Y (lag L)
  B. Common input:   U -> X (lag 0),  U -> Y (lag L),  no X->Y

For each scenario, runs eCCM(U|X), eCCM(U|Y), eCCM(X|U), eCCM(Y|U),
eCCM(X|Y), and eCCM(Y|X)
using the exact same pipeline functions as Paper_sim_eCCM.py, and
produces diagnostic plots.

Outputs saved to: example/eccm_<rnn_mode>/
"""
from __future__ import annotations

import sys
import time
import shutil
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class _TeeStream:
    """Write to both a file and the original stream."""
    def __init__(self, stream, log_file):
        self._stream = stream
        self._log = log_file
    def write(self, msg):
        self._stream.write(msg)
        self._log.write(msg)
    def flush(self):
        self._stream.flush()
        self._log.flush()

# --- Make RNNCausality importable (same as Paper_sim_eCCM.py) ---------------
PROJ = Path(__file__).resolve().parent
RNNC_DIR = PROJ / "RNNCausality"
if str(RNNC_DIR) not in sys.path:
    sys.path.insert(0, str(RNNC_DIR))
import DelayEmbedding.EfficientCCM as CCM  # noqa: E402

from Paper_sim_common import (
    _transpose_pair_axes,
    ECCM_COLORS, LW_MAIN, LW_SEC, LW_REF, MS_MAIN, REF_STYLE, LEGEND_KW,
)
from Paper_sim_eCCM_aux import (
    _pair_cross_correlogram,
    flatten_surrogate_peak_samples,
)
from sim_IC_aux import set_plot_style

# ============================================================================
# 1. Simulation
# ============================================================================
def simulate_ricker(
    R: np.ndarray,        # (N, N) coupling matrix
    L: np.ndarray,        # (N, N) integer lag matrix
    T: int,               # number of time steps
    n_trials: int = 1,
    seed: int = 42,
) -> np.ndarray:
    """Simulate log-Ricker map, shape (n_trials, T, N)."""
    N = R.shape[0]
    max_lag = int(L.max())
    rng = np.random.default_rng(seed)
    x = np.zeros((n_trials, T, N), dtype=np.float64)
    x[:, :max_lag + 1, :] = rng.uniform(0.5, 2.0, (n_trials, max_lag + 1, N))
    for t in range(max_lag, T - 1):
        coupling = np.zeros((n_trials, N), dtype=np.float64)
        for j in range(N):
            for i in range(N):
                lag = int(L[i, j])
                coupling[:, i] += R[i, j] * np.exp(x[:, t - lag, j])
        x[:, t + 1, :] = x[:, t, :] + np.diag(R)[np.newaxis, :] - coupling
    return x


def simulate_ricker_driven(
    R: np.ndarray,        # (N, N) coupling matrix
    L: np.ndarray,        # (N, N) integer lag matrix
    u: np.ndarray,        # (n_trials, T, N) external drive
    seed: int = 42,
) -> np.ndarray:
    """Simulate log-Ricker map with additive external drive u[t]."""
    n_trials, T, N = u.shape
    max_lag = int(L.max())
    rng = np.random.default_rng(seed)
    x = np.zeros((n_trials, T, N), dtype=np.float64)
    x[:, :max_lag + 1, :] = rng.uniform(0.5, 2.0, (n_trials, max_lag + 1, N))
    for t in range(max_lag, T - 1):
        coupling = np.zeros((n_trials, N), dtype=np.float64)
        for j in range(N):
            for i in range(N):
                lag = int(L[i, j])
                coupling[:, i] += R[i, j] * np.exp(x[:, t - lag, j])
        x[:, t + 1, :] = x[:, t, :] + np.diag(R)[np.newaxis, :] - coupling + u[:, t, :]
    return x


def simulate_logistic(
    W: np.ndarray,        # (N, N) cross-coupling weights
    L: np.ndarray,        # (N, N) integer lag matrix
    T: int,               # number of time steps
    n_trials: int = 1,
    seed: int = 42,
    r=3.9,                # scalar or (N,) per-unit logistic parameter
) -> np.ndarray:
    """Simulate bounded lag-coupled logistic map, shape (n_trials, T, N)."""
    W = np.asarray(W, dtype=np.float64)
    N = W.shape[0]
    r = np.broadcast_to(np.asarray(r, dtype=np.float64), (N,)).copy()
    if np.any(r <= 0.0) or np.any(r > 4.0):
        raise ValueError("Logistic parameter r must satisfy 0 < r <= 4.")
    if np.any(W < 0.0):
        raise ValueError("Logistic couplings must be nonnegative.")

    row_sums = W.sum(axis=1)
    if np.any(row_sums > 1.0):
        raise ValueError("Each logistic coupling row must sum to <= 1.")

    max_lag = int(L.max())
    rng = np.random.default_rng(seed)
    x = np.zeros((n_trials, T, N), dtype=np.float64)
    x[:, :max_lag + 1, :] = rng.uniform(0.1, 0.9, (n_trials, max_lag + 1, N))
    self_w = 1.0 - row_sums

    for t in range(max_lag, T - 1):
        fx_self = r[np.newaxis, :] * x[:, t, :] * (1.0 - x[:, t, :])
        x_next = self_w[np.newaxis, :] * fx_self
        for j in range(N):
            for i in range(N):
                wij = W[i, j]
                if i == j or wij == 0.0:
                    continue
                lag = int(L[i, j])
                x_del = x[:, t - lag, j]
                x_next[:, i] += wij * x_del
        x[:, t + 1, :] = x_next
    return x


def generate_ou(n_trials, T, rho=0.95, sigma_eps=0.1, seed=99):
    """Generate OU process c[t], shape (n_trials, T)."""
    rng = np.random.default_rng(seed)
    c = np.zeros((n_trials, T), dtype=np.float64)
    c[:, 0] = rng.normal(0.0, sigma_eps / np.sqrt(1.0 - rho**2), n_trials)
    eps = rng.normal(0.0, sigma_eps, (n_trials, T))
    for t in range(1, T):
        c[:, t] = rho * c[:, t - 1] + eps[:, t]
    return c


def generate_ci_source(
    ci_mode: str,
    T: int,
    n_trials: int,
    seed: int,
    r_self: float,
    logistic_r: float,
    ou_rho: float,
    ou_sigma: float,
) -> np.ndarray:
    """Generate the CI source U as a single autonomous process."""
    if ci_mode == "ricker":
        R_u = np.array([[r_self]], dtype=np.float64)
        L_u = np.array([[0]], dtype=np.int32)
        return simulate_ricker(R_u, L_u, T, n_trials, seed)[:, :, 0]
    if ci_mode == "logistic":
        W_u = np.zeros((1, 1), dtype=np.float64)
        L_u = np.zeros((1, 1), dtype=np.int32)
        return simulate_logistic(W_u, L_u, T, n_trials, seed, r=logistic_r)[:, :, 0]
    if ci_mode == "ou":
        return generate_ou(n_trials, T, rho=ou_rho, sigma_eps=ou_sigma, seed=seed)
    raise ValueError(f"Unknown CI_MODE: {ci_mode!r}. Use 'ricker', 'logistic', or 'ou'.")


def _ci_source_to_logistic_drive(
    u: np.ndarray,
    ci_mode: str,
    logistic_r: float,
    ou_alpha: float,
) -> np.ndarray:
    """Map the CI source into [0, 1] before driving logistic X/Y nodes."""
    if ci_mode == "logistic":
        return np.clip(u, 0.0, 1.0)
    if ci_mode == "ou":
        u = ou_alpha * u
    u = np.clip(u, -10.0, 10.0)
    return 1.0 / (1.0 + np.exp(-u))


def simulate_logistic_common_input(
    u: np.ndarray,
    ci_mode: str,
    T: int,
    n_trials: int,
    seed: int,
    delay: int,
    logistic_r,           # scalar or (2,) per-unit logistic parameter
    couple: float,
    ou_alpha: float,
) -> np.ndarray:
    """Simulate disconnected logistic X/Y nodes with a shared source U."""
    r = np.broadcast_to(np.asarray(logistic_r, dtype=np.float64), (2,)).copy()
    src = _ci_source_to_logistic_drive(u, ci_mode, float(r.mean()), ou_alpha)
    max_lag = max(0, int(delay))
    rng = np.random.default_rng(seed)
    xy = np.zeros((n_trials, T, 2), dtype=np.float64)
    xy[:, :max_lag + 1, :] = rng.uniform(0.1, 0.9, (n_trials, max_lag + 1, 2))
    self_w = 1.0 - couple

    for t in range(max_lag, T - 1):
        fx = r[np.newaxis, :] * xy[:, t, :] * (1.0 - xy[:, t, :])
        xy[:, t + 1, 0] = self_w * fx[:, 0] + couple * src[:, t]
        xy[:, t + 1, 1] = self_w * fx[:, 1] + couple * src[:, t - int(delay)]
    return xy


def simulate_ricker_common_input(
    u: np.ndarray,
    ci_mode: str,
    T: int,
    n_trials: int,
    seed: int,
    delay: int,
    r_self,               # scalar or (2,) per-unit self-interaction
    couple: float,
    ou_alpha: float,
) -> np.ndarray:
    """Simulate disconnected Ricker X/Y nodes with a shared source U."""
    rs = np.broadcast_to(np.asarray(r_self, dtype=np.float64), (2,)).copy()
    if ci_mode == "ou":
        u_drive = np.zeros((n_trials, T, 2), dtype=np.float64)
        u_drive[:, :, 0] = ou_alpha * u
        u_drive[:, int(delay):, 1] = ou_alpha * u[:, :T - int(delay)]
        R_xy = np.zeros((2, 2), dtype=np.float64)
        L_xy = np.zeros((2, 2), dtype=np.int32)
        for i in range(2):
            R_xy[i, i] = rs[i]
            L_xy[i, i] = 0
        return simulate_ricker_driven(R_xy, L_xy, u_drive, seed=seed)

    src = np.exp(np.clip(u, -10.0, 10.0))
    max_lag = max(0, int(delay))
    rng = np.random.default_rng(seed)
    xy = np.zeros((n_trials, T, 2), dtype=np.float64)
    xy[:, :max_lag + 1, :] = rng.uniform(0.5, 2.0, (n_trials, max_lag + 1, 2))

    for t in range(max_lag, T - 1):
        coupling_x = rs[0] * np.exp(np.clip(xy[:, t, 0], -10.0, 10.0))
        coupling_y = rs[1] * np.exp(np.clip(xy[:, t, 1], -10.0, 10.0))
        coupling_x += couple * src[:, t]
        coupling_y += couple * src[:, t - int(delay)]
        xy[:, t + 1, 0] = xy[:, t, 0] + rs[0] - coupling_x
        xy[:, t + 1, 1] = xy[:, t, 1] + rs[1] - coupling_y
    return xy


# ============================================================================
# 2. Pipeline helpers (exact copies from Paper_sim_eCCM.py)
# ============================================================================
def _backend_selected_pairs(selected_pairs_local: np.ndarray) -> np.ndarray:
    local = np.asarray(selected_pairs_local, dtype=np.int32).reshape(-1, 2)
    return np.ascontiguousarray(local[:, ::-1], dtype=np.int32)


def _normalize_backend_ccm_outputs(ccmout: dict) -> dict:
    out = dict(ccmout)
    for key in ("maxdim_fcf", "pvals", "cf_signs", "surrogate_fcf",
                "surr_fcf", "surr_xcorr", "all_fcf", "all_corrs",
                "all_cf_signs",
                "reverse_surrogate_fcf", "reverse_surrogate_corr"):
        if key in out:
            out[key] = _transpose_pair_axes(out[key])
    return out


# ============================================================================
# 3. Run eCCM
# ============================================================================
def run_eccm(series, lags, selected_pairs, save_path,
             dim=5, kfolds=3, n_surrogates=20, delay=1):
    """Run eCCM using the exact same CCM.CCM call as Paper_sim_eCCM.py."""
    ccmout = CCM.CCM(
        np.ascontiguousarray(series, dtype=np.float32),
        dim_min=1, dim_max=dim, kfolds=kfolds, delay=delay,
        lags=lags, random_projection=False,
        compute_pvalue=True, n_surrogates=n_surrogates,
        regular_pvalue=True,
        surrogate_test_lags=np.arange(1, dtype=np.int32),
        save=True, save_path=save_path,
        only_hubs=False, find_optimum_dims=True,
        max_processes=1,
        selected_pairs=_backend_selected_pairs(selected_pairs),
    )
    ccmout = _normalize_backend_ccm_outputs(ccmout)
    # RNNCausality scales lag indices by the embedding delay when lagstep=0.
    effective_delay = 1 if int(delay) == 0 else int(delay)
    ccmout["effective_lags"] = effective_delay * np.asarray(lags, dtype=np.int32)
    return ccmout


# ============================================================================
# 4. Plotting
# ============================================================================
def _diagnostic_stats(ccmout, lags, x_use, src_idx, tgt_idx):
    """Compute diagnostics for one directed pair."""
    efcf = np.asarray(ccmout["maxdim_fcf"], dtype=np.float64)
    plot_lags = np.asarray(ccmout.get("effective_lags", lags), dtype=np.int32).reshape(-1)
    mean_fcf = np.nanmean(efcf, axis=0)
    kfolds = efcf.shape[0]

    profile = mean_fcf[src_idx, tgt_idx, :]
    fold_profiles = efcf[:, src_idx, tgt_idx, :]
    peak_idx = int(np.nanargmax(profile))
    peak_lag = int(plot_lags[peak_idx])
    peak_val = float(profile[peak_idx])

    surr_fcf = ccmout.get("surrogate_fcf")
    has_surr = surr_fcf is not None
    if has_surr:
        surr_fcf = np.asarray(surr_fcf, dtype=np.float64)
        surr_peak = flatten_surrogate_peak_samples(surr_fcf)
        null_vals = surr_peak[:, src_idx, tgt_idx]
        null_vals = null_vals[np.isfinite(null_vals)]
        surr_flat = surr_fcf.reshape(-1, *surr_fcf.shape[-3:])
        ci_up = float(np.nanpercentile(surr_flat[:, src_idx, tgt_idx, 0], 99.5))
        ci_lo = float(np.nanpercentile(surr_flat[:, src_idx, tgt_idx, 0], 0.5))
        ci_med = float(np.nanmedian(surr_flat[:, src_idx, tgt_idx, 0]))
    else:
        null_vals = np.array([])
        ci_up = float("nan")
        ci_lo = float("nan")
        ci_med = float("nan")

    xcorr = _pair_cross_correlogram(
        x_use, source_idx=src_idx, target_idx=tgt_idx, lags=plot_lags
    )

    if null_vals.size > 0:
        from scipy import stats as _sp_stats
        mu, sigma = float(np.mean(null_vals)), float(np.std(null_vals))
        if sigma > 0:
            pval = float(_sp_stats.norm.sf(peak_val, mu, sigma))
        else:
            pval = float("nan")
    else:
        pval = float("nan")

    return dict(
        plot_lags=plot_lags,
        fold_profiles=fold_profiles,
        profile=profile,
        peak_lag=peak_lag,
        peak_val=peak_val,
        null_vals=null_vals,
        ci_up=ci_up,
        ci_lo=ci_lo,
        ci_med=ci_med,
        xcorr=xcorr,
        pval=pval,
        has_surr=has_surr,
        kfolds=kfolds,
    )


def _plot_diagnostic_row(axes, stats, row_label, show_col_titles=False):
    """Render one 3-panel diagnostic row into preallocated axes."""
    plot_lags = stats["plot_lags"]
    fold_profiles = stats["fold_profiles"]
    profile = stats["profile"]
    peak_lag = stats["peak_lag"]
    peak_val = stats["peak_val"]
    null_vals = stats["null_vals"]
    ci_up = stats["ci_up"]
    ci_lo = stats["ci_lo"]
    ci_med = stats["ci_med"]
    xcorr = stats["xcorr"]
    pval = stats["pval"]
    has_surr = stats["has_surr"]
    kfolds = stats["kfolds"]

    ax = axes[0]
    for k in range(kfolds):
        ax.plot(plot_lags, fold_profiles[k], color=ECCM_COLORS["a2b"], alpha=0.2, lw=LW_REF)
    ax.plot(plot_lags, profile, color=ECCM_COLORS["a2b"], lw=LW_MAIN, label="mean")
    if has_surr:
        ax.axhspan(ci_lo, ci_up, color="gray", alpha=0.25, label="Surr 99% CI")
        ax.axhline(ci_med, color="gray", ls="--", lw=LW_REF, alpha=0.5)
    ax.plot(peak_lag, peak_val, "o", color="red", ms=MS_MAIN, zorder=5,
            label=f"peak lag={peak_lag}")
    ax.set_xlabel("Lag"); ax.set_ylabel("eCCM (FCF)")
    if show_col_titles:
        ax.set_title("eCCM vs lag", fontsize=8)
    ax.legend(**LEGEND_KW, loc="best")
    ax.text(-0.42, 0.5, row_label, transform=ax.transAxes, rotation=90,
            va="center", ha="center", fontweight="bold", fontsize=8)

    ax = axes[1]
    if null_vals.size > 0:
        ax.hist(null_vals, bins=25, color="#666666", alpha=0.35, edgecolor="#666666",
                lw=LW_REF, label=f"surrogates (n={null_vals.size})")
        from scipy import stats as _sp_stats
        mu, sigma = float(np.mean(null_vals)), float(np.std(null_vals))
        if sigma > 0:
            x_fit = np.linspace(null_vals.min(), max(null_vals.max(), peak_val), 200)
            bin_w = (null_vals.max() - null_vals.min()) / 25
            ax.plot(x_fit, _sp_stats.norm.pdf(x_fit, mu, sigma) * null_vals.size * bin_w,
                    color="#666666", lw=LW_SEC, label=f"Gauss ($\\mu$={mu:.3f})")
            pval = float(_sp_stats.norm.sf(peak_val, mu, sigma))
        else:
            pval = float("nan")
        ax.axvline(peak_val, color="red", lw=LW_SEC, ls="--",
                   label=f"peak={peak_val:.3f}, p={pval:.2g}")
    else:
        ax.text(0.5, 0.5, "no surrogates", ha="center", va="center",
                transform=ax.transAxes, fontsize=8)
    ax.set_xlabel("Peak eCCM"); ax.set_ylabel("count")
    if show_col_titles:
        ax.set_title("Surrogate null", fontsize=8)
    ax.legend(**LEGEND_KW, loc="best")

    ax = axes[2]
    ax.plot(plot_lags, xcorr, color=ECCM_COLORS["a2b"], lw=LW_MAIN, label="trial avg")
    ax.axhline(0.0, **REF_STYLE)
    ax.axvline(peak_lag, color="red", lw=LW_SEC, ls="--", alpha=0.7)
    ax.set_xlabel("Lag"); ax.set_ylabel("Pearson r")
    if show_col_titles:
        ax.set_title("Cross-correlogram", fontsize=8)
    ax.legend(**LEGEND_KW, loc="best")


def plot_diagnostics_grid(ccmout, lags, x_use, diagnostics, title, out_dir, fname):
    """Stack all requested 3-panel diagnostics into an n-row x 3 grid."""
    set_plot_style(fontsize=8.0)
    subplot_w = 8.5 / 5.0
    subplot_h = 0.8 * 8.5 / 5.0
    n_rows = len(diagnostics)
    fig, axes = plt.subplots(
        n_rows, 3,
        figsize=(3 * subplot_w, n_rows * subplot_h),
        constrained_layout=True,
    )
    if n_rows == 1:
        axes = np.asarray([axes])
    results = {}

    for row, spec in enumerate(diagnostics):
        row_axes = axes[row]
        stats = _diagnostic_stats(
            ccmout, lags, x_use, src_idx=spec["src_idx"], tgt_idx=spec["tgt_idx"]
        )
        _plot_diagnostic_row(
            row_axes, stats, row_label=spec["row_label"], show_col_titles=(row == 0)
        )
        results[spec["key"]] = dict(
            peak_lag=stats["peak_lag"], peak_val=stats["peak_val"], pval=stats["pval"]
        )

    fig.suptitle(title, fontsize=8, fontweight="bold")
    fpath = out_dir / f"{fname}.pdf"
    fig.savefig(fpath, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {fpath}")
    return results


def plot_traces(x, names, title, out_dir, fname):
    """Plot example traces for one trial, shape (T, N)."""
    set_plot_style(fontsize=8.0)
    T, N = x.shape
    xmax = min(500, T - 1) if "ricker" in title else min(50, T - 1)
    fig, axes = plt.subplots(N, 1, figsize=(8, 1.5 * N), sharex=True,
                             constrained_layout=True)
    if N == 1: axes = [axes]
    colors = ["#0072B2", "#009E73", "#D55E00", "#CC79A7"]
    for i in range(N):
        axes[i].plot(np.arange(T), x[:, i], color=colors[i % len(colors)], lw=0.8)
        axes[i].set_ylabel(names[i])
    axes[-1].set_xlabel("Time step")
    axes[-1].set_xlim(0, xmax)
    fig.suptitle(title, fontsize=10, fontweight="bold")
    fpath = out_dir / f"{fname}.pdf"
    fig.savefig(fpath, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {fpath}")


def log_array_stats(name, x, axis_names=None):
    """Print per-variable summary statistics."""
    flat = x.reshape(-1, x.shape[-1]) if x.ndim >= 2 else x.reshape(-1, 1)
    N = flat.shape[1]
    names = axis_names or [str(i) for i in range(N)]
    print(f"  [{name}] shape={x.shape}")
    for i in range(N):
        v = flat[:, i]; fin = v[np.isfinite(v)]
        print(f"    {names[i]:6s}: min={fin.min():10.3f}  max={fin.max():10.3f}  "
              f"mean={fin.mean():10.3f}  std={fin.std():10.3f}")


def canonical_mode(mode: str, *, allow_ou: bool = False) -> str:
    """Accept the requested shorthand but keep internal mode handling simple."""
    lut = {
        "ricker": "ricker",
        "logistic": "logistic",
    }
    if allow_ou:
        lut["ou"] = "ou"
    key = str(mode).strip().lower()
    if key not in lut:
        opts = ["ricker", "logistic"] + (["ou"] if allow_ou else [])
        raise ValueError(f"Unknown mode {mode!r}. Use one of {opts}.")
    return lut[key]


# ============================================================================
# 5. Main
# ============================================================================
if __name__ == "__main__":
    OUT_BASE = PROJ / "example"

    # ------------------------------------------------------------------
    # Parameters (easy to customize)
    # ------------------------------------------------------------------
    # Autonomous node dynamics
    RNN_MODE  = "ricker"  # "ricker" (current) or "logistic"

    # Common input parameters
    CI_MODE   = "logistic"  # "ricker", "logistic", or "ou"
    CI_DELAY  = 4         # lag from U to Y


    T          = 2000      # time steps per trial
    N_TRIALS   = 20        # number of trials (concatenated for CCM)
    BURNIN     = 100      # discard first steps
    SEED       = 2025
    MAX_LAG    = 10       # base CCM lag range [-MAX_LAG, +MAX_LAG] before delay scaling
    # CCM_DIM    = 3        # max embedding dimension for ricker
    if RNN_MODE == 'ricker':CCM_DIM    = 3        # max embedding dimension
    elif RNN_MODE == 'logistic': CCM_DIM = 3        # logistic is simpler, so smaller max dim suffices
    CCM_KFOLDS = 4
    CCM_NSURR  = 20       # number of surrogates
    CCM_DELAY  = 1        # embedding delay

    # Ricker parameters (from Conner's Rmats: R_self ~ 3.66-3.80 per node)
    R_SELF    = 3.7       # self-interaction (chaotic regime, matches manuscript)
    R_COUPLE  = 0.4       # cross-coupling strength (Conner sweeps 0 to 0.6)

    # Logistic parameters: r near 4 stays chaotic; row-sum coupling <= 1 keeps x in [0, 1]
    LOGISTIC_R       = 3.9
    LOGISTIC_COUPLE  = 0.3

    # Per-unit parameter jitter: small Gaussian perturbation so X and Y have
    # slightly different intrinsic dynamics while remaining in the chaotic regime.
    PARAM_JITTER_STD = 0.05


    # OU parameters (only used when CI_MODE = "ou")
    OU_ALPHA  = 0.01       # drive amplitude
    OU_RHO    = 0.2      # OU autocorrelation
    OU_SIGMA  = 0.2       # OU noise std

    RNN_MODE = canonical_mode(RNN_MODE)
    CI_MODE = canonical_mode(CI_MODE, allow_ou=True)

    OUT = OUT_BASE / f"eccm_{RNN_MODE}"
    OUT.mkdir(parents=True, exist_ok=True)

    _log_fh = open(OUT / "run.log", "w")
    sys.stdout = _TeeStream(sys.__stdout__, _log_fh)
    sys.stderr = _TeeStream(sys.__stderr__, _log_fh)
    _t0_global = time.time()

    # Log all parameters
    print("Parameters:")
    print(f"  T={T}, N_TRIALS={N_TRIALS}, BURNIN={BURNIN}, SEED={SEED}")
    print(f"  MAX_LAG={MAX_LAG}, CCM_DIM={CCM_DIM}, CCM_KFOLDS={CCM_KFOLDS}")
    print(f"  CCM_NSURR={CCM_NSURR}, CCM_DELAY={CCM_DELAY}")
    print(f"  RNN_MODE={RNN_MODE}, CI_MODE={CI_MODE}, CI_DELAY={CI_DELAY}")
    print(f"  R_SELF={R_SELF}, R_COUPLE={R_COUPLE}")
    print(f"  LOGISTIC_R={LOGISTIC_R}, LOGISTIC_COUPLE={LOGISTIC_COUPLE}")
    print(f"  PARAM_JITTER_STD={PARAM_JITTER_STD}")
    if CI_MODE == "ou":
        print(f"  OU_ALPHA={OU_ALPHA}, OU_RHO={OU_RHO}, OU_SIGMA={OU_SIGMA}")
    print(f"  Total CCM series length = {N_TRIALS * (T - BURNIN)}")
    print()
    lags = np.arange(-MAX_LAG, MAX_LAG + 1, dtype=np.int32)
    N_nodes = 3  # U=0, X=1, Y=2

    # Generate per-unit jittered parameters (deterministic from SEED)
    _jitter_rng = np.random.default_rng(SEED + 7777)
    # 3-node versions for causal chain (U=0, X=1, Y=2)
    r_self_3 = R_SELF + PARAM_JITTER_STD * _jitter_rng.standard_normal(N_nodes)
    logistic_r_3 = np.clip(
        LOGISTIC_R + PARAM_JITTER_STD * _jitter_rng.standard_normal(N_nodes),
        3.57, 4.0,  # keep in chaotic regime for logistic map
    )
    # 2-node versions for CI scenario (X=0, Y=1)
    r_self_2 = r_self_3[1:]      # reuse X, Y jitter
    logistic_r_2 = logistic_r_3[1:]
    print(f"  Per-unit R_SELF (U,X,Y):    {r_self_3}")
    print(f"  Per-unit LOGISTIC_R (U,X,Y): {logistic_r_3}")
    print()

    # ------------------------------------------------------------------
    # Scenario A: Causal chain  U -> X (lag 0) -> Y (lag L)
    # ------------------------------------------------------------------
    print("=" * 60)
    print(f"SCENARIO A: Causal chain ({RNN_MODE})  U -> X (lag 0) -> Y (lag {CI_DELAY})")
    print("=" * 60)

    C_causal = np.zeros((N_nodes, N_nodes), dtype=np.float64)
    L_causal = np.zeros((N_nodes, N_nodes), dtype=np.int32)
    C_causal[1, 0] = R_COUPLE if RNN_MODE == "ricker" else LOGISTIC_COUPLE
    C_causal[2, 1] = R_COUPLE if RNN_MODE == "ricker" else LOGISTIC_COUPLE
    L_causal[1, 0] = 0
    L_causal[2, 1] = CI_DELAY

    if RNN_MODE == "ricker":
        R_causal = C_causal.copy()
        for i in range(N_nodes):
            R_causal[i, i] = r_self_3[i]
            L_causal[i, i] = 0
        print(f"\n  R =\n{R_causal}")
        print(f"  L =\n{L_causal}")
        x_causal = simulate_ricker(R_causal, L_causal, T, N_TRIALS, SEED)
    else:
        print(f"\n  W =\n{C_causal}")
        print(f"  L =\n{L_causal}")
        print(f"  per-unit r = {logistic_r_3}")
        print(f"  self weights = {1.0 - C_causal.sum(axis=1)}")
        x_causal = simulate_logistic(C_causal, L_causal, T, N_TRIALS, SEED,
                                     r=logistic_r_3)
    x_use_c = x_causal[:, BURNIN:, :]
    log_array_stats("causal x_use", x_use_c, ["U", "X", "Y"])

    plot_traces(x_use_c[0], ["U", "X", "Y"],
                f"Causal chain ({RNN_MODE}): U->X(lag0)->Y(lag{CI_DELAY})",
                OUT, "traces_causal")

    series_c = x_use_c.reshape(-1, N_nodes).copy()
    series_c = (series_c - series_c.mean(axis=0)) / series_c.std(axis=0)
    log_array_stats("causal z-scored", series_c, ["U", "X", "Y"])

    pairs_c = np.array([[0, 1], [0, 2], [1, 0], [2, 0], [1, 2], [2, 1]], dtype=np.int32)
    save_c = str(OUT / "causal_ccm") + "/"
    shutil.rmtree(save_c, ignore_errors=True)
    Path(save_c).mkdir(parents=True, exist_ok=True)

    _t0 = time.time()
    print("\n  Running eCCM (causal chain)...")
    ccmout_c = run_eccm(series_c, lags, pairs_c, save_c,
                        dim=CCM_DIM, kfolds=CCM_KFOLDS,
                        n_surrogates=CCM_NSURR, delay=CCM_DELAY)
    print(f"  CCM finished in {time.time() - _t0:.1f}s")

    causal_specs = [
        dict(key="ux", src_idx=0, tgt_idx=1, row_label="U->X"),
        dict(key="uy", src_idx=0, tgt_idx=2, row_label="U->Y"),
        dict(key="xu", src_idx=1, tgt_idx=0, row_label="X->U"),
        dict(key="yu", src_idx=2, tgt_idx=0, row_label="Y->U"),
        dict(key="xy", src_idx=1, tgt_idx=2, row_label="X->Y"),
        dict(key="yx", src_idx=2, tgt_idx=1, row_label="Y->X"),
    ]
    res_c = plot_diagnostics_grid(
        ccmout_c, lags, x_use_c, causal_specs,
        title="Causal diagnostics",
        out_dir=OUT, fname="causal_diagnostics")
    res_ux_c = res_c["ux"]
    res_uy_c = res_c["uy"]
    res_xu_c = res_c["xu"]
    res_yu_c = res_c["yu"]
    res_xy_c = res_c["xy"]
    res_yx_c = res_c["yx"]

    print(f"\n  U->X: peak_lag={res_ux_c['peak_lag']:+d}  "
          f"peak={res_ux_c['peak_val']:.3f}  p={res_ux_c['pval']:.2g}")
    print(f"  U->Y: peak_lag={res_uy_c['peak_lag']:+d}  "
          f"peak={res_uy_c['peak_val']:.3f}  p={res_uy_c['pval']:.2g}")
    print(f"  X->U: peak_lag={res_xu_c['peak_lag']:+d}  "
          f"peak={res_xu_c['peak_val']:.3f}  p={res_xu_c['pval']:.2g}")
    print(f"  Y->U: peak_lag={res_yu_c['peak_lag']:+d}  "
          f"peak={res_yu_c['peak_val']:.3f}  p={res_yu_c['pval']:.2g}")
    print(f"\n  X->Y: peak_lag={res_xy_c['peak_lag']:+d}  "
          f"peak={res_xy_c['peak_val']:.3f}  p={res_xy_c['pval']:.2g}")
    print(f"  Y->X: peak_lag={res_yx_c['peak_lag']:+d}  "
          f"peak={res_yx_c['peak_val']:.3f}  p={res_yx_c['pval']:.2g}")

    # ------------------------------------------------------------------
    # Scenario B: Common input  U -> X (lag 0), U -> Y (lag L), no X->Y
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"SCENARIO B: Common input (U={CI_MODE}, XY={RNN_MODE})  "
          f"U -> X (lag 0), U -> Y (lag {CI_DELAY})")
    print("=" * 60)

    if CI_MODE == RNN_MODE and CI_MODE in {"ricker", "logistic"}:
        C_ci = np.zeros((N_nodes, N_nodes), dtype=np.float64)
        L_ci = np.zeros((N_nodes, N_nodes), dtype=np.int32)
        couple = R_COUPLE if CI_MODE == "ricker" else LOGISTIC_COUPLE
        C_ci[1, 0] = couple
        C_ci[2, 0] = couple
        L_ci[1, 0] = 0
        L_ci[2, 0] = CI_DELAY

        if CI_MODE == "ricker":
            R_ci = C_ci.copy()
            for i in range(N_nodes):
                R_ci[i, i] = r_self_3[i]
                L_ci[i, i] = 0
            print(f"\n  R =\n{R_ci}")
            print(f"  L =\n{L_ci}")
            x_ci = simulate_ricker(R_ci, L_ci, T, N_TRIALS, SEED + 1)
        else:
            print(f"\n  W =\n{C_ci}")
            print(f"  L =\n{L_ci}")
            print(f"  per-unit r = {logistic_r_3}")
            print(f"  self weights = {1.0 - C_ci.sum(axis=1)}")
            x_ci = simulate_logistic(C_ci, L_ci, T, N_TRIALS, SEED + 1,
                                     r=logistic_r_3)
        x_use_ci = x_ci[:, BURNIN:, :]
        trace_names = ["U", "X", "Y"]
        log_array_stats("CI x_use", x_use_ci, trace_names)

    elif CI_MODE == "ou" and RNN_MODE == "ricker":
        N_ci = 2  # X=0, Y=1
        R_ci2 = np.zeros((N_ci, N_ci), dtype=np.float64)
        L_ci2 = np.zeros((N_ci, N_ci), dtype=np.int32)
        for i in range(N_ci):
            R_ci2[i, i] = r_self_2[i]
            L_ci2[i, i] = 0

        c_ou = generate_ou(N_TRIALS, T, rho=OU_RHO, sigma_eps=OU_SIGMA,
                           seed=SEED + 100)
        u_ou = np.zeros((N_TRIALS, T, N_ci), dtype=np.float64)
        u_ou[:, :, 0] = OU_ALPHA * c_ou
        u_ou[:, CI_DELAY:, 1] = OU_ALPHA * c_ou[:, :T - CI_DELAY]
        log_array_stats("OU drive u", u_ou, ["u_X", "u_Y"])

        x_ci_raw = simulate_ricker_driven(R_ci2, L_ci2, u_ou, seed=SEED + 2)
        c_use = c_ou[:, BURNIN:]
        x_use_xy = x_ci_raw[:, BURNIN:, :]
        x_use_ci = np.concatenate([c_use[:, :, np.newaxis], x_use_xy], axis=2)
        trace_names = ["c (OU)", "X", "Y"]
        log_array_stats("CI x_use", x_use_ci, trace_names)

    else:
        u_ci = generate_ci_source(
            CI_MODE, T, N_TRIALS, SEED + 100, R_SELF, LOGISTIC_R, OU_RHO, OU_SIGMA
        )
        if RNN_MODE == "logistic":
            x_ci_xy = simulate_logistic_common_input(
                u_ci, CI_MODE, T, N_TRIALS, SEED + 101, CI_DELAY,
                logistic_r_2, LOGISTIC_COUPLE, OU_ALPHA,
            )
        else:
            x_ci_xy = simulate_ricker_common_input(
                u_ci, CI_MODE, T, N_TRIALS, SEED + 101, CI_DELAY,
                r_self_2, R_COUPLE, OU_ALPHA,
            )
        x_use_ci = np.concatenate(
            [u_ci[:, BURNIN:, np.newaxis], x_ci_xy[:, BURNIN:, :]], axis=2
        )
        trace_names = ["c (OU)", "X", "Y"] if CI_MODE == "ou" else ["U", "X", "Y"]
        print(f"\n  Mixed-mode CI: U={CI_MODE}, XY={RNN_MODE}, X/Y disconnected")
        log_array_stats("CI x_use", x_use_ci, trace_names)

    plot_traces(x_use_ci[0], trace_names,
                f"Common input (U={CI_MODE}, XY={RNN_MODE}): U->X(lag0), U->Y(lag{CI_DELAY})",
                OUT, "traces_ci")

    # Z-score the full U/X/Y stack so U-based diagnostics use the same run.
    series_ci = x_use_ci.reshape(-1, x_use_ci.shape[-1]).copy()
    series_ci = (series_ci - series_ci.mean(axis=0)) / series_ci.std(axis=0)
    log_array_stats("CI z-scored", series_ci, trace_names)

    pairs_ci = np.array([[0, 1], [0, 2], [1, 0], [2, 0], [1, 2], [2, 1]], dtype=np.int32)
    save_ci = str(OUT / "ci_ccm") + "/"
    shutil.rmtree(save_ci, ignore_errors=True)
    Path(save_ci).mkdir(parents=True, exist_ok=True)

    _t0 = time.time()
    print(f"\n  Running eCCM (common input, U={CI_MODE}, XY={RNN_MODE})...")
    ccmout_ci = run_eccm(series_ci, lags, pairs_ci, save_ci,
                         dim=CCM_DIM, kfolds=CCM_KFOLDS,
                         n_surrogates=CCM_NSURR, delay=CCM_DELAY)
    print(f"  CCM finished in {time.time() - _t0:.1f}s")

    x_diag = x_use_ci
    ci_specs = [
        dict(key="ux", src_idx=0, tgt_idx=1, row_label="U->X"),
        dict(key="uy", src_idx=0, tgt_idx=2, row_label="U->Y"),
        dict(key="xu", src_idx=1, tgt_idx=0, row_label="X->U"),
        dict(key="yu", src_idx=2, tgt_idx=0, row_label="Y->U"),
        dict(key="xy", src_idx=1, tgt_idx=2, row_label="X->Y"),
        dict(key="yx", src_idx=2, tgt_idx=1, row_label="Y->X"),
    ]
    res_ci = plot_diagnostics_grid(
        ccmout_ci, lags, x_diag, ci_specs,
        title=f"Common-input diagnostics (U={CI_MODE}, XY={RNN_MODE})",
        out_dir=OUT, fname="ci_diagnostics")
    res_ux_ci = res_ci["ux"]
    res_uy_ci = res_ci["uy"]
    res_xu_ci = res_ci["xu"]
    res_yu_ci = res_ci["yu"]
    res_xy_ci = res_ci["xy"]
    res_yx_ci = res_ci["yx"]

    print(f"\n  U->X: peak_lag={res_ux_ci['peak_lag']:+d}  "
          f"peak={res_ux_ci['peak_val']:.3f}  p={res_ux_ci['pval']:.2g}")
    print(f"  U->Y: peak_lag={res_uy_ci['peak_lag']:+d}  "
          f"peak={res_uy_ci['peak_val']:.3f}  p={res_uy_ci['pval']:.2g}")
    print(f"  X->U: peak_lag={res_xu_ci['peak_lag']:+d}  "
          f"peak={res_xu_ci['peak_val']:.3f}  p={res_xu_ci['pval']:.2g}")
    print(f"  Y->U: peak_lag={res_yu_ci['peak_lag']:+d}  "
          f"peak={res_yu_ci['peak_val']:.3f}  p={res_yu_ci['pval']:.2g}")
    print(f"  X->Y: peak_lag={res_xy_ci['peak_lag']:+d}  "
          f"peak={res_xy_ci['peak_val']:.3f}  p={res_xy_ci['pval']:.2g}")
    print(f"  Y->X: peak_lag={res_yx_ci['peak_lag']:+d}  "
          f"peak={res_yx_ci['peak_val']:.3f}  p={res_yx_ci['pval']:.2g}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'':30s} {'peak_lag':>10s} {'peak_FCF':>10s} {'p-value':>10s}")
    print("-" * 62)
    for name, res in [
        ("Causal: U->X", res_ux_c),
        ("Causal: U->Y", res_uy_c),
        ("Causal: X->U", res_xu_c),
        ("Causal: Y->U", res_yu_c),
        ("Causal: X->Y", res_xy_c),
        ("Causal: Y->X", res_yx_c),
        (f"CI({CI_MODE}|{RNN_MODE}): U->X", res_ux_ci),
        (f"CI({CI_MODE}|{RNN_MODE}): U->Y", res_uy_ci),
        (f"CI({CI_MODE}|{RNN_MODE}): X->U", res_xu_ci),
        (f"CI({CI_MODE}|{RNN_MODE}): Y->U", res_yu_ci),
        (f"CI({CI_MODE}|{RNN_MODE}): X->Y", res_xy_ci),
        (f"CI({CI_MODE}|{RNN_MODE}): Y->X", res_yx_ci),
    ]:
        print(f"  {name:28s} {res['peak_lag']:+10d} {res['peak_val']:10.3f} {res['pval']:10.2g}")

    print(f"\nAll outputs in: {OUT}")
    print("\nExpected behavior:")
    print("  Causal U->X: positive peak lag (direct)")
    print("  Causal U->Y: positive peak lag (indirect via X)")
    print("  Causal X->U: negative peak lag or non-significant")
    print("  Causal Y->U: negative peak lag or non-significant")
    print(f"  Causal X->Y: positive peak lag (= causal delay {CI_DELAY})")
    print("  CI     U->X: positive peak lag")
    print(f"  CI     U->Y: positive peak lag (= common-input delay {CI_DELAY})")
    print("  CI     X->U: negative peak lag or non-significant")
    print("  CI     Y->U: negative peak lag or non-significant")
    print("  Causal Y->X: negative peak lag or non-significant")
    print("  CI     X->Y: depends on CI mechanism (confound vs rejection)")
    print("  CI     Y->X: negative peak lag or non-significant")
    print("\nCode convention: positive lag = look at past = Ye et al.'s NEGATIVE tau")

    print(f"\nTotal runtime: {time.time() - _t0_global:.1f}s")
    print(f"Log saved to: {OUT / 'run.log'}")
    _log_fh.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
