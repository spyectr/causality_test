"""
sauer_sugihara_aux.py — DetC / DirC coupling-detection pipeline.

Implements the methods from:
  Sauer & Sugihara (2025), "Robust methods to detect coupling among
  nonlinear dynamic time series", Phys. Rev. E 111, 064208.

With statistical fixes:
  - Theiler window (ACF-based FWHM) to exclude temporally correlated NNs
  - Mean p-hat (j=3..10) as the single DetC summary statistic
  - Circular-shift surrogates for empirical null distribution
  - Sidak correction for min-over-lags multiple comparisons
  - Kendall tau (one-sided) for DirC instead of OLS slope
  - DirC computed in both directions for coupling classification

Key public functions
--------------------
run_detc_dirc          — full pipeline (drop-in replacement for run_eccm)
plot_detc_dirc_grid    — 4-column diagnostic figure
"""
from __future__ import annotations

import warnings
from typing import Dict, Tuple

import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy import stats as sp_stats
from scipy.spatial.distance import cdist

# ── Optional numba acceleration (matches Paper_sim_eCCM_aux.py convention) ──
try:
    from numba import njit, prange
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False

    def njit(*_args, **_kwargs):  # type: ignore
        def _decorator(func):
            return func
        return _decorator

    def prange(*args, **kwargs):  # type: ignore
        return range(*args, **kwargs)


# ============================================================================
# 0. Numba kernels — inner loops of compute_detc / compute_dirc
# ============================================================================
# These two kernels replace the two big numpy hotspots in the previous
# vectorised implementation:
#   • `np.argpartition(D, nn, axis=1)[:, :nn]`  (~32 ms @ 500×9500, nn=12)
#     → max-heap of size nn per row, parallel across rows (~1.2 ms, ~26×)
#   • 12-iteration broadcast loop `np.sum(D < cross_d[:, j:j+1], axis=1)`
#     (~73 ms)  →  tight triple loop, parallel across rows (~6 ms, ~12×)
#
# When numba is unavailable we fall back to the pure-numpy versions used
# previously so the module still imports and runs everywhere.

@njit(parallel=True, fastmath=True)
def _k_smallest_idx_numba(D, k):
    """Row-wise indices of the k smallest entries (unsorted within k).

    Uses a size-k binary max-heap: scan all N columns once per row, O(N log k).
    Parallelises across rows via prange.  Matches np.argpartition's guarantee
    (first k entries are the k smallest as a SET, internal order arbitrary).
    """
    n, N = D.shape
    out = np.empty((n, k), dtype=np.int64)
    for i in prange(n):
        heap_v = np.empty(k, dtype=np.float64)
        heap_i = np.empty(k, dtype=np.int64)
        # Seed with first k elements
        for j in range(k):
            heap_v[j] = D[i, j]
            heap_i[j] = j
        # Heapify (max-heap: parent >= children)
        for start in range(k // 2 - 1, -1, -1):
            pos = start
            while True:
                l = 2 * pos + 1
                r = 2 * pos + 2
                big = pos
                if l < k and heap_v[l] > heap_v[big]:
                    big = l
                if r < k and heap_v[r] > heap_v[big]:
                    big = r
                if big == pos:
                    break
                heap_v[pos], heap_v[big] = heap_v[big], heap_v[pos]
                heap_i[pos], heap_i[big] = heap_i[big], heap_i[pos]
                pos = big
        # Scan remaining columns; push into heap if smaller than current max
        for idx in range(k, N):
            v = D[i, idx]
            if v < heap_v[0]:
                heap_v[0] = v
                heap_i[0] = idx
                pos = 0
                while True:
                    l = 2 * pos + 1
                    r = 2 * pos + 2
                    big = pos
                    if l < k and heap_v[l] > heap_v[big]:
                        big = l
                    if r < k and heap_v[r] > heap_v[big]:
                        big = r
                    if big == pos:
                        break
                    heap_v[pos], heap_v[big] = heap_v[big], heap_v[pos]
                    heap_i[pos], heap_i[big] = heap_i[big], heap_i[pos]
                    pos = big
        for j in range(k):
            out[i, j] = heap_i[j]
    return out


@njit(parallel=True, fastmath=True)
def _rank_rowwise_numba(D, cross_d):
    """Row-wise rank of each cross_d[i, j] among entries of D[i, :].

    rank[i, j] = #{k : D[i, k] < cross_d[i, j]}.
    Triple loop, parallel across rows.  inf entries in D naturally contribute
    zero (inf < finite = False); inf entries in cross_d give rank = n_valid.
    """
    n_valid, N = D.shape
    nn = cross_d.shape[1]
    ranks = np.empty((n_valid, nn), dtype=np.int32)
    for i in prange(n_valid):
        for j in range(nn):
            cv = cross_d[i, j]
            cnt = 0
            for k in range(N):
                if D[i, k] < cv:
                    cnt += 1
            ranks[i, j] = cnt
    return ranks


def _k_smallest_idx(D, k):
    """Dispatcher: numba kernel if available, else numpy argpartition."""
    if _HAS_NUMBA:
        return _k_smallest_idx_numba(D, k)
    return np.argpartition(D, k, axis=1)[:, :k]


def _rank_rowwise(D, cross_d):
    """Dispatcher: numba kernel if available, else numpy broadcast loop."""
    if _HAS_NUMBA:
        return _rank_rowwise_numba(D, cross_d)
    n_valid, nn = cross_d.shape
    ranks = np.empty((n_valid, nn), dtype=np.int32)
    for _j in range(nn):
        ranks[:, _j] = np.sum(D < cross_d[:, _j : _j + 1], axis=1)
    return ranks


# ============================================================================
# 1. Constants
# ============================================================================
J_MIN: int = 3        # first neighbor rank (paper convention; j=1,2 too noisy)
J_MAX: int = 10       # last neighbor rank
NN_MAX: int = 12      # total neighbors retrieved (>= J_MAX)
DIRC_ALPHA: float = 0.05   # significance level for DirC classification

# ── Plot style (matching Paper_sim_common.py) ──
S_UNIT = 8.5 / 5.0
_C_UV = "#009E73"     # bluish-green  (DetC / DirC U→V)
_C_VU = "#D55E00"     # vermillion    (DirC V→U)
_C_SURR = "#888888"   # surrogate null
_C_PEAK = "#CC0000"   # peak marker
_LW = 2.0
_LW_FOLD = 0.8
_MS = 5
_FS = 8
_FS_LG = 6
_FS_AN = 6
_LEGEND_KW = dict(fontsize=_FS_LG, frameon=False, borderpad=0.3,
                  handlelength=1.6, handletextpad=0.5)


def set_plot_style(fontsize: float = _FS) -> None:
    """Lightweight rcParams setup."""
    import matplotlib as _mpl
    _mpl.rcParams.update({
        "font.size": fontsize, "axes.labelsize": fontsize,
        "axes.titlesize": fontsize,
        "xtick.labelsize": fontsize - 1,
        "ytick.labelsize": fontsize - 1,
        "legend.fontsize": fontsize - 2,
        "lines.linewidth": _LW, "lines.markersize": _MS,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    })


# ============================================================================
# 2. Theiler window (ACF-based FWHM)
# ============================================================================
def _acf_fwhm(z: np.ndarray, max_lag: int | None = None) -> int:
    """Half-width at half-max of |ACF| for a 1-D series."""
    z = np.asarray(z, dtype=np.float64).ravel()
    N = z.size
    if max_lag is None:
        max_lag = min(N // 2, 500)
    z = z - z.mean()
    var = np.dot(z, z)
    if var == 0.0:
        return 1
    for h in range(1, max_lag + 1):
        acf_h = np.dot(z[:N - h], z[h:]) / var
        if abs(acf_h) < 0.5:
            return h
    return max_lag


def theiler_window(
    u: np.ndarray, v: np.ndarray, e: int,
    *, trial_len: int | None = None,
) -> int:
    """w = max(FWHM_u, FWHM_v, e).

    If *trial_len* is given the ACF is computed per trial and averaged.
    """
    u = np.asarray(u, dtype=np.float64).ravel()
    v = np.asarray(v, dtype=np.float64).ravel()
    if trial_len is not None and trial_len > 0:
        n_tr = u.size // trial_len
        wu = int(np.round(np.mean([_acf_fwhm(u[i * trial_len:(i + 1) * trial_len])
                                    for i in range(n_tr)])))
        wv = int(np.round(np.mean([_acf_fwhm(v[i * trial_len:(i + 1) * trial_len])
                                    for i in range(n_tr)])))
    else:
        wu, wv = _acf_fwhm(u), _acf_fwhm(v)
    return max(wu, wv, e)


# ============================================================================
# 3. Delay embedding
# ============================================================================
def make_delay_vectors(x: np.ndarray, e: int, tau: int = 1) -> np.ndarray:
    """Embed 1-D *x* into (N_vec, e) matrix via stride trick (contiguous copy).

    The delay vector at index *i* is
        [ x[i], x[i+tau], x[i+2*tau], ..., x[i+(e-1)*tau] ].
    *tau* is the embedding stride in **samples** (not seconds).  The caller
    is responsible for converting a physical-time delay into samples
    via ``tau = int(round(tau_sec / dt))``.  Default ``tau=1`` preserves
    the legacy behaviour used for discrete maps.
    """
    x = np.ascontiguousarray(x, dtype=np.float64).ravel()
    tau = int(tau)
    if tau < 1:
        raise ValueError(f"emb_tau must be >= 1 sample, got {tau}")
    span = (e - 1) * tau
    N_vec = x.size - span
    if N_vec < 1:
        return np.empty((0, e), dtype=np.float64)
    s = x.strides[0]
    out = as_strided(x, shape=(N_vec, e), strides=(s, s * tau))
    return out.copy()


# ============================================================================
# 4. DetC core — per-reference-point, memory-friendly
# ============================================================================
def _theiler_mask_1d(d: np.ndarray, t: int, w: int) -> None:
    """Set d[s] = inf for |s - t| < w  (in-place)."""
    lo = max(0, t - w + 1)
    hi = min(d.size, t + w)
    d[lo:hi] = np.inf


def _detc_one_ref(
    u_emb: np.ndarray, v_emb: np.ndarray,
    t: int, w: int, nn: int, j_min: int, j_max: int,
) -> np.ndarray | None:
    """p-hat_j(t) for a single reference point *t*.

    Returns (n_j,) or None if too few valid neighbours.
    """
    N_vec = u_emb.shape[0]

    # ── V: find nn NNs ──
    d_V = np.linalg.norm(v_emb - v_emb[t], axis=1)
    _theiler_mask_1d(d_V, t, w)
    n_finite_V = int(np.sum(np.isfinite(d_V)))
    if n_finite_V < nn:
        return None
    nn_idx = np.argpartition(d_V, nn)[:nn]          # unsorted nn indices

    # ── U: rank cross-distances ──
    d_U = np.linalg.norm(u_emb - u_emb[t], axis=1)
    _theiler_mask_1d(d_U, t, w)
    finite_U = np.isfinite(d_U)
    n_valid = int(finite_U.sum())
    if n_valid < nn:
        return None
    d_U_sorted = np.sort(d_U[finite_U])

    cross_d = d_U[nn_idx]                           # (nn,)
    cross_ranks = np.searchsorted(d_U_sorted, cross_d, side="left")
    cross_ranks_sorted = np.sort(cross_ranks)        # j-th smallest

    j_arr = np.arange(j_min, j_max + 1, dtype=np.float64)
    r_j = cross_ranks_sorted[j_arr.astype(int) - 1] / max(n_valid, 1)
    phat_t = (nn + 1.0) * r_j / j_arr
    return phat_t


def compute_detc(
    u_emb: np.ndarray, v_emb: np.ndarray, w: int,
    nn: int = NN_MAX, j_min: int = J_MIN, j_max: int = J_MAX,
    max_ref: int = 500,
) -> Tuple[np.ndarray, float]:
    """DetC for one pair at one lag (one fold) — fully vectorised.

    Strategy: one cdist call gives all (max_ref × N_vec) distances at once;
    Theiler masking is a fast slice-assignment loop; argpartition + numpy
    broadcasting replace the Python loop over reference points.

    Memory: ~2 × max_ref × N_vec × 8 bytes (two float64 distance matrices)
    plus a (CHUNK × nn × N_vec) bool array for the rank computation.
    Typical footprint: ~80 MB for max_ref=500, N_vec=9 500.

    Returns (phat_j, phat_mean).  phat_j has shape (n_j,).
    """
    N_vec = u_emb.shape[0]
    n_j = j_max - j_min + 1
    if N_vec < nn + w:
        return np.ones(n_j, dtype=np.float64), 1.0

    n_ref = min(N_vec, max_ref)
    ref_idx = np.round(np.linspace(0, N_vec - 1, n_ref)).astype(int)

    # ── Squared-Euclidean distance matrices (ordering-invariant for ranking) ──
    D_V = cdist(v_emb[ref_idx], v_emb, metric="sqeuclidean")  # (n_ref, N_vec)
    D_U = cdist(u_emb[ref_idx], u_emb, metric="sqeuclidean")  # (n_ref, N_vec)

    # ── Theiler masking + analytical valid-neighbour count ──
    # Both D_V and D_U get the same window masked, so n_fin_V == n_fin_U;
    # we compute n_u analytically (window size = hi-lo) avoiding two
    # expensive np.sum(np.isfinite(...), axis=1) calls.
    n_u_all = np.empty(n_ref, dtype=np.int32)
    for i, t in enumerate(ref_idx):
        lo, hi = max(0, t - w + 1), min(N_vec, t + w)
        D_V[i, lo:hi] = np.inf
        D_U[i, lo:hi] = np.inf
        n_u_all[i] = N_vec - (hi - lo)

    valid = n_u_all >= nn
    if not np.any(valid):
        return np.ones(n_j, dtype=np.float64), 1.0

    D_V_v = D_V[valid]        # (n_valid, N_vec)
    D_U_v = D_U[valid]        # (n_valid, N_vec)
    n_u   = n_u_all[valid]    # (n_valid,)  — denominator for percentile rank
    n_valid = int(np.sum(valid))

    # ── nn nearest neighbours in V-space (unsorted within the nn set) ──
    # Numba kernel: size-nn max-heap per row, parallel over rows (~26× over numpy).
    nn_idx = _k_smallest_idx(D_V_v, nn)                    # (n_valid, nn) int64

    # ── U-space distances at those nn indices (the "cross-distances") ──
    cross_d = np.take_along_axis(D_U_v, nn_idx, axis=1)    # (n_valid, nn)

    # ── Rank each cross-distance among *all* U-distances from the same ref pt ──
    # ranks[i, j] = #{k : D_U_v[i,k] < cross_d[i,j]}.  inf in D_U_v excluded
    # naturally; inf in cross_d (Theiler-masked U-NN) gives rank = n_u[i] → p-hat ≫ 1.
    # Numba kernel: tight triple loop, parallel over rows (~12× over numpy).
    ranks = _rank_rowwise(D_U_v, cross_d)                  # (n_valid, nn) int32

    r_pct = ranks.astype(np.float64) / np.maximum(n_u[:, None], 1)  # (n_valid, nn)

    # ── Sort cross-ranks per ref point — DetC uses the j-th SMALLEST ──
    cross_ranks_sorted = np.sort(r_pct, axis=1)           # (n_valid, nn)

    # ── Select j = j_min .. j_max and compute p-hat per ref point ──
    j_arr = np.arange(j_min, j_max + 1, dtype=np.float64)            # (n_j,)
    r_j   = cross_ranks_sorted[:, j_arr.astype(int) - 1]             # (n_valid, n_j)
    phat_t = (nn + 1.0) * r_j / j_arr[None, :]                       # (n_valid, n_j)

    phat_j = np.mean(phat_t, axis=0)                                  # (n_j,)
    return phat_j, float(np.mean(phat_j))


# ============================================================================
# 5. DirC core — per-reference-point
# ============================================================================
def _dirc_one_ref(
    u_emb: np.ndarray, v_emb: np.ndarray,
    t: int, w: int, nn: int, j_min: int, j_max: int,
) -> np.ndarray | None:
    """Percentile rank r(j) for the j-th NN, single reference *t*.

    Returns (n_j,) or None.
    """
    N_vec = u_emb.shape[0]

    d_V = np.linalg.norm(v_emb - v_emb[t], axis=1)
    _theiler_mask_1d(d_V, t, w)
    n_finite_V = int(np.sum(np.isfinite(d_V)))
    if n_finite_V < nn:
        return None
    # Sorted NNs in V (need order: 1st NN, 2nd NN, ...)
    nn_idx = np.argpartition(d_V, nn)[:nn]
    nn_order = np.argsort(d_V[nn_idx])
    nn_sorted = nn_idx[nn_order]                     # sorted by V-distance

    d_U = np.linalg.norm(u_emb - u_emb[t], axis=1)
    _theiler_mask_1d(d_U, t, w)
    finite_U = np.isfinite(d_U)
    n_valid = int(finite_U.sum())
    if n_valid < nn:
        return None
    d_U_sorted = np.sort(d_U[finite_U])

    j_arr = np.arange(j_min, j_max + 1, dtype=int)
    nn_at_j = nn_sorted[j_arr - 1]                  # j-th NN indices
    cross_d = d_U[nn_at_j]
    ranks = np.searchsorted(d_U_sorted, cross_d, side="left")
    r_pct = ranks.astype(np.float64) / max(n_valid, 1)
    return r_pct


def compute_dirc(
    u_emb: np.ndarray, v_emb: np.ndarray, w: int,
    nn: int = NN_MAX, j_min: int = J_MIN, j_max: int = J_MAX,
    max_ref: int = 500,
) -> Tuple[np.ndarray, float]:
    """DirC for one direction at one lag (one fold) — fully vectorised.

    Unlike DetC, DirC needs the j-th NEAREST V-neighbour (ordered by V-
    distance), not the j-th smallest cross-rank.  After finding the nn NNs
    with argpartition, we sort them by V-distance to get the ordered list,
    then pick the j-th one and rank its U-distance.

    Returns (rbar_j, tau_k).  rbar_j has shape (n_j,).
    """
    N_vec = u_emb.shape[0]
    n_j = j_max - j_min + 1
    if N_vec < nn + w:
        return 0.5 * np.ones(n_j, dtype=np.float64), 0.0

    n_ref = min(N_vec, max_ref)
    ref_idx = np.round(np.linspace(0, N_vec - 1, n_ref)).astype(int)

    D_V = cdist(v_emb[ref_idx], v_emb, metric="sqeuclidean")  # (n_ref, N_vec)
    D_U = cdist(u_emb[ref_idx], u_emb, metric="sqeuclidean")  # (n_ref, N_vec)

    n_u_all = np.empty(n_ref, dtype=np.int32)
    for i, t in enumerate(ref_idx):
        lo, hi = max(0, t - w + 1), min(N_vec, t + w)
        D_V[i, lo:hi] = np.inf
        D_U[i, lo:hi] = np.inf
        n_u_all[i] = N_vec - (hi - lo)

    valid = n_u_all >= nn
    if not np.any(valid):
        return 0.5 * np.ones(n_j, dtype=np.float64), 0.0

    D_V_v = D_V[valid]
    D_U_v = D_U[valid]
    n_u    = n_u_all[valid]
    n_valid = int(np.sum(valid))

    # ── nn NNs in V (unsorted within the set) — numba k-smallest kernel ──
    nn_idx_us = _k_smallest_idx(D_V_v, nn)                       # (n_valid, nn) int64

    # ── Sort those nn indices by their V-distances → j-th nearest is [j-1] ──
    V_nn_d    = np.take_along_axis(D_V_v, nn_idx_us, axis=1)    # (n_valid, nn)
    sort_ord  = np.argsort(V_nn_d, axis=1)                       # (n_valid, nn)
    nn_sorted = np.take_along_axis(nn_idx_us, sort_ord, axis=1)  # (n_valid, nn) sorted

    # ── j-th nearest V-neighbour (1-indexed, j_min..j_max) ──
    j_arr   = np.arange(j_min, j_max + 1)                        # (n_j,)
    nn_at_j = nn_sorted[:, j_arr - 1].astype(np.int64)           # (n_valid, n_j)

    # ── U-distances at the j-th V-NN positions ──
    cross_d = np.take_along_axis(D_U_v, nn_at_j, axis=1)        # (n_valid, n_j)

    # ── Rank each cross-distance among U-distances (numba kernel) ──
    ranks = _rank_rowwise(D_U_v, cross_d)                        # (n_valid, n_j) int32

    r_pct  = ranks.astype(np.float64) / np.maximum(n_u[:, None], 1)  # (n_valid, n_j)
    rbar_j = np.mean(r_pct, axis=0)                                    # (n_j,)

    if n_j >= 3:
        tau_k, _ = sp_stats.kendalltau(j_arr, rbar_j)
        if not np.isfinite(tau_k):
            tau_k = 0.0
    else:
        tau_k = 0.0
    return rbar_j, float(tau_k)


# ============================================================================
# 6. Circular-shift surrogates (per-trial)
# ============================================================================
def circular_shift_surrogates(
    v_trials: np.ndarray,
    n_surr: int,
    min_shift: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Generate *n_surr* circular-shift surrogates of trial-structured V.

    Parameters
    ----------
    v_trials : (n_trials, T) — one row per trial
    n_surr   : number of surrogates
    min_shift: minimum shift (>= Theiler window)
    rng      : numpy Generator

    Returns list of *n_surr* arrays, each (n_trials, T).
    """
    n_trials, T = v_trials.shape
    hi = max(min_shift + 1, T - min_shift)
    surrogates: list[np.ndarray] = []
    for _ in range(n_surr):
        surr = np.empty_like(v_trials)
        for tr in range(n_trials):
            shift = int(rng.integers(min_shift, hi))
            surr[tr] = np.roll(v_trials[tr], shift)
        surrogates.append(surr)
    return surrogates


# ============================================================================
# 7. Lag alignment
# ============================================================================
def align_lag(u: np.ndarray, v: np.ndarray, tau: int):
    """Return (u_aligned, v_aligned) for lag tau.

    Convention: positive tau ⟹ V is shifted forward: U_t  vs  V_{t+tau}.
    """
    u = np.asarray(u, dtype=np.float64).ravel()
    v = np.asarray(v, dtype=np.float64).ravel()
    T = min(u.size, v.size)
    if tau > 0:
        return u[:T - tau].copy(), v[tau:T].copy()
    if tau < 0:
        return u[-tau:T].copy(), v[:T + tau].copy()
    return u[:T].copy(), v[:T].copy()


# ============================================================================
# 8. Scoring
# ============================================================================
def sidak_correct(pval, n_tests: int):
    """Sidak: p_adj = 1 - (1 - p_raw)^n_tests."""
    p = np.clip(np.asarray(pval, dtype=np.float64), 0.0, 1.0)
    return 1.0 - (1.0 - p) ** n_tests


def gaussian_pvalue_lower(obs: float, null: np.ndarray):
    """Lower-tail p-value (coupling ⟹ p-hat < 1)."""
    null = np.asarray(null, dtype=np.float64).ravel()
    null = null[np.isfinite(null)]
    if null.size < 2:
        return 1.0, np.nan, np.nan
    mu, sig = float(null.mean()), float(null.std(ddof=1))
    if sig <= 0:
        pv = 0.0 if obs < mu else 1.0
    else:
        pv = float(sp_stats.norm.cdf(obs, loc=mu, scale=sig))
    pv = float(np.clip(pv, np.finfo(np.float64).tiny, 1.0))
    return pv, mu, sig


def _ranksum_below(obs_line: np.ndarray, surr_cloud: np.ndarray) -> float:
    """One-sided Wilcoxon rank-sum test: is *obs_line* stochastically
    smaller than *surr_cloud*?

    Parameters
    ----------
    obs_line   : (n_j,)                  observed rbar_j values
    surr_cloud : (n_surrogates, n_j)     surrogate rbar values

    Returns
    -------
    p-value. Under coupling, observed rbar is systematically lower than
    the surrogate cloud (low U-rank for near-V-neighbours), so
    ``alternative='less'`` on scipy.stats.ranksums detects this.
    NaN inputs are dropped. If either sample has fewer than 2 finite
    values, returns 1.0 (cannot reject H0).
    """
    obs_flat  = np.asarray(obs_line,  dtype=np.float64).ravel()
    surr_flat = np.asarray(surr_cloud, dtype=np.float64).ravel()
    obs_flat  = obs_flat[np.isfinite(obs_flat)]
    surr_flat = surr_flat[np.isfinite(surr_flat)]
    if obs_flat.size < 2 or surr_flat.size < 2:
        return 1.0
    try:
        _, pv = sp_stats.ranksums(obs_flat, surr_flat, alternative="less")
    except TypeError:
        # scipy < 1.7: two-sided ranksums then halve for one-sided (lower).
        z, pv_two = sp_stats.ranksums(obs_flat, surr_flat)
        pv = 0.5 * pv_two if z < 0 else 1.0 - 0.5 * pv_two
    pv = float(np.clip(pv, np.finfo(np.float64).tiny, 1.0))
    return pv


def gaussian_pvalue_upper(obs: float, null: np.ndarray):
    """Upper-tail p-value (DirC: positive tau ⟹ one-to-one)."""
    null = np.asarray(null, dtype=np.float64).ravel()
    null = null[np.isfinite(null)]
    if null.size < 2:
        return 1.0, np.nan, np.nan
    mu, sig = float(null.mean()), float(null.std(ddof=1))
    if sig <= 0:
        pv = 0.0 if obs > mu else 1.0
    else:
        pv = float(sp_stats.norm.sf(obs, loc=mu, scale=sig))
    pv = float(np.clip(pv, np.finfo(np.float64).tiny, 1.0))
    return pv, mu, sig


def _linreg_slope(j_arr: np.ndarray, rbar_j: np.ndarray) -> float:
    """OLS slope β of rbar_j on j_arr. Returns 0.0 if undefined/non-finite.

    Matches the Kendall-τ test's ``(j_arr, rbar_j)`` inputs exactly:
    under coupling, rbar_j rises with j ⟹ β > 0, so upper-tail surrogate
    p-value is the right analogue of the τ test.
    """
    j = np.asarray(j_arr, dtype=np.float64).ravel()
    y = np.asarray(rbar_j, dtype=np.float64).ravel()
    m = np.isfinite(j) & np.isfinite(y)
    if int(m.sum()) < 3:
        return 0.0
    j = j[m]; y = y[m]
    jm = j.mean()
    denom = float(np.sum((j - jm) ** 2))
    if denom <= 0.0:
        return 0.0
    num = float(np.sum((j - jm) * (y - y.mean())))
    beta = num / denom
    if not np.isfinite(beta):
        return 0.0
    return float(beta)


def classify_coupling(pval_UV: float, pval_VU: float,
                      alpha: float = DIRC_ALPHA) -> str:
    """Paper Table (p. 5): coupling classification from DirC p-values."""
    uv = pval_UV < alpha
    vu = pval_VU < alpha
    if uv and vu:
        return "U<->V/GS"
    if uv and not vu:
        return "U->V"
    if not uv and vu:
        return "V->U"
    return "latent"


def detect_coupling_strict(
    *,
    detc_pval_adj: float,
    best_lag: int,
    dirc_slope: float,
    dirc_pval_slope: float,
    dirc_pval_rs: float,
    alpha: float = DIRC_ALPHA,
    slope_name: str = "tau",
) -> tuple[bool, str]:
    """Strict per-direction detection rule for coupling A → B.

    All of the following must hold, evaluated for the ordered pair (U=A, V=B):

    1. DetC significant      : ``detc_pval_adj < alpha``
    2. DetC lag non-negative : ``best_lag >= 0``  (V lags U at the best lag)
    3. DirC slope positive   : ``dirc_slope > 0``
    4. DirC slope significant: ``dirc_pval_slope < alpha``
    5. DirC ranksum signif.  : ``dirc_pval_rs   < alpha``

    The slope statistic is selected by the caller (Kendall τ or OLS β);
    *slope_name* is only used to format the diagnostic *reason* string.
    """
    if not (detc_pval_adj < alpha):
        return False, f"DetC ns (p_adj={detc_pval_adj:.2g})"
    if not (best_lag >= 0):
        return False, f"best_lag<0 ({best_lag:+d})"
    if not (dirc_slope > 0):
        return False, f"{slope_name}<=0 ({dirc_slope:+.3g})"
    if not (dirc_pval_slope < alpha):
        return False, f"{slope_name} ns (p_{slope_name}={dirc_pval_slope:.2g})"
    if not (dirc_pval_rs < alpha):
        return False, f"ranksum ns (p_rs={dirc_pval_rs:.2g})"
    return True, "ok"


# ============================================================================
# 9. Cross-correlogram & sign
# ============================================================================
def pair_cross_correlogram(
    x_use: np.ndarray, src: int, tgt: int, lags: np.ndarray,
) -> np.ndarray:
    """Trial-averaged Pearson r at each lag.  x_use: (n_trials, T, N)."""
    x_use = np.asarray(x_use, dtype=np.float64)
    lags = np.asarray(lags, dtype=int).ravel()
    n_trials, T, _ = x_use.shape
    out = np.zeros(lags.size, dtype=np.float64)
    for li, tau in enumerate(lags):
        r_sum, n_ok = 0.0, 0
        for tr in range(n_trials):
            if tau > 0:
                a, b = x_use[tr, :T - tau, src], x_use[tr, tau:, tgt]
            elif tau < 0:
                a, b = x_use[tr, -tau:, src], x_use[tr, :T + tau, tgt]
            else:
                a, b = x_use[tr, :, src], x_use[tr, :, tgt]
            if a.size < 3:
                continue
            a, b = a - a.mean(), b - b.mean()
            den = np.sqrt(np.dot(a, a) * np.dot(b, b))
            if den > 0:
                r_sum += np.dot(a, b) / den
                n_ok += 1
        out[li] = r_sum / n_ok if n_ok else 0.0
    return out


def sign_at_lag(x_use: np.ndarray, src: int, tgt: int, tau: int) -> float:
    """Sign of the trial-averaged cross-correlation at lag *tau*."""
    r = pair_cross_correlogram(x_use, src, tgt, np.array([tau]))
    v = r[0]
    if not np.isfinite(v) or abs(v) < 1e-12:
        return 0.0
    return float(np.sign(v))


# ============================================================================
# 10. Main pipeline
# ============================================================================
def run_detc_dirc(
    x_use: np.ndarray,
    lags: np.ndarray,
    selected_pairs: np.ndarray,
    *,
    e: int = 8,
    emb_tau: int = 1,
    kfolds: int = 4,
    n_surrogates: int = 20,
    max_ref: int = 500,
    nn: int = NN_MAX,
    j_min: int = J_MIN,
    j_max: int = J_MAX,
    dirc_alpha: float = DIRC_ALPHA,
    seed: int = 42,
    verbose: bool = True,
    score_type: str = "pvalue",
    n_jobs: int = 1,
    dirc_VU: bool = False,
    dirc_slope_test: str = "kendall",
) -> dict:
    """Full DetC + DirC pipeline.

    Parameters
    ----------
    x_use : (n_trials, T_per_trial, N)  trial-structured time series
    lags  : 1-D int array of lags to sweep
    selected_pairs : (n_pairs, 2) int — [source, target] indices
    e, kfolds, n_surrogates, max_ref : pipeline hyper-parameters
    nn, j_min, j_max : DetC/DirC neighbour and rank settings
    dirc_alpha : significance level for DirC classification
    seed : random seed for surrogates
    n_jobs : int, default 1
        Number of parallel jobs for the pair loop.  Pass -1 to use all CPU
        cores.  Uses joblib's threading backend (numpy releases the GIL for
        BLAS/C calls, so threading gives true parallelism without pickle
        overhead and with shared memory for *x_use*).
    score_type : one of 'pvalue' | 'phat' | 'zscore'
        'pvalue'  — −log₁₀(p_adj)              [default; may saturate at ≈308]
        'phat'    — 1 − p̂_best                  [0 under null, > 0 = coupling]
        'zscore'  — (μ_null − p̂_best) / σ_null  [Gaussian z, unsigned]
        All three are always stored; this parameter selects which populates
        ``score`` and ``signed_score``.

    Returns dict with (N, N, ...) arrays filled at requested pairs.
    """
    x_use = np.asarray(x_use, dtype=np.float64)
    lags = np.asarray(lags, dtype=int).ravel()
    pairs = np.asarray(selected_pairs, dtype=int).reshape(-1, 2)

    n_trials, T_trial, N = x_use.shape
    n_lags = lags.size
    n_j = j_max - j_min + 1
    n_pairs = pairs.shape[0]

    # ── Trial-based K-fold indices ──
    fold_size = n_trials // kfolds
    fold_slices: list[np.ndarray] = []
    for k in range(kfolds):
        s = k * fold_size
        e_idx = s + fold_size if k < kfolds - 1 else n_trials
        fold_slices.append(np.arange(s, e_idx))

    # ── Allocate output arrays ──
    # Big per-pair arrays are stored DENSELY as (…, P, …) indexed by pair id
    # rather than sparsely (…, N, N, …). This avoids the 10+ GB of NaN
    # allocation at N=1024 when only a small subset of pairs is analysed.
    # Small (N, N) scalar-metadata arrays are still sparse for API clarity.
    # A lookup table  pair_idx[src, tgt] → pi (or -1)  maps (src, tgt) back
    # to the dense pair index.
    nan_2 = lambda: np.full((N, N), np.nan, dtype=np.float64)

    pair_idx = np.full((N, N), -1, dtype=np.int32)
    pair_idx[pairs[:, 0], pairs[:, 1]] = np.arange(n_pairs, dtype=np.int32)
    P = n_pairs

    out: Dict = dict(
        # ── Pair index bookkeeping ──
        selected_pairs=pairs.copy().astype(np.int32),          # (P, 2)
        pair_idx=pair_idx,                                     # (N, N) int32
        # ── DetC per-fold / per-lag (DENSE per-pair) ──
        phat_j_folds=np.full((kfolds, P, n_lags, n_j), np.nan),
        phat_mean_folds=np.full((kfolds, P, n_lags), np.nan),
        phat_j_avg=np.full((P, n_lags, n_j), np.nan),
        phat_mean_avg=np.full((P, n_lags), np.nan),
        # ── Surrogate DetC (DENSE per-pair) ──
        surr_phat_j=np.full((n_surrogates, P, n_j), np.nan),
        surr_phat_mean=np.full((n_surrogates, P), np.nan),
        # ── Scoring (scalar, sparse (N, N)) ──
        best_lag_idx=np.full((N, N), -1, dtype=int),
        best_lag=np.full((N, N), 0, dtype=int),
        best_phat=nan_2(),
        pval_raw=nan_2(),
        pval_adjusted=nan_2(),
        null_mu=nan_2(), null_sigma=nan_2(),
        score_pval=nan_2(), score_phat=nan_2(), score_zscore=nan_2(),
        score=nan_2(), signed_score=nan_2(),
        # ── DirC at best lag — U→V is always populated (DENSE per-pair) ──
        dirc_rbar_UV=np.full((P, n_j), np.nan),
        dirc_tau_UV_folds=np.full((kfolds, P), np.nan),
        dirc_tau_UV=nan_2(),
        surr_dirc_tau_UV=np.full((n_surrogates, P), np.nan),
        surr_dirc_rbar_UV=np.full((n_surrogates, P, n_j), np.nan),
        dirc_pval_UV=nan_2(),
        dirc_ranksum_pval_UV=nan_2(),
        # ── DirC V→U — only allocated when dirc_VU=True ──
        dirc_rbar_VU=(np.full((P, n_j), np.nan) if dirc_VU else None),
        dirc_tau_VU_folds=(np.full((kfolds, P), np.nan) if dirc_VU else None),
        dirc_tau_VU=(nan_2() if dirc_VU else None),
        surr_dirc_tau_VU=(np.full((n_surrogates, P), np.nan) if dirc_VU else None),
        surr_dirc_rbar_VU=(np.full((n_surrogates, P, n_j), np.nan) if dirc_VU else None),
        dirc_pval_VU=(nan_2() if dirc_VU else None),
        dirc_ranksum_pval_VU=(nan_2() if dirc_VU else None),
        # ── DirC linear-slope (mean β over folds) + surrogate null ──
        dirc_beta_UV_folds=np.full((kfolds, P), np.nan),
        dirc_beta_UV=nan_2(),
        surr_dirc_beta_UV=np.full((n_surrogates, P), np.nan),
        dirc_pval_beta_UV=nan_2(),
        dirc_beta_VU_folds=(np.full((kfolds, P), np.nan) if dirc_VU else None),
        dirc_beta_VU=(nan_2() if dirc_VU else None),
        surr_dirc_beta_VU=(np.full((n_surrogates, P), np.nan) if dirc_VU else None),
        dirc_pval_beta_VU=(nan_2() if dirc_VU else None),
        # ── Labels / detection / misc ──
        dirc_label=np.full((N, N), "", dtype=object),
        detected=np.zeros((N, N), dtype=np.int8),
        detected_reason=np.full((N, N), "", dtype=object),
        signs=nan_2(),
        lags=lags.copy(),
        theiler_w=np.full((N, N), 0, dtype=int),
        j_range=np.arange(j_min, j_max + 1),
        e=e, emb_tau=int(emb_tau),
        kfolds=kfolds, n_surrogates=n_surrogates,
        dirc_VU=bool(dirc_VU),
        dirc_slope_test=str(dirc_slope_test),
    )

    _slope_test_norm = str(dirc_slope_test).lower()
    if _slope_test_norm not in ("kendall", "linear"):
        raise ValueError(
            f"dirc_slope_test must be 'kendall' or 'linear' "
            f"(got {dirc_slope_test!r})"
        )

    # ── Dispatch pair loop (serial or parallel) ──
    _pair_args = [
        (pi, x_use, int(pairs[pi, 0]), int(pairs[pi, 1]),
         lags, e, int(emb_tau), kfolds, n_surrogates, max_ref,
         nn, j_min, j_max, dirc_alpha, seed + pi * 1000,
         fold_slices, out, verbose if n_jobs == 1 else False, score_type,
         bool(dirc_VU), _slope_test_norm)
        for pi in range(n_pairs)
    ]

    if n_jobs == 1:
        for pi, args in enumerate(_pair_args):
            src, tgt = int(pairs[pi, 0]), int(pairs[pi, 1])
            if verbose:
                print(f"  Pair {pi + 1}/{n_pairs}: node {src} -> node {tgt}")
            _run_pair(*args)
    else:
        # Joblib threading backend: numpy (cdist, argpartition, boolean sums)
        # releases the GIL ⟹ true parallel execution.  Pairs write to
        # disjoint (src,tgt) indices in *out*, so no race conditions.
        from joblib import Parallel, delayed

        def _run_pair_verbose(pi, args):
            src, tgt = int(pairs[pi, 0]), int(pairs[pi, 1])
            if verbose:
                print(f"  Pair {pi + 1}/{n_pairs}: node {src} -> {tgt}  [thread]")
            _run_pair(*args)

        Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_run_pair_verbose)(pi, args) for pi, args in enumerate(_pair_args)
        )

    return out


def _run_pair(
    pi, x_use, src, tgt, lags, e, emb_tau, kfolds, n_surrogates, max_ref,
    nn, j_min, j_max, dirc_alpha, seed,
    fold_slices, out, verbose, score_type="pvalue",
    dirc_VU: bool = False,
    dirc_slope_test: str = "kendall",
):
    """Compute DetC + DirC for a single directed pair, filling *out* in-place."""
    n_trials, T_trial, N = x_use.shape
    n_lags = lags.size
    n_j = j_max - j_min + 1
    rng = np.random.default_rng(seed)

    # ── Theiler window (from full concatenated series) ──
    u_full = x_use[:, :, src].ravel()
    v_full = x_use[:, :, tgt].ravel()
    w = theiler_window(u_full, v_full, e, trial_len=T_trial)
    out["theiler_w"][src, tgt] = w
    if verbose:
        print(f"    Theiler window w = {w}")

    # ============================================================
    # A.  DetC sweep over lags, per fold
    # ============================================================
    phat_j_folds = np.full((kfolds, n_lags, n_j), np.nan)
    phat_mean_folds = np.full((kfolds, n_lags), np.nan)

    for k, test_idx in enumerate(fold_slices):
        u_test = x_use[test_idx, :, src].reshape(-1)
        v_test = x_use[test_idx, :, tgt].reshape(-1)

        for li, tau in enumerate(lags):
            u_a, v_a = align_lag(u_test, v_test, tau)
            u_emb = make_delay_vectors(u_a, e, emb_tau)
            v_emb = make_delay_vectors(v_a, e, emb_tau)
            if u_emb.shape[0] < nn + w:
                continue
            pj, pm = compute_detc(u_emb, v_emb, w, nn, j_min, j_max, max_ref)
            phat_j_folds[k, li] = pj
            phat_mean_folds[k, li] = pm

    # fold-average
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        phat_j_avg = np.nanmean(phat_j_folds, axis=0)      # (n_lags, n_j)
        phat_mean_avg = np.nanmean(phat_mean_folds, axis=0)  # (n_lags,)

    out["phat_j_folds"][:, pi, :, :] = phat_j_folds
    out["phat_mean_folds"][:, pi, :] = phat_mean_folds
    out["phat_j_avg"][pi] = phat_j_avg
    out["phat_mean_avg"][pi] = phat_mean_avg

    # ============================================================
    # B.  Surrogate DetC at lag 0, per fold
    # ============================================================
    surr_phat_j_all = np.full((n_surrogates, kfolds, n_j), np.nan)
    surr_phat_mean_all = np.full((n_surrogates, kfolds), np.nan)

    for k, test_idx in enumerate(fold_slices):
        u_test = x_use[test_idx, :, src].reshape(-1)
        v_test_trials = x_use[test_idx, :, tgt]        # (n_test, T_trial)

        u_emb_lag0 = make_delay_vectors(u_test, e, emb_tau)

        surrs = circular_shift_surrogates(v_test_trials, n_surrogates, w, rng)
        for s, v_surr_trials in enumerate(surrs):
            v_surr = v_surr_trials.reshape(-1)
            v_surr_emb = make_delay_vectors(v_surr, e, emb_tau)
            if v_surr_emb.shape[0] < nn + w:
                continue
            # Align lengths (lag-0: both same length, but embedding trims equally)
            L = min(u_emb_lag0.shape[0], v_surr_emb.shape[0])
            pj, pm = compute_detc(
                u_emb_lag0[:L], v_surr_emb[:L], w, nn, j_min, j_max, max_ref,
            )
            surr_phat_j_all[s, k] = pj
            surr_phat_mean_all[s, k] = pm

    # fold-average surrogates → (n_surrogates,) null distribution
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        surr_phat_mean_avg = np.nanmean(surr_phat_mean_all, axis=1)  # (n_surr,)
        surr_phat_j_avg = np.nanmean(surr_phat_j_all, axis=1)        # (n_surr, n_j)

    out["surr_phat_mean"][:, pi] = surr_phat_mean_avg
    out["surr_phat_j"][:, pi, :] = surr_phat_j_avg

    # ============================================================
    # C.  Scoring: best lag, then *per-fold* Gaussian null; average the
    #     per-fold p-values (rather than testing significance of the
    #     fold-averaged p-hat).  best_lag itself is still selected on the
    #     fold-averaged DetC curve so that DirC below uses a single lag.
    # ============================================================
    valid_lags = np.isfinite(phat_mean_avg)
    if not np.any(valid_lags):
        return
    phat_for_min = np.where(valid_lags, phat_mean_avg, np.inf)
    best_li   = int(np.argmin(phat_for_min))
    best_lag  = int(lags[best_li])
    best_phat = float(phat_mean_avg[best_li])

    # Per-fold p-values at best_lag:  observed phat_k vs that fold's null cloud.
    _pvals_folds = np.full(kfolds, np.nan)
    _mu_folds    = np.full(kfolds, np.nan)
    _sig_folds   = np.full(kfolds, np.nan)
    for _k in range(kfolds):
        _ph_k  = float(phat_mean_folds[_k, best_li])
        _null_k = surr_phat_mean_all[:, _k]           # (n_surrogates,)
        _p, _mu, _sig = gaussian_pvalue_lower(_ph_k, _null_k)
        _pvals_folds[_k] = _p
        _mu_folds[_k]    = _mu
        _sig_folds[_k]   = _sig
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pval_raw = float(np.nanmean(_pvals_folds))
        mu_null  = float(np.nanmean(_mu_folds))
        sig_null = float(np.nanmean(_sig_folds))
    pval_raw = float(np.clip(pval_raw, np.finfo(np.float64).tiny, 1.0))
    pval_adj = float(sidak_correct(pval_raw, n_lags))

    # ── Three score flavours ──
    # 'pvalue' : −log10(p_adj)            may plateau at ≈308 for very strong coupling
    # 'phat'   : 1 − p̂_best               0 under H0; > 0 signals coupling; ≤ 1
    # 'zscore' : (μ0 − p̂_best) / σ0       Gaussian z-score, does not saturate
    s_pval = float(-np.log10(max(pval_adj, np.finfo(np.float64).tiny)))
    s_phat = float(1.0 - best_phat)
    if sig_null > 0.0 and np.isfinite(sig_null):
        s_zscore = float((mu_null - best_phat) / sig_null)
    else:
        s_zscore = 0.0

    _score_map = {"pvalue": s_pval, "phat": s_phat, "zscore": s_zscore}
    score = _score_map.get(score_type, s_pval)

    out["best_lag_idx"][src, tgt]  = best_li
    out["best_lag"][src, tgt]      = best_lag
    out["best_phat"][src, tgt]     = best_phat
    out["pval_raw"][src, tgt]      = pval_raw
    out["pval_adjusted"][src, tgt] = pval_adj
    out["null_mu"][src, tgt]       = mu_null
    out["null_sigma"][src, tgt]    = sig_null
    out["score_pval"][src, tgt]    = s_pval
    out["score_phat"][src, tgt]    = s_phat
    out["score_zscore"][src, tgt]  = s_zscore
    out["score"][src, tgt]         = score
    if verbose:
        print(f"    DetC  best_lag={best_lag:+d}  p-hat={best_phat:.4f}  "
              f"pval_adj={pval_adj:.2g}  "
              f"z={s_zscore:.2f}  1-p̂={s_phat:.4f}")

    # ============================================================
    # D.  DirC at best lag, per fold (both directions)
    # ============================================================
    dirc_rbar_UV_folds = np.full((kfolds, n_j), np.nan)
    dirc_rbar_VU_folds = np.full((kfolds, n_j), np.nan)
    dirc_tau_UV_arr = np.full(kfolds, np.nan)
    dirc_tau_VU_arr = np.full(kfolds, np.nan)
    dirc_m_UV_arr = np.full(kfolds, np.nan)
    dirc_m_VU_arr = np.full(kfolds, np.nan)
    _j_arr_local = np.arange(j_min, j_max + 1, dtype=np.float64)

    for k, test_idx in enumerate(fold_slices):
        u_test = x_use[test_idx, :, src].reshape(-1)
        v_test = x_use[test_idx, :, tgt].reshape(-1)
        u_a, v_a = align_lag(u_test, v_test, best_lag)
        u_emb = make_delay_vectors(u_a, e, emb_tau)
        v_emb = make_delay_vectors(v_a, e, emb_tau)
        if u_emb.shape[0] < nn + w:
            continue
        rbar_uv, tau_uv = compute_dirc(u_emb, v_emb, w, nn, j_min, j_max, max_ref)
        dirc_rbar_UV_folds[k] = rbar_uv
        dirc_tau_UV_arr[k] = tau_uv
        dirc_m_UV_arr[k] = _linreg_slope(_j_arr_local, rbar_uv)
        if dirc_VU:
            rbar_vu, tau_vu = compute_dirc(v_emb, u_emb, w, nn, j_min, j_max, max_ref)
            dirc_rbar_VU_folds[k] = rbar_vu
            dirc_tau_VU_arr[k] = tau_vu
            dirc_m_VU_arr[k] = _linreg_slope(_j_arr_local, rbar_vu)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        tau_UV_mean = float(np.nanmean(dirc_tau_UV_arr))
        tau_VU_mean = float(np.nanmean(dirc_tau_VU_arr))
        m_UV_mean   = float(np.nanmean(dirc_m_UV_arr))
        m_VU_mean   = float(np.nanmean(dirc_m_VU_arr))
        rbar_UV_avg = np.nanmean(dirc_rbar_UV_folds, axis=0)
        rbar_VU_avg = np.nanmean(dirc_rbar_VU_folds, axis=0)

    out["dirc_tau_UV_folds"][:, pi] = dirc_tau_UV_arr
    out["dirc_tau_UV"][src, tgt] = tau_UV_mean
    out["dirc_beta_UV_folds"][:, pi] = dirc_m_UV_arr
    out["dirc_beta_UV"][src, tgt] = m_UV_mean
    out["dirc_rbar_UV"][pi] = rbar_UV_avg
    if dirc_VU:
        out["dirc_tau_VU_folds"][:, pi] = dirc_tau_VU_arr
        out["dirc_tau_VU"][src, tgt] = tau_VU_mean
        out["dirc_beta_VU_folds"][:, pi] = dirc_m_VU_arr
        out["dirc_beta_VU"][src, tgt] = m_VU_mean
        out["dirc_rbar_VU"][pi] = rbar_VU_avg

    # ============================================================
    # E.  Surrogate DirC at best_lag (the lag that minimises DetC p-hat)
    # ============================================================
    surr_tau_UV = np.full(n_surrogates, np.nan)
    surr_tau_VU = np.full(n_surrogates, np.nan)
    surr_rbar_UV = np.full((n_surrogates, n_j), np.nan)
    surr_rbar_VU = np.full((n_surrogates, n_j), np.nan)

    # Compute over folds and average
    _surr_tau_UV_folds = np.full((n_surrogates, kfolds), np.nan)
    _surr_tau_VU_folds = np.full((n_surrogates, kfolds), np.nan)
    _surr_rbar_UV_folds = np.full((n_surrogates, kfolds, n_j), np.nan)
    _surr_rbar_VU_folds = np.full((n_surrogates, kfolds, n_j), np.nan)
    _surr_m_UV_folds = np.full((n_surrogates, kfolds), np.nan)
    _surr_m_VU_folds = np.full((n_surrogates, kfolds), np.nan)

    for k, test_idx in enumerate(fold_slices):
        u_test = x_use[test_idx, :, src].reshape(-1)
        v_test_trials = x_use[test_idx, :, tgt]
        T_trial = v_test_trials.shape[1]

        # Generate surrogates on the *unaligned* V trials, then apply the
        # best_lag alignment to (U, V_surr) before embedding — mirrors the
        # observed-DirC pipeline (section D) so the null is evaluated at
        # the same lag as the statistic.
        surrs = circular_shift_surrogates(v_test_trials, n_surrogates, w, rng)

        for s, v_surr_trials in enumerate(surrs):
            v_surr = v_surr_trials.reshape(-1)
            # align at best_lag
            u_a, v_a = align_lag(u_test, v_surr, best_lag)
            u_emb = make_delay_vectors(u_a, e, emb_tau)
            v_surr_emb = make_delay_vectors(v_a, e, emb_tau)
            L = min(u_emb.shape[0], v_surr_emb.shape[0])
            if L < nn + w:
                continue
            rb_uv, tk_uv = compute_dirc(
                u_emb[:L], v_surr_emb[:L], w, nn, j_min, j_max, max_ref)
            _surr_tau_UV_folds[s, k] = tk_uv
            _surr_rbar_UV_folds[s, k] = rb_uv
            _surr_m_UV_folds[s, k] = _linreg_slope(_j_arr_local, rb_uv)
            if dirc_VU:
                rb_vu, tk_vu = compute_dirc(
                    v_surr_emb[:L], u_emb[:L], w, nn, j_min, j_max, max_ref)
                _surr_tau_VU_folds[s, k] = tk_vu
                _surr_rbar_VU_folds[s, k] = rb_vu
                _surr_m_VU_folds[s, k] = _linreg_slope(_j_arr_local, rb_vu)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        surr_tau_UV = np.nanmean(_surr_tau_UV_folds, axis=1)
        surr_tau_VU = np.nanmean(_surr_tau_VU_folds, axis=1)
        surr_rbar_UV = np.nanmean(_surr_rbar_UV_folds, axis=1)
        surr_rbar_VU = np.nanmean(_surr_rbar_VU_folds, axis=1)
        surr_m_UV = np.nanmean(_surr_m_UV_folds, axis=1)   # (n_surr,)
        surr_m_VU = np.nanmean(_surr_m_VU_folds, axis=1)

    out["surr_dirc_tau_UV"][:, pi] = surr_tau_UV
    out["surr_dirc_rbar_UV"][:, pi, :] = surr_rbar_UV
    out["surr_dirc_beta_UV"][:, pi] = surr_m_UV
    if dirc_VU:
        out["surr_dirc_tau_VU"][:, pi] = surr_tau_VU
        out["surr_dirc_rbar_VU"][:, pi, :] = surr_rbar_VU
        out["surr_dirc_beta_VU"][:, pi] = surr_m_VU

    # ── Mean-β (linear slope) surrogate p-value: upper-tail on β̄_obs ──
    pval_m_uv, _, _ = gaussian_pvalue_upper(m_UV_mean, surr_m_UV)
    out["dirc_pval_beta_UV"][src, tgt] = float(pval_m_uv)
    if dirc_VU:
        pval_m_vu, _, _ = gaussian_pvalue_upper(m_VU_mean, surr_m_VU)
        out["dirc_pval_beta_VU"][src, tgt] = float(pval_m_vu)
    else:
        pval_m_vu = float("nan")

    # ── DirC slope significance (one-sided upper tail on Kendall τ) ──
    # Per-fold p-values then average (matches new DetC logic).
    _pval_uv_folds = np.full(kfolds, np.nan)
    _pval_vu_folds = np.full(kfolds, np.nan)
    for _k in range(kfolds):
        _p_uv, _, _ = gaussian_pvalue_upper(dirc_tau_UV_arr[_k],
                                             _surr_tau_UV_folds[:, _k])
        _pval_uv_folds[_k] = _p_uv
        if dirc_VU:
            _p_vu, _, _ = gaussian_pvalue_upper(dirc_tau_VU_arr[_k],
                                                 _surr_tau_VU_folds[:, _k])
            _pval_vu_folds[_k] = _p_vu
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pval_uv = float(np.nanmean(_pval_uv_folds))
        pval_vu = float(np.nanmean(_pval_vu_folds)) if dirc_VU else float("nan")
    pval_uv = float(np.clip(pval_uv, np.finfo(np.float64).tiny, 1.0))
    if dirc_VU:
        pval_vu = float(np.clip(pval_vu, np.finfo(np.float64).tiny, 1.0))

    # ── Ranksum gate (pool folds; pool surrogates × folds; one-sided 'less') ──
    # Test whether the observed rbar_j values, pooled across folds, are
    # stochastically *below* the surrogate cloud, also pooled across folds.
    # If not, the observed line is indistinguishable from the null —
    # coupling is rejected for this direction regardless of the slope.
    _obs_uv_pool  = dirc_rbar_UV_folds.ravel()
    _surr_uv_pool = _surr_rbar_UV_folds.ravel()
    rs_pval_uv = _ranksum_below(_obs_uv_pool, _surr_uv_pool)
    if dirc_VU:
        _obs_vu_pool  = dirc_rbar_VU_folds.ravel()
        _surr_vu_pool = _surr_rbar_VU_folds.ravel()
        rs_pval_vu = _ranksum_below(_obs_vu_pool, _surr_vu_pool)
    else:
        rs_pval_vu = float("nan")

    # Pick which slope test drives the gate / label.
    # Both p_τ and p_β are always computed and reported; this switch only
    # selects which one participates in the detection decision.
    if str(dirc_slope_test).lower() == "linear":
        gate_slope_uv      = m_UV_mean
        gate_pval_slope_uv = pval_m_uv
        gate_slope_vu      = m_VU_mean
        gate_pval_slope_vu = pval_m_vu
        slope_name = "beta"
    else:  # default: "kendall"
        gate_slope_uv      = tau_UV_mean
        gate_pval_slope_uv = pval_uv
        gate_slope_vu      = tau_VU_mean
        gate_pval_slope_vu = pval_vu
        slope_name = "tau"

    # Effective per-direction p-value = worst of (ranksum, slope).
    # If the ranksum fails to reject H0, slope significance is overruled.
    eff_pval_uv = max(gate_pval_slope_uv, rs_pval_uv)
    if dirc_VU:
        eff_pval_vu = max(gate_pval_slope_vu, rs_pval_vu)
        label = classify_coupling(eff_pval_uv, eff_pval_vu, dirc_alpha)
    else:
        # VU direction disabled: label from UV alone.
        label = "U->V" if eff_pval_uv < dirc_alpha else "latent"

    out["dirc_pval_UV"][src, tgt] = pval_uv
    out["dirc_ranksum_pval_UV"][src, tgt] = rs_pval_uv
    if dirc_VU:
        out["dirc_pval_VU"][src, tgt] = pval_vu
        out["dirc_ranksum_pval_VU"][src, tgt] = rs_pval_vu
    out["dirc_label"][src, tgt] = label

    # ── Strict detection rule for A -> B (U=src, V=tgt) ──
    detected_uv, reason_uv = detect_coupling_strict(
        detc_pval_adj=float(pval_adj),
        best_lag=int(best_lag),
        dirc_slope=float(gate_slope_uv),
        dirc_pval_slope=float(gate_pval_slope_uv),
        dirc_pval_rs=float(rs_pval_uv),
        alpha=float(dirc_alpha),
        slope_name=slope_name,
    )
    out["detected"][src, tgt] = int(detected_uv)
    out["detected_reason"][src, tgt] = reason_uv

    if verbose:
        if dirc_VU:
            print(f"    DirC  tau_K_UV={tau_UV_mean:+.3f} "
                  f"(p_tau={pval_uv:.2g}, p_rs={rs_pval_uv:.2g}, "
                  f"beta={m_UV_mean:+.3g}, p_beta={pval_m_uv:.2g})  "
                  f"tau_K_VU={tau_VU_mean:+.3f} "
                  f"(p_tau={pval_vu:.2g}, p_rs={rs_pval_vu:.2g}, "
                  f"beta={m_VU_mean:+.3g}, p_beta={pval_m_vu:.2g})  -> {label}")
        else:
            print(f"    DirC  tau_K_UV={tau_UV_mean:+.3f} "
                  f"(p_tau={pval_uv:.2g}, p_rs={rs_pval_uv:.2g}, "
                  f"beta={m_UV_mean:+.3g}, p_beta={pval_m_uv:.2g})  -> {label}")
        _verdict = "DETECTED" if detected_uv else f"REJECTED ({reason_uv})"
        print(f"    STRICT {src}->{tgt}: {_verdict}")

    # ── Sign from cross-correlogram at best lag ──
    sgn = sign_at_lag(x_use, src, tgt, best_lag)
    out["signs"][src, tgt] = sgn
    out["signed_score"][src, tgt] = score * sgn


# ============================================================================
# 11. Diagnostic plotting — 4-column grid
# ============================================================================
def plot_detc_dirc_grid(
    result: dict,
    x_use: np.ndarray,
    diagnostics: list[dict],
    title: str,
    out_dir,
    fname: str,
    dt: float = 1.0,
    dirc_VU: bool = False,
) -> Dict[str, dict]:
    """Plot 4-column diagnostic grid, one row per pair.

    Each element of *diagnostics* is a dict with keys
    ``src_idx, tgt_idx, row_label, key``.

    Columns:
      1. DetC (mean p-hat) vs lag  — per-fold + mean + surrogate 95 % CI
      2. Per-j profile at best lag — folds + mean + surrogate 95 % CI + p = 1 ref
      3. Cross-correlogram
      4. DirC at best lag — both directions + surrogate 95 % CI + classification

    Returns dict  key → {best_lag, best_phat, pval, label}.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    set_plot_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_rows = len(diagnostics)
    sw = S_UNIT
    sh = 0.85 * S_UNIT
    fig, axes = plt.subplots(
        n_rows, 4, figsize=(4 * sw, n_rows * sh), constrained_layout=True,
    )
    if n_rows == 1:
        axes = np.asarray([axes])

    lags = result["lags"]
    # Plot-axis lag values (seconds if dt given, else raw sample steps).
    dt = float(dt)
    lags_plot = lags * dt
    lag_unit_label = "Lag (s)" if dt != 1.0 else "Lag"
    j_range = result["j_range"]
    kfolds = result["kfolds"]
    n_surr = result["n_surrogates"]
    # Dense-pair layout lookup (new; see run_detc_dirc allocation block).
    pair_idx = result.get("pair_idx")
    info: Dict[str, dict] = {}

    for row, spec in enumerate(diagnostics):
        si, ti = spec["src_idx"], spec["tgt_idx"]
        ax = axes[row]
        if pair_idx is None:
            raise KeyError(
                "plot_detc_dirc_grid requires result['pair_idx'] (dense-pair layout)"
            )
        pi = int(pair_idx[si, ti])
        if pi < 0:
            raise ValueError(
                f"diagnostic pair ({si},{ti}) was not in selected_pairs; "
                "use diagnostic_pairs_as_selected() to include it before "
                "calling run_detc_dirc."
            )

        # ── gather data ──
        pm_folds = result["phat_mean_folds"][:, pi, :]          # (K, n_lags)
        pm_avg = result["phat_mean_avg"][pi, :]                  # (n_lags,)
        pj_folds = result["phat_j_folds"][:, pi, :, :]          # (K, n_lags, n_j)
        pj_avg = result["phat_j_avg"][pi, :, :]                 # (n_lags, n_j)
        surr_pm = result["surr_phat_mean"][:, pi]                # (S,)
        surr_pj = result["surr_phat_j"][:, pi, :]               # (S, n_j)
        best_li = int(result["best_lag_idx"][si, ti])
        best_lag_val = int(result["best_lag"][si, ti])
        best_phat = float(result["best_phat"][si, ti])
        pval_adj = float(result["pval_adjusted"][si, ti])
        rbar_UV = result["dirc_rbar_UV"][pi]                     # (n_j,)
        s_rbar_UV = result["surr_dirc_rbar_UV"][:, pi, :]       # (S, n_j)
        label = str(result["dirc_label"][si, ti])
        pval_uv = float(result["dirc_pval_UV"][si, ti])
        _rs_uv_arr = result.get("dirc_ranksum_pval_UV")
        rs_pval_uv = float(_rs_uv_arr[si, ti]) if _rs_uv_arr is not None else float("nan")
        tau_UV_val = float(result["dirc_tau_UV"][si, ti])
        # Linear-slope m and surrogate p_m (may be absent on legacy results)
        _m_UV_arr = result.get("dirc_beta_UV")
        _pm_UV_arr = result.get("dirc_pval_beta_UV")
        m_UV_val = float(_m_UV_arr[si, ti]) if _m_UV_arr is not None else float("nan")
        pm_uv = float(_pm_UV_arr[si, ti]) if _pm_UV_arr is not None else float("nan")
        # VU branch (only populated when dirc_VU=True at run_detc_dirc time)
        if dirc_VU:
            _rbar_VU_arr = result.get("dirc_rbar_VU")
            _surr_rbar_VU_arr = result.get("surr_dirc_rbar_VU")
            _tau_VU_arr = result.get("dirc_tau_VU")
            _pval_VU_arr = result.get("dirc_pval_VU")
            _rs_VU_arr = result.get("dirc_ranksum_pval_VU")
            if (_rbar_VU_arr is None or _surr_rbar_VU_arr is None
                    or _tau_VU_arr is None or _pval_VU_arr is None):
                raise ValueError(
                    "plot_detc_dirc_grid called with dirc_VU=True but the "
                    "result has no VU arrays (was run_detc_dirc called with "
                    "dirc_VU=False?)."
                )
            rbar_VU = _rbar_VU_arr[pi]
            s_rbar_VU = _surr_rbar_VU_arr[:, pi, :]
            tau_VU_val = float(_tau_VU_arr[si, ti])
            pval_vu = float(_pval_VU_arr[si, ti])
            rs_pval_vu = float(_rs_VU_arr[si, ti]) if _rs_VU_arr is not None else float("nan")
            _m_VU_arr = result.get("dirc_beta_VU")
            _pm_VU_arr = result.get("dirc_pval_beta_VU")
            m_VU_val = float(_m_VU_arr[si, ti]) if _m_VU_arr is not None else float("nan")
            pm_vu = float(_pm_VU_arr[si, ti]) if _pm_VU_arr is not None else float("nan")
        else:
            rbar_VU = None; s_rbar_VU = None
            tau_VU_val = float("nan"); pval_vu = float("nan"); rs_pval_vu = float("nan")
            m_VU_val = float("nan"); pm_vu = float("nan")

        # ────────────── Col 0: DetC vs lag ──────────────
        a = ax[0]
        for k in range(kfolds):
            a.plot(lags_plot, pm_folds[k], color=_C_UV, alpha=0.2, lw=_LW_FOLD)
        a.plot(lags_plot, pm_avg, color=_C_UV, lw=_LW, label="mean $\\hat{p}$")
        # surrogate 95 % CI (horizontal band)
        fin = surr_pm[np.isfinite(surr_pm)]
        if fin.size > 1:
            lo, hi = np.percentile(fin, [2.5, 97.5])
            a.axhspan(lo, hi, color=_C_SURR, alpha=0.18, label="surr 95 % CI")
            a.axhline(np.median(fin), color=_C_SURR, ls="--", lw=_LW_FOLD, alpha=0.6)
        best_lag_plot = best_lag_val * dt
        _lag_lbl = (f"min lag={best_lag_plot:.2g} s" if dt != 1.0
                    else f"min lag={best_lag_val}")
        a.plot(best_lag_plot, best_phat, "o", color=_C_PEAK, ms=_MS, zorder=5,
               label=_lag_lbl)
        a.axhline(1.0, color="k", lw=_LW_FOLD, ls=":", alpha=0.5)
        a.set_xlabel(lag_unit_label)
        a.set_ylabel("mean $\\hat{p}$")
        if row == 0:
            a.set_title("DetC vs lag", fontsize=_FS)
        a.legend(**_LEGEND_KW, loc="best")
        # row label
        a.text(-0.45, 0.5, spec["row_label"], transform=a.transAxes,
               rotation=90, va="center", ha="center", fontweight="bold", fontsize=_FS)

        # ────────────── Col 1: per-j at best lag ──────────────
        a = ax[1]
        for k in range(kfolds):
            a.plot(j_range, pj_folds[k, best_li], color=_C_UV, alpha=0.2, lw=_LW_FOLD)
        a.plot(j_range, pj_avg[best_li], color=_C_UV, lw=_LW, label="mean")
        fin_j = surr_pj[np.all(np.isfinite(surr_pj), axis=1)]
        if fin_j.shape[0] > 1:
            lo_j = np.percentile(fin_j, 2.5, axis=0)
            hi_j = np.percentile(fin_j, 97.5, axis=0)
            a.fill_between(j_range, lo_j, hi_j, color=_C_SURR, alpha=0.18, label="surr 95 %")
        a.axhline(1.0, color="k", lw=_LW_FOLD, ls=":", alpha=0.5, label="$p=1$")
        a.set_xlabel("Neighbour rank $j$")
        a.set_ylabel("$\\hat{p}_j$")
        if row == 0:
            a.set_title(f"$\\hat{{p}}_j$ at best lag", fontsize=_FS)
        _tau_txt = (f"$\\tau$={best_lag_val*dt:.2g} s"
                    if dt != 1.0 else f"$\\tau$={best_lag_val}")
        a.text(0.02, 0.02,
               f"{_tau_txt}, $p$={pval_adj:.2g}",
               transform=a.transAxes, fontsize=_FS_AN, va="bottom")
        a.legend(**_LEGEND_KW, loc="best")

        # ────────────── Col 2: Cross-correlogram ──────────────
        a = ax[2]
        xcorr = pair_cross_correlogram(x_use, si, ti, lags)
        a.plot(lags_plot, xcorr, color=_C_UV, lw=_LW, label="trial avg")
        a.axhline(0, color="#999999", lw=_LW_FOLD, ls=":", alpha=0.5)
        a.axvline(best_lag_val * dt, color=_C_PEAK, ls="--",
                  lw=_LW_FOLD * 1.2, alpha=0.7)
        a.set_xlabel(lag_unit_label)
        a.set_ylabel("Pearson $r$")
        if row == 0:
            a.set_title("Cross-correlogram", fontsize=_FS)
        a.legend(**_LEGEND_KW, loc="best")

        # ────────────── Col 3: DirC at best lag ──────────────
        a = ax[3]
        a.plot(j_range, rbar_UV * 100, color=_C_UV, lw=_LW,
               marker="o", ms=_MS - 1, label="DirC(U,V)")
        if dirc_VU:
            a.plot(j_range, rbar_VU * 100, color=_C_VU, lw=_LW,
                   marker="s", ms=_MS - 1, label="DirC(V,U)")
        # surrogate CI
        fin_uv = s_rbar_UV[np.all(np.isfinite(s_rbar_UV), axis=1)]
        if fin_uv.shape[0] > 1:
            a.fill_between(j_range,
                           np.percentile(fin_uv, 2.5, axis=0) * 100,
                           np.percentile(fin_uv, 97.5, axis=0) * 100,
                           color=_C_UV, alpha=0.12)
        if dirc_VU:
            fin_vu = s_rbar_VU[np.all(np.isfinite(s_rbar_VU), axis=1)]
            if fin_vu.shape[0] > 1:
                a.fill_between(j_range,
                               np.percentile(fin_vu, 2.5, axis=0) * 100,
                               np.percentile(fin_vu, 97.5, axis=0) * 100,
                               color=_C_VU, alpha=0.12)
        a.set_xlabel("Neighbour rank $j$")
        a.set_ylabel("rank (%)")
        if row == 0:
            a.set_title("DirC at best lag", fontsize=_FS)
        if dirc_VU:
            _annot = (f"$\\tau_K^{{UV}}$={tau_UV_val:+.2f} "
                      f"$p_\\tau$={pval_uv:.2g} $p_{{rs}}$={rs_pval_uv:.2g} "
                      f"$\\beta^{{UV}}$={m_UV_val:+.3g} "
                      f"$p_\\beta$={pm_uv:.2g}\n"
                      f"$\\tau_K^{{VU}}$={tau_VU_val:+.2f} "
                      f"$p_\\tau$={pval_vu:.2g} $p_{{rs}}$={rs_pval_vu:.2g} "
                      f"$\\beta^{{VU}}$={m_VU_val:+.3g} "
                      f"$p_\\beta$={pm_vu:.2g}\n"
                      f"{label}")
        else:
            _annot = (f"$\\tau_K^{{UV}}$={tau_UV_val:+.2f} "
                      f"$p_\\tau$={pval_uv:.2g} $p_{{rs}}$={rs_pval_uv:.2g} "
                      f"$\\beta$={m_UV_val:+.3g} $p_\\beta$={pm_uv:.2g}\n"
                      f"{label}")
        a.text(0.02, 0.97, _annot,
               transform=a.transAxes, fontsize=_FS_AN, va="top",
               bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1))
        a.legend(**_LEGEND_KW, loc="lower right")

        info[spec["key"]] = dict(
            best_lag=best_lag_val, best_phat=best_phat,
            pval=pval_adj, label=label,
        )

    fig.suptitle(title, fontsize=_FS, fontweight="bold")
    fpath = out_dir / f"{fname}.pdf"
    fig.savefig(fpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fpath}")
    return info
