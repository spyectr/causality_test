"""
New_CI_example.py — DetC / DirC coupling-detection example.

Drop-in replacement for eCCM_CI_example.py using the Sauer-Sugihara (2025)
DetC + DirC methods instead of eCCM.

Three scenarios:
  A. Causal chain:   U -> X (lag 0) -> Y (lag CI_DELAY)
  B. Common input:   U -> X (lag 0),  U -> Y (lag CI_DELAY),  no X->Y
  C. RecXY:          U -> X (lag 0),  X -> Y (lag RECLAG_X2Y),
                     Y -> X (lag RECLAG_Y2X) with same strength as X->Y

For each scenario, runs DetC + DirC for all six directed pairs
(U->X, U->Y, X->U, Y->U, X->Y, Y->X) with dirc_VU=True so both
U->V and V->U DirC statistics are produced, and writes 4-column
diagnostic plots.

Outputs saved to: example/detc_dirc_<rnn_mode>/
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")

# ── Project imports ──
PROJ = Path(__file__).resolve().parent

# Make RNNCausality importable (needed by eCCM_CI_example on import)
RNNC_DIR = PROJ / "RNNCausality"
if str(RNNC_DIR) not in sys.path:
    sys.path.insert(0, str(RNNC_DIR))

from eCCM_CI_example import (                   # noqa: E402
    simulate_ricker,
    simulate_logistic,
    generate_ou,
    generate_ci_source,
    simulate_logistic_common_input,
    simulate_ricker_common_input,
    log_array_stats,
    plot_traces,
    canonical_mode,
)

from sauer_sugihara_aux import (                 # noqa: E402
    run_detc_dirc,
    plot_detc_dirc_grid,
)


# ============================================================================
# Tee stream (duplicate stdout/stderr to a log file)
# ============================================================================
class _TeeStream:
    def __init__(self, stream, log_file):
        self._stream, self._log = stream, log_file
    def write(self, msg):
        self._stream.write(msg); self._log.write(msg)
    def flush(self):
        self._stream.flush(); self._log.flush()


# ============================================================================
# Equation printer (discrete maps)
# ============================================================================
def _print_discrete_equations(mode: str, R_or_W: np.ndarray, L: np.ndarray,
                               names: list[str],
                               r_per_unit: np.ndarray | None = None):
    """Print the update rule of every node with its numerical parameters.

    Ricker:      x_i(t+1) = x_i(t) * exp( R[i,i] - sum_j R[i,j] * x_j(t - L[i,j]) )
                 (off-diagonal entries are the inter-node coupling coefficients)
    Logistic:    x_i(t+1) = r_i * x_i(t) * (1 - x_i(t))
                            - sum_{j!=i} W[i,j] * x_j(t - L[i,j]) * (1 - x_j(...))
                 (form used by simulate_logistic)
    """
    N = R_or_W.shape[0]
    for i in range(N):
        terms = []
        if mode == "ricker":
            self_r = R_or_W[i, i]
            for j in range(N):
                if j == i:
                    continue
                c = R_or_W[i, j]
                if c != 0.0:
                    terms.append(f"{c:+.4g} * {names[j]}(t-{int(L[i,j])})")
            coup = (" - (" + " + ".join(terms) + ")") if terms else ""
            print(f"  {names[i]}(t+1) = {names[i]}(t) * exp( {self_r:.4g}"
                  f" - {names[i]}(t){coup} )")
        elif mode == "logistic":
            r_i = r_per_unit[i] if r_per_unit is not None else float('nan')
            for j in range(N):
                if j == i:
                    continue
                c = R_or_W[i, j]
                if c != 0.0:
                    terms.append(f"{c:+.4g} * {names[j]}(t-{int(L[i,j])})"
                                 f"*(1-{names[j]}(t-{int(L[i,j])}))")
            coup = (" + " + " + ".join(terms)) if terms else ""
            print(f"  {names[i]}(t+1) = {r_i:.4g} * {names[i]}(t) *"
                  f" (1 - {names[i]}(t)){coup}")


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    OUT_BASE = PROJ / "example"

    # ------------------------------------------------------------------
    # Parameters (same as eCCM_CI_example.py unless noted)
    # ------------------------------------------------------------------
    RNN_MODE  = "ricker"
    CI_MODE   = "logistic"
    CI_DELAY  = 6
    RECLAG_X2Y = 1       # X->Y lag in scenario C (RecXY)
    RECLAG_Y2X = 6       # Y->X lag in scenario C (RecXY)

    T          = 2000
    N_TRIALS   = 20
    BURNIN     = 100
    SEED       = 2025
    MAX_LAG    = 10

    # DetC / DirC specific
    EMB_DIM    = 5        # embedding dimension (paper default)
    KFOLDS     = 4
    N_SURR     = 50
    MAX_REF    = 500      # max reference points per DetC / DirC call
    SCORE_TYPE = "phat"   # 'pvalue' | 'phat' | 'zscore'
    N_JOBS     = max(1, (os.cpu_count() or 4) - 2)  # parallel pairs via joblib threading
    # DirC slope test used by the detection gate / DirC label.
    #   'kendall' (default) — Kendall τ + per-fold gaussian-pvalue average
    #   'linear'            — OLS slope β + surrogate null on β̄
    # Both p_τ and p_β are always computed and reported; this only selects
    # which one drives the detection decision.
    DIRC_SLOPE_TEST = "kendall"

    # Ricker parameters
    R_SELF     = 3.7
    R_COUPLE   = 0.3

    # Logistic parameters
    LOGISTIC_R       = 3.9
    LOGISTIC_COUPLE  = 0.3

    PARAM_JITTER_STD = 0.05

    # OU parameters (only when CI_MODE == "ou")
    OU_ALPHA   = 0.01
    OU_RHO     = 0.2
    OU_SIGMA   = 0.2

    RNN_MODE = canonical_mode(RNN_MODE)
    CI_MODE  = canonical_mode(CI_MODE, allow_ou=True)

    OUT = OUT_BASE / f"detc_dirc_{RNN_MODE}"
    OUT.mkdir(parents=True, exist_ok=True)

    _log_fh = open(OUT / "run.log", "w")
    sys.stdout = _TeeStream(sys.__stdout__, _log_fh)
    sys.stderr = _TeeStream(sys.__stderr__, _log_fh)
    _t0_global = time.time()

    # Log parameters
    print("Parameters:")
    print(f"  T={T}, N_TRIALS={N_TRIALS}, BURNIN={BURNIN}, SEED={SEED}")
    print(f"  MAX_LAG={MAX_LAG}, EMB_DIM={EMB_DIM}, KFOLDS={KFOLDS}")
    print(f"  N_SURR={N_SURR}, MAX_REF={MAX_REF}")
    print(f"  RNN_MODE={RNN_MODE}, CI_MODE={CI_MODE}, CI_DELAY={CI_DELAY}")
    print(f"  R_SELF={R_SELF}, R_COUPLE={R_COUPLE}")
    print(f"  LOGISTIC_R={LOGISTIC_R}, LOGISTIC_COUPLE={LOGISTIC_COUPLE}")
    print(f"  PARAM_JITTER_STD={PARAM_JITTER_STD}")
    if CI_MODE == "ou":
        print(f"  OU_ALPHA={OU_ALPHA}, OU_RHO={OU_RHO}, OU_SIGMA={OU_SIGMA}")
    print(f"  Total series length per trial = {T - BURNIN}")
    print()

    lags = np.arange(-MAX_LAG, MAX_LAG + 1, dtype=np.int32)
    N_nodes = 3  # U=0, X=1, Y=2

    # Per-unit jitter
    _jrng = np.random.default_rng(SEED + 7777)
    r_self_3 = R_SELF + PARAM_JITTER_STD * _jrng.standard_normal(N_nodes)
    logistic_r_3 = np.clip(
        LOGISTIC_R + PARAM_JITTER_STD * _jrng.standard_normal(N_nodes),
        3.57, 4.0,
    )
    r_self_2 = r_self_3[1:]
    logistic_r_2 = logistic_r_3[1:]
    print(f"  Per-unit R_SELF (U,X,Y):     {r_self_3}")
    print(f"  Per-unit LOGISTIC_R (U,X,Y): {logistic_r_3}")
    print()

    pairs = np.array(
        [[0, 1], [0, 2], [1, 0], [2, 0], [1, 2], [2, 1]], dtype=np.int32
    )
    names = ["U", "X", "Y"]
    spec_list = [
        dict(key="ux", src_idx=0, tgt_idx=1, row_label="U->X"),
        dict(key="uy", src_idx=0, tgt_idx=2, row_label="U->Y"),
        dict(key="xu", src_idx=1, tgt_idx=0, row_label="X->U"),
        dict(key="yu", src_idx=2, tgt_idx=0, row_label="Y->U"),
        dict(key="xy", src_idx=1, tgt_idx=2, row_label="X->Y"),
        dict(key="yx", src_idx=2, tgt_idx=1, row_label="Y->X"),
    ]

    # ==================================================================
    # Scenario A: Causal chain  U -> X (lag 0) -> Y (lag L)
    # ==================================================================
    print("=" * 60)
    print(f"SCENARIO A: Causal chain ({RNN_MODE})  "
          f"U -> X (lag 0) -> Y (lag {CI_DELAY})")
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
        print(f"\n  R =\n{R_causal}")
        print(f"  L =\n{L_causal}")
        print("\n  --- Scenario A update equations (Ricker) ---")
        _print_discrete_equations("ricker", R_causal, L_causal, names)
        x_causal = simulate_ricker(R_causal, L_causal, T, N_TRIALS, SEED)
    else:
        print(f"\n  W =\n{C_causal}")
        print(f"  L =\n{L_causal}")
        print("\n  --- Scenario A update equations (Logistic) ---")
        _print_discrete_equations("logistic", C_causal, L_causal, names,
                                  r_per_unit=logistic_r_3)
        x_causal = simulate_logistic(C_causal, L_causal, T, N_TRIALS, SEED,
                                     r=logistic_r_3)
    x_use_c = x_causal[:, BURNIN:, :]
    log_array_stats("causal x_use", x_use_c, names)
    plot_traces(x_use_c[0], names,
                f"Causal chain ({RNN_MODE}): U->X(lag0)->Y(lag{CI_DELAY})",
                OUT, "traces_causal")

    # ── Run DetC + DirC ──
    _t0 = time.time()
    print("\n  Running DetC + DirC (causal chain)...")
    res_c = run_detc_dirc(
        x_use_c, lags, pairs,
        e=EMB_DIM, kfolds=KFOLDS, n_surrogates=N_SURR,
        max_ref=MAX_REF, seed=SEED,
        score_type=SCORE_TYPE, n_jobs=N_JOBS,
        dirc_VU=True,
    )
    print(f"  Finished in {time.time() - _t0:.1f}s\n")

    # ── Plot diagnostics ──
    info_c = plot_detc_dirc_grid(
        res_c, x_use_c, spec_list,
        title=f"Causal chain ({RNN_MODE}) — DetC / DirC diagnostics",
        out_dir=OUT, fname="causal_diagnostics",
        dirc_VU=True,
    )

    for k in ("ux", "uy", "xu", "yu", "xy", "yx"):
        d = info_c[k]
        print(f"  {k.upper():>4s}: best_lag={d['best_lag']:+d}  "
              f"p-hat={d['best_phat']:.4f}  p={d['pval']:.2g}  {d['label']}")

    # ==================================================================
    # Scenario B: Common input  U -> X (lag 0), U -> Y (lag L), no X->Y
    # ==================================================================
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
            print(f"\n  R =\n{R_ci}")
            print(f"  L =\n{L_ci}")
            print("\n  --- Scenario B update equations (Ricker) ---")
            _print_discrete_equations("ricker", R_ci, L_ci, names)
            x_ci = simulate_ricker(R_ci, L_ci, T, N_TRIALS, SEED + 1)
        else:
            print(f"\n  W =\n{C_ci}")
            print(f"  L =\n{L_ci}")
            print("\n  --- Scenario B update equations (Logistic) ---")
            _print_discrete_equations("logistic", C_ci, L_ci, names,
                                      r_per_unit=logistic_r_3)
            x_ci = simulate_logistic(C_ci, L_ci, T, N_TRIALS, SEED + 1,
                                     r=logistic_r_3)
        x_use_ci = x_ci[:, BURNIN:, :]
        trace_names = names

    elif CI_MODE == "ou" and RNN_MODE == "ricker":
        N_ci = 2
        R_ci2 = np.zeros((N_ci, N_ci), dtype=np.float64)
        L_ci2 = np.zeros((N_ci, N_ci), dtype=np.int32)
        for i in range(N_ci):
            R_ci2[i, i] = r_self_2[i]
        c_ou = generate_ou(N_TRIALS, T, rho=OU_RHO, sigma_eps=OU_SIGMA,
                           seed=SEED + 100)
        from eCCM_CI_example import simulate_ricker_driven
        u_ou = np.zeros((N_TRIALS, T, N_ci), dtype=np.float64)
        u_ou[:, :, 0] = OU_ALPHA * c_ou
        u_ou[:, CI_DELAY:, 1] = OU_ALPHA * c_ou[:, :T - CI_DELAY]
        x_ci_raw = simulate_ricker_driven(R_ci2, L_ci2, u_ou, seed=SEED + 2)
        c_use = c_ou[:, BURNIN:]
        x_use_xy = x_ci_raw[:, BURNIN:, :]
        x_use_ci = np.concatenate([c_use[:, :, np.newaxis], x_use_xy], axis=2)
        trace_names = ["c (OU)", "X", "Y"]

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
        trace_names = ["c (OU)", "X", "Y"] if CI_MODE == "ou" else names

    log_array_stats("CI x_use", x_use_ci, trace_names)
    plot_traces(x_use_ci[0], trace_names,
                f"Common input (U={CI_MODE}, XY={RNN_MODE}): "
                f"U->X(lag0), U->Y(lag{CI_DELAY})",
                OUT, "traces_ci")

    # ── Run DetC + DirC ──
    _t0 = time.time()
    print(f"\n  Running DetC + DirC (common input, U={CI_MODE}, XY={RNN_MODE})...")
    res_ci = run_detc_dirc(
        x_use_ci, lags, pairs,
        e=EMB_DIM, kfolds=KFOLDS, n_surrogates=N_SURR,
        max_ref=MAX_REF, seed=SEED + 5000,
        score_type=SCORE_TYPE, n_jobs=N_JOBS,
        dirc_VU=True,
    )
    print(f"  Finished in {time.time() - _t0:.1f}s\n")

    # ── Plot diagnostics ──
    info_ci = plot_detc_dirc_grid(
        res_ci, x_use_ci, spec_list,
        title=f"Common input (U={CI_MODE}, XY={RNN_MODE}) — DetC / DirC diagnostics",
        out_dir=OUT, fname="ci_diagnostics",
        dirc_VU=True,
    )

    for k in ("ux", "uy", "xu", "yu", "xy", "yx"):
        d = info_ci[k]
        print(f"  {k.upper():>4s}: best_lag={d['best_lag']:+d}  "
              f"p-hat={d['best_phat']:.4f}  p={d['pval']:.2g}  {d['label']}")

    # ==================================================================
    # Scenario C: RecXY
    #   U -> X (lag 0),  X -> Y (lag RECLAG_X2Y),  Y -> X (lag RECLAG_Y2X)
    #   coupling strength Y->X == X->Y
    # ==================================================================
    print("\n" + "=" * 60)
    print(f"SCENARIO C: RecXY ({RNN_MODE})  "
          f"U->X(lag 0), X->Y(lag {RECLAG_X2Y}), Y->X(lag {RECLAG_Y2X})")
    print("=" * 60)

    couple_rec = R_COUPLE if RNN_MODE == "ricker" else LOGISTIC_COUPLE
    C_rec = np.zeros((N_nodes, N_nodes), dtype=np.float64)
    L_rec = np.zeros((N_nodes, N_nodes), dtype=np.int32)
    # U -> X (lag 0)
    C_rec[1, 0] = couple_rec
    L_rec[1, 0] = 0
    # X -> Y (lag RECLAG_X2Y)
    C_rec[2, 1] = couple_rec
    L_rec[2, 1] = RECLAG_X2Y
    # Y -> X (lag RECLAG_Y2X), same strength as X->Y
    C_rec[1, 2] = couple_rec
    L_rec[1, 2] = RECLAG_Y2X

    if RNN_MODE == "ricker":
        R_rec = C_rec.copy()
        for i in range(N_nodes):
            R_rec[i, i] = r_self_3[i]
        print(f"\n  R =\n{R_rec}")
        print(f"  L =\n{L_rec}")
        print("\n  --- Scenario C update equations (Ricker) ---")
        _print_discrete_equations("ricker", R_rec, L_rec, names)
        x_rec = simulate_ricker(R_rec, L_rec, T, N_TRIALS, SEED + 2)
    else:
        print(f"\n  W =\n{C_rec}")
        print(f"  L =\n{L_rec}")
        print("\n  --- Scenario C update equations (Logistic) ---")
        _print_discrete_equations("logistic", C_rec, L_rec, names,
                                  r_per_unit=logistic_r_3)
        x_rec = simulate_logistic(C_rec, L_rec, T, N_TRIALS, SEED + 2,
                                  r=logistic_r_3)
    x_use_r = x_rec[:, BURNIN:, :]
    log_array_stats("RecXY x_use", x_use_r, names)
    plot_traces(x_use_r[0], names,
                f"RecXY ({RNN_MODE}): U->X(lag0), "
                f"X->Y(lag{RECLAG_X2Y}), Y->X(lag{RECLAG_Y2X})",
                OUT, "traces_recxy")

    # ── Run DetC + DirC ──
    _t0 = time.time()
    print("\n  Running DetC + DirC (RecXY)...")
    res_r = run_detc_dirc(
        x_use_r, lags, pairs,
        e=EMB_DIM, kfolds=KFOLDS, n_surrogates=N_SURR,
        max_ref=MAX_REF, seed=SEED + 9000,
        score_type=SCORE_TYPE, n_jobs=N_JOBS,
        dirc_VU=True,
    )
    print(f"  Finished in {time.time() - _t0:.1f}s\n")

    # ── Plot diagnostics ──
    info_r = plot_detc_dirc_grid(
        res_r, x_use_r, spec_list,
        title=f"RecXY ({RNN_MODE}) — DetC / DirC diagnostics",
        out_dir=OUT, fname="recxy_diagnostics",
        dirc_VU=True,
    )

    for k in ("ux", "uy", "xu", "yu", "xy", "yx"):
        d = info_r[k]
        print(f"  {k.upper():>4s}: best_lag={d['best_lag']:+d}  "
              f"p-hat={d['best_phat']:.4f}  p={d['pval']:.2g}  {d['label']}")

    # ==================================================================
    # Summary
    # ==================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    def _print_summary_block(label_names, info_dict, res):
        for k, (s, t) in zip(["ux", "uy", "xu", "yu", "xy", "yx"],
                              [(0,1),(0,2),(1,0),(2,0),(1,2),(2,1)]):
            d = info_dict[k]
            drc = str(res["dirc_label"][s, t])
            p_tau_uv = float(res["dirc_pval_UV"][s, t])
            p_tau_vu = float(res["dirc_pval_VU"][s, t])
            p_rs_uv  = float(res["dirc_ranksum_pval_UV"][s, t])
            p_rs_vu  = float(res["dirc_ranksum_pval_VU"][s, t])
            beta_uv  = float(res["dirc_beta_UV"][s, t])
            beta_vu  = float(res["dirc_beta_VU"][s, t])
            p_beta_uv = float(res["dirc_pval_beta_UV"][s, t])
            p_beta_vu = float(res["dirc_pval_beta_VU"][s, t])
            det = int(res["detected"][s, t])
            reason = str(res["detected_reason"][s, t])
            verdict = "DETECTED" if det else f"REJECTED ({reason})"
            print(f"    {label_names[s]}->{label_names[t]:2s}  "
                  f"lag={d['best_lag']:+3d}  "
                  f"p-hat={d['best_phat']:.3f}  p={d['pval']:.2g}  "
                  f"DirC: {drc}  "
                  f"[UV p_tau={p_tau_uv:.2g} p_rs={p_rs_uv:.2g} "
                  f"beta={beta_uv:+.3g} p_beta={p_beta_uv:.2g} | "
                  f"VU p_tau={p_tau_vu:.2g} p_rs={p_rs_vu:.2g} "
                  f"beta={beta_vu:+.3g} p_beta={p_beta_vu:.2g}]  "
                  f"--> {verdict}")

    print("\n  Causal chain  U -> X (lag 0) -> Y (lag {})".format(CI_DELAY))
    print("  " + "-" * 56)
    _print_summary_block(names, info_c, res_c)

    print(f"\n  Common input  U -> X (lag 0), U -> Y (lag {CI_DELAY})")
    print("  " + "-" * 56)
    _print_summary_block(trace_names, info_ci, res_ci)

    print(f"\n  RecXY  U->X(lag 0), "
          f"X->Y(lag {RECLAG_X2Y}), Y->X(lag {RECLAG_Y2X})")
    print("  " + "-" * 56)
    _print_summary_block(names, info_r, res_r)

    elapsed = time.time() - _t0_global
    print(f"\n  Total elapsed: {elapsed:.1f}s ({elapsed / 60:.1f}min)")
    print(f"  Output directory: {OUT}")
    # Restore stdout/stderr before closing log to avoid teardown exceptions
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    _log_fh.close()
