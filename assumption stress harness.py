"""
Assumption Stress Harness for Split Conformal Prediction
=========================================================
Author : Mariam Mohamed Elelidy
Purpose: Systematically break each assumption of split conformal prediction,
         one at a time, and measure the exact coverage damage it causes.

Motivation
----------
Split conformal prediction comes with a finite-sample coverage guarantee —
but only under specific assumptions. The guarantee says nothing about what
happens when those assumptions are violated in deployment.

This harness answers the question practitioners actually face:
"If my data violates assumption X by degree Y, how far does coverage fall?"

Assumptions tested
------------------
  1. BASELINE         — all assumptions satisfied (sanity check)
  2. COVARIATE SHIFT  — test X drawn from a shifted distribution
  3. LABEL NOISE SHIFT — test noise σ differs from calibration noise σ
  4. TEMPORAL ORDER    — data sorted by index; cal/test not exchangeable
  5. HEAVY TAILS       — calibration contaminated with outliers (inflates q)
  6. SMALL CAL SET     — n_cal reduced; quantile grid becomes coarse
  7. CONCEPT DRIFT     — true weights w* change between cal and test

Output
------
  For each scenario: empirical coverage, avg interval width, q, and a
  coverage damage score vs baseline. All results stored as tensors.

Usage
-----
    python "assumption stress harness.py"
    python "assumption stress harness.py" --seeds 20 --alpha 0.05
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch


# ────────────────────────────────────────────────────────────────────────────
# Core primitives (same as baseline conformal_prediction.py)
# ────────────────────────────────────────────────────────────────────────────

def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    """Closed-form ridge: ŵ = (X'X + λI)⁻¹ X'y."""
    A = X.T @ X + lam * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ y)


def conformal_quantile(abs_residuals: np.ndarray, alpha: float) -> float:
    """Standard finite-sample rank rule: k = ⌈(n+1)(1-α)⌉-th order statistic."""
    n = abs_residuals.size
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)
    return float(np.sort(abs_residuals)[k - 1])


# ────────────────────────────────────────────────────────────────────────────
# Data generation primitives
# ────────────────────────────────────────────────────────────────────────────

def make_data(
    rng: np.random.Generator,
    n: int,
    d: int,
    noise_scale: float = 0.6,
    w_true: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate (X, y, w_true). Returns w_true so it can be reused."""
    X = rng.normal(size=(n, d))
    if w_true is None:
        w_true = rng.normal(size=(d,))
    y = X @ w_true + noise_scale * rng.normal(size=(n,))
    return X, y, w_true


def split_indices(rng: np.random.Generator, n: int, n_train: int, n_cal: int):
    idx = rng.permutation(n)
    tr  = idx[:n_train]
    cal = idx[n_train : n_train + n_cal]
    te  = idx[n_train + n_cal :]
    return tr, cal, te


# ────────────────────────────────────────────────────────────────────────────
# Evaluation kernel (shared by all scenarios)
# ────────────────────────────────────────────────────────────────────────────

def evaluate(
    w: np.ndarray,
    X_cal: np.ndarray, y_cal: np.ndarray,
    X_te:  np.ndarray, y_te:  np.ndarray,
    alpha: float,
) -> dict[str, float]:
    """Calibrate on (X_cal, y_cal) and evaluate coverage on (X_te, y_te)."""
    cal_pred = X_cal @ w
    abs_res  = np.abs(y_cal - cal_pred)
    q        = conformal_quantile(abs_res, alpha)

    te_pred = X_te @ w
    lo = te_pred - q
    hi = te_pred + q
    covered = (y_te >= lo) & (y_te <= hi)

    return {
        "coverage":  float(np.mean(covered)),
        "avg_width": float(np.mean(hi - lo)),
        "q":         q,
        "n_cal":     len(y_cal),
        "n_test":    len(y_te),
    }


# ────────────────────────────────────────────────────────────────────────────
# Scenario definitions
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class Scenario:
    name:        str
    key:         str
    description: str
    assumption_broken: str
    run_fn:      Callable   # (rng, n, d, alpha) -> dict


def _baseline(rng, n, d, alpha, n_cal_frac=0.20, noise=0.6, **_):
    n_tr  = int(0.6  * n)
    n_cal = int(n_cal_frac * n)
    X, y, w_true = make_data(rng, n, d, noise_scale=noise)
    tr, cal, te  = split_indices(rng, n, n_tr, n_cal)
    w = ridge_fit(X[tr], y[tr])
    return evaluate(w, X[cal], y[cal], X[te], y[te], alpha)


def scenario_baseline(rng, n, d, alpha):
    """All assumptions satisfied. Sets the coverage ceiling."""
    return _baseline(rng, n, d, alpha)


def scenario_covariate_shift(rng, n, d, alpha, shift_magnitude=3.0):
    """Test X drawn from N(shift, I) instead of N(0, I).

    Breaks: exchangeability between calibration and test residuals.
    Expected effect: model extrapolates; residuals grow; coverage drops.
    """
    n_tr  = int(0.6 * n)
    n_cal = int(0.2 * n)

    X, y, w_true = make_data(rng, n, d)
    tr, cal, te  = split_indices(rng, n, n_tr, n_cal)
    w = ridge_fit(X[tr], y[tr])

    # Calibration: in-distribution
    X_cal, y_cal = X[cal], y[cal]

    # Test: shifted distribution
    X_te_shift = X[te] + shift_magnitude          # covariate shift
    y_te_shift = X_te_shift @ w_true + 0.6 * rng.normal(size=len(te))

    return evaluate(w, X_cal, y_cal, X_te_shift, y_te_shift, alpha)


def scenario_label_noise_shift(rng, n, d, alpha, test_noise_scale=2.5):
    """Test labels have higher noise than calibration labels.

    Breaks: representative calibration split.
    Expected effect: calibration underestimates q needed for noisy test;
    coverage drops because intervals are too narrow.
    """
    n_tr  = int(0.6 * n)
    n_cal = int(0.2 * n)

    X, _, w_true = make_data(rng, n, d)
    tr, cal, te  = split_indices(rng, n, n_tr, n_cal)
    w = ridge_fit(X[tr] , X[tr] @ w_true + 0.6 * rng.normal(size=n_tr))

    # Calibration: low noise
    y_cal = X[cal] @ w_true + 0.6 * rng.normal(size=len(cal))
    # Test: high noise — residuals will be larger than calibration expected
    y_te  = X[te]  @ w_true + test_noise_scale * rng.normal(size=len(te))

    return evaluate(w, X[cal], y_cal, X[te], y_te, alpha)


def scenario_temporal_ordering(rng, n, d, alpha, trend_slope=0.05):
    """Data ordered by time; calibration and test are not exchangeable.

    Breaks: exchangeability — later points have systematically different
    residuals due to concept drift accumulating over time.
    """
    n_tr  = int(0.6 * n)
    n_cal = int(0.2 * n)

    X, _, w_true = make_data(rng, n, d)

    # Add a time trend: y drifts upward over time
    time_idx = np.arange(n)
    y = X @ w_true + 0.6 * rng.normal(size=n) + trend_slope * time_idx

    # Use sequential (non-permuted) splits to simulate temporal ordering
    tr  = np.arange(n_tr)
    cal = np.arange(n_tr, n_tr + n_cal)
    te  = np.arange(n_tr + n_cal, n)

    w = ridge_fit(X[tr], y[tr])
    return evaluate(w, X[cal], y[cal], X[te], y[te], alpha)


def scenario_heavy_tails(rng, n, d, alpha, contamination=0.10, outlier_scale=8.0):
    """Calibration contaminated with large-residual outliers.

    Breaks: the conformal quantile is inflated by outlier residuals,
    producing over-wide intervals and artificially high coverage.
    """
    n_tr  = int(0.6 * n)
    n_cal = int(0.2 * n)

    X, y, w_true = make_data(rng, n, d)
    tr, cal, te  = split_indices(rng, n, n_tr, n_cal)
    w = ridge_fit(X[tr], y[tr])

    # Contaminate calibration labels with outliers
    y_cal_contaminated = y[cal].copy()
    n_outliers = int(contamination * len(cal))
    outlier_idx = rng.choice(len(cal), size=n_outliers, replace=False)
    y_cal_contaminated[outlier_idx] += outlier_scale * rng.normal(size=n_outliers)

    return evaluate(w, X[cal], y_cal_contaminated, X[te], y[te], alpha)


def scenario_small_cal_set(rng, n, d, alpha, n_cal_override=15):
    """Very small calibration set — quantile grid is coarse.

    Breaks: with few calibration points, the k-th order statistic jumps
    in large increments. Coverage variability explodes; mean may hold but
    individual runs become unreliable.
    """
    n_tr = int(0.6 * n)
    X, y, w_true = make_data(rng, n, d)

    tr_idx  = np.arange(n_tr)
    cal_idx = np.arange(n_tr, n_tr + n_cal_override)
    te_idx  = np.arange(n_tr + n_cal_override, n)

    w = ridge_fit(X[tr_idx], y[tr_idx])
    return evaluate(w, X[cal_idx], y[cal_idx], X[te_idx], y[te_idx], alpha)


def scenario_concept_drift(rng, n, d, alpha, drift_scale=1.5):
    """True weights w* change between calibration and test.

    Breaks: residual distribution on test differs from calibration residuals
    because the data-generating process has changed.
    """
    n_tr  = int(0.6 * n)
    n_cal = int(0.2 * n)

    X, y, w_true = make_data(rng, n, d)
    tr, cal, te  = split_indices(rng, n, n_tr, n_cal)
    w = ridge_fit(X[tr], y[tr])

    # Calibration: original w*
    y_cal = X[cal] @ w_true + 0.6 * rng.normal(size=len(cal))

    # Test: drifted w* — model's predictions are now systematically wrong
    w_drifted = w_true + drift_scale * rng.normal(size=d)
    y_te = X[te] @ w_drifted + 0.6 * rng.normal(size=len(te))

    return evaluate(w, X[cal], y_cal, X[te], y_te, alpha)


# ────────────────────────────────────────────────────────────────────────────
# Scenario registry
# ────────────────────────────────────────────────────────────────────────────

SCENARIOS: list[Scenario] = [
    Scenario(
        name="Baseline",
        key="baseline",
        description="All assumptions satisfied. Sets the coverage ceiling.",
        assumption_broken="—",
        run_fn=scenario_baseline,
    ),
    Scenario(
        name="Covariate Shift",
        key="covariate_shift",
        description="Test X drawn from N(3, I) instead of N(0, I).",
        assumption_broken="Exchangeability",
        run_fn=scenario_covariate_shift,
    ),
    Scenario(
        name="Label Noise Shift",
        key="label_noise",
        description="Test noise σ = 2.5× calibration noise σ.",
        assumption_broken="Representative calibration",
        run_fn=scenario_label_noise_shift,
    ),
    Scenario(
        name="Temporal Ordering",
        key="temporal",
        description="Data sorted by time index; cal/test not exchangeable.",
        assumption_broken="Exchangeability (temporal)",
        run_fn=scenario_temporal_ordering,
    ),
    Scenario(
        name="Heavy Tails",
        key="heavy_tails",
        description="10% of calibration points contaminated with 8σ outliers.",
        assumption_broken="Representative calibration (outlier inflation)",
        run_fn=scenario_heavy_tails,
    ),
    Scenario(
        name="Small Cal Set",
        key="small_cal",
        description="n_cal = 15 instead of 120.",
        assumption_broken="Calibration set size",
        run_fn=scenario_small_cal_set,
    ),
    Scenario(
        name="Concept Drift",
        key="concept_drift",
        description="True weights w* shift by N(0, 1.5²) between cal and test.",
        assumption_broken="Consistent data-generating process",
        run_fn=scenario_concept_drift,
    ),
]


# ────────────────────────────────────────────────────────────────────────────
# Multi-seed runner
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class ScenarioStats:
    name:              str
    assumption_broken: str
    mean_coverage:     float
    std_coverage:      float
    mean_width:        float
    std_width:         float
    mean_q:            float
    coverage_damage:   float      # vs baseline mean coverage
    raw:               torch.Tensor = field(repr=False)   # [seeds, 3]


def run_scenario(
    scenario: Scenario,
    seeds: list[int],
    n: int,
    d: int,
    alpha: float,
) -> ScenarioStats:
    rows = []
    for s in seeds:
        rng = np.random.default_rng(s)
        result = scenario.run_fn(rng, n, d, alpha)
        rows.append([result["coverage"], result["avg_width"], result["q"]])

    t = torch.tensor(rows, dtype=torch.float32)
    mean = t.mean(dim=0)
    std  = t.std(dim=0, unbiased=False)

    return ScenarioStats(
        name=scenario.name,
        assumption_broken=scenario.assumption_broken,
        mean_coverage=mean[0].item(),
        std_coverage=std[0].item(),
        mean_width=mean[1].item(),
        std_width=std[1].item(),
        mean_q=mean[2].item(),
        coverage_damage=0.0,    # filled in after baseline computed
        raw=t,
    )


# ────────────────────────────────────────────────────────────────────────────
# Terminal report
# ────────────────────────────────────────────────────────────────────────────

def _bar(value: float, lo: float = 0.0, hi: float = 1.0, width: int = 20) -> str:
    if hi <= lo:
        hi = lo + 1e-9
    x    = max(0.0, min(1.0, (value - lo) / (hi - lo)))
    fill = int(round(x * width))
    return "█" * fill + "░" * (width - fill)


def _damage_label(damage: float) -> str:
    if   damage >=  0.005: return "▲ OVER-COVERED (too wide)"
    elif damage >= -0.005: return "✓ on target"
    elif damage >= -0.03:  return "▼ mild drop"
    elif damage >= -0.08:  return "▼▼ moderate drop"
    else:                  return "▼▼▼ SEVERE DROP"


def print_report(
    stats_list: list[ScenarioStats],
    target: float,
    n: int,
    d: int,
    alpha: float,
    n_seeds: int,
) -> None:
    sep = "─" * 78
    print()
    print("┌" + sep + "┐")
    print("│  Assumption Stress Harness — Split Conformal Prediction" + " " * 22 + "│")
    print(f"│  α = {alpha:.2f}  │  target coverage = {target:.2f}  │  "
          f"n = {n}  d = {d}  │  seeds = {n_seeds}" + " " * 14 + "│")
    print("└" + sep + "┘")
    print()
    print("  Each scenario breaks exactly one assumption.")
    print("  'Coverage damage' = scenario mean coverage − baseline mean coverage.")
    print()

    # Summary table
    col = f"  {'Scenario':<22}  {'Assumption broken':<35}  {'coverage':>8}  {'±':>6}  {'damage':>8}"
    print(col)
    print("  " + "─" * 86)
    for st in stats_list:
        dmg_str  = f"{st.coverage_damage:+.4f}"
        flag     = _damage_label(st.coverage_damage)
        bar_str  = _bar(st.mean_coverage, 0.0, 1.0, 16)
        print(
            f"  {st.name:<22}  {st.assumption_broken:<35}  "
            f"{st.mean_coverage:.4f}  ±{st.std_coverage:.4f}  "
            f"{dmg_str}  {flag}"
        )
    print()

    # Per-scenario detail
    for st in stats_list:
        print(f"  {'─'*78}")
        print(f"  {st.name}  (broken: {st.assumption_broken})")
        cov_bar = _bar(st.mean_coverage, 0.0, 1.0)
        tgt_bar = _bar(target, 0.0, 1.0)
        print(f"    coverage  {st.mean_coverage:.4f} ±{st.std_coverage:.4f}  {cov_bar}")
        print(f"    target    {target:.4f}            {tgt_bar}")
        print(f"    width     {st.mean_width:.4f} ±{st.std_width:.4f}")
        print(f"    q         {st.mean_q:.4f}")
        print(f"    damage    {st.coverage_damage:+.4f}  {_damage_label(st.coverage_damage)}")
        print(f"    raw tensor [seeds × 3]  shape={list(st.raw.shape)}")
        print(f"    {st.raw}")
    print()


# ────────────────────────────────────────────────────────────────────────────
# Final tensor summary
# ────────────────────────────────────────────────────────────────────────────

def print_tensor_summary(stats_list: list[ScenarioStats]) -> None:
    print("═" * 78)
    print("FINAL TENSOR SUMMARY")
    print("Rows: scenarios (in order).  Cols: [mean_coverage, mean_width, mean_q, damage]")
    print("═" * 78)
    data = [
        [st.mean_coverage, st.mean_width, st.mean_q, st.coverage_damage]
        for st in stats_list
    ]
    final = torch.tensor(data, dtype=torch.float32)
    print(final)
    print()
    print("Scenario index map:")
    for i, st in enumerate(stats_list):
        print(f"  [{i}]  {st.name}")


# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Assumption stress harness for split conformal prediction"
    )
    p.add_argument("--n",     type=int,   default=600,  help="dataset size")
    p.add_argument("--d",     type=int,   default=8,    help="feature dimension")
    p.add_argument("--alpha", type=float, default=0.10, help="miscoverage level")
    p.add_argument("--seeds", type=int,   default=50,   help="seeds per scenario")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    seeds  = list(range(args.seeds))
    target = 1.0 - args.alpha

    print(f"\nRunning {len(SCENARIOS)} scenarios × {len(seeds)} seeds …")

    stats_list: list[ScenarioStats] = []
    for sc in SCENARIOS:
        print(f"  {sc.name:<30}", end="", flush=True)
        st = run_scenario(sc, seeds, args.n, args.d, args.alpha)
        stats_list.append(st)
        print(f"coverage = {st.mean_coverage:.4f} ± {st.std_coverage:.4f}")

    # Compute coverage damage relative to baseline
    baseline_cov = stats_list[0].mean_coverage
    for st in stats_list:
        st.coverage_damage = st.mean_coverage - baseline_cov

    print_report(stats_list, target, args.n, args.d, args.alpha, len(seeds))
    print_tensor_summary(stats_list)


if __name__ == "__main__":
    main()
