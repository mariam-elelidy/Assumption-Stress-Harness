# Assumption Stress Harness — Split Conformal Prediction

> *A reliability method that fails silently under mild distributional shift is more dangerous than no reliability method at all.*

---

## What this is

A systematic harness that **breaks each assumption of split conformal prediction independently**, measures the exact coverage damage, and classifies whether the failure is detectable from the intervals themselves.

This is the companion artifact to [`conformal_prediction.py`](https://gist.github.com/mariam-elelidy/9257762ee0e8e8df2ba6cdb5908076fd), which demonstrates that the method works when assumptions hold. This harness answers the prior question: **when do those assumptions fail in practice, and how bad is it?**

---

## Core result

Seven scenarios tested across 50 seeds each (n=600, d=8, α=0.10):

| Scenario | Coverage | Damage | Detectable from width? |
|---|---|---|---|
| Baseline | 0.909 | — | — |
| Covariate Shift | 0.876 | -0.032 | No — width unchanged |
| **Label Noise Shift** | **0.305** | **-0.604** | **No — silent failure** |
| **Temporal Ordering** | **0.126** | **-0.783** | Width exploded (48×) — but coverage still 0.13 |
| Heavy Tails | 0.984 | +0.075 | Width 50% wider — over-covered, not under |
| Small Cal Set | 0.933 | +0.024 | Width variance 13× higher |
| **Concept Drift** | **0.201** | **-0.708** | **No — silent failure** |

**Three of six violations cause coverage below 0.25 while interval width stays unchanged.** A system monitoring width as a proxy for reliability would miss all three.

---

## Quick start

```bash
pip install numpy torch

# Run with defaults (n=600, d=8, α=0.10, 50 seeds)
python "assumption stress harness.py"

# More seeds for tighter confidence intervals
python "assumption stress harness.py" --seeds 200 --alpha 0.05
```

**CLI arguments:**

| Flag | Default | Description |
|---|---|---|
| `--n` | 600 | Dataset size |
| `--d` | 8 | Feature dimension |
| `--alpha` | 0.10 | Miscoverage level (1-α = target coverage) |
| `--seeds` | 50 | Seeds per scenario |

---

## Scenarios

```
[0] Baseline          — all assumptions satisfied (coverage ceiling)
[1] Covariate Shift   — test X ~ N(3, I) instead of N(0, I)
[2] Label Noise Shift — test σ = 2.5× calibration σ
[3] Temporal Ordering — sequential split on time-trended data
[4] Heavy Tails       — 10% calibration outliers inflate q
[5] Small Cal Set     — n_cal = 15 instead of 120
[6] Concept Drift     — w* shifts between calibration and test
```

Each scenario is implemented as a single function and registered in `SCENARIOS`. Adding a new scenario requires writing one function and one `Scenario(...)` entry — nothing else changes.

---

## How to read the output

**Coverage damage** is the primary metric: how far does mean coverage fall vs the baseline where all assumptions hold?

**Width** is the secondary signal: does the interval width change under the violation? If coverage drops but width is stable, the failure is **silent** — nothing in the intervals signals the problem.

**Severity classification:**

| Damage | Label |
|---|---|
| < −0.50 with stable width | ▼▼▼ SEVERE DROP (silent) |
| −0.30 to −0.50 | ▼▼▼ SEVERE DROP |
| −0.03 to −0.30 | ▼▼ or ▼ moderate/mild |
| ±0.03 | ✓ on target |
| > +0.03 | ▲ OVER-COVERED (too wide) |

---

## Key findings

**Temporal ordering is the worst-case scenario.** Time-ordered data is the norm in deployment (sensor streams, medical records, financial data), not an edge case. Treating it as "roughly exchangeable" produced the single worst outcome: coverage 0.126 with interval width 48× baseline. Both useless and confidently wrong simultaneously.

**Silent failures are the real risk.** Label noise shift and concept drift produced coverage ~0.20 with width nearly identical to baseline. Interval width — the most natural proxy for uncertainty — gave no signal. These failures require ground-truth monitoring to detect, not just interval inspection.

**Outlier inflation and small calibration sets have opposite diagnostics.** Heavy tails produce systematically wide intervals (detectable). Small calibration sets produce high-variance intervals where the mean looks fine but individual runs are unreliable.

**Covariate shift is the recoverable case.** Only −0.032 damage; correctable with importance-weighted conformal prediction (Tibshirani et al., 2019).

---

## Repository layout

```
├── README.md                     ← this file
├── assumption stress harness.py  ← harness implementation
├── output.txt                    ← annotated run output (50 seeds)
└── writeup.md                    ← full technical writeup
```

See [`writeup.md`](writeup.md) for the complete analysis including failure severity classification, cross-scenario patterns, and connections to the companion conformal prediction artifact.

---

## Extending the harness

To add a new scenario:

```python
def scenario_my_violation(rng, n, d, alpha):
    # break one assumption here
    ...
    return evaluate(w, X_cal, y_cal, X_te, y_te, alpha)

SCENARIOS.append(Scenario(
    name="My Violation",
    key="my_violation",
    description="What changes.",
    assumption_broken="Which assumption breaks.",
    run_fn=scenario_my_violation,
))
```

Everything else — multi-seed runner, report formatting, tensor summary — is generic.

---

## References

- Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.
- Angelopoulos, A. N., & Bates, S. (2023). Conformal prediction: A gentle introduction. *Foundations and Trends in Machine Learning*, 16(4), 494–591.
- Tibshirani, R. J., Barber, R. F., Candès, E. J., & Ramdas, A. (2019). Conformal prediction under covariate shift. *NeurIPS*.
- Barber, R. F., Candès, E. J., Ramdas, A., & Tibshirani, R. J. (2023). Conformal prediction beyond exchangeability. *Annals of Statistics*.

---
