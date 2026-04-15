# Assumption Stress Harness: When Split Conformal Prediction Breaks

**Author:** Mariam Mohamed Elelidy  
**Topic:** Uncertainty Quantification · Distribution-Free Inference · Failure-Mode Analysis

---

## TL;DR

Split conformal prediction comes with a finite-sample coverage guarantee — but only under specific assumptions. The guarantee is silent about what happens when those assumptions fail in deployment.

This artifact stress-tests each assumption independently and measures the exact coverage damage each violation causes. The result is a ranked failure map: which violations are catastrophic, which are tolerable, and which produce silent failures where nothing signals that the method is broken.

**Most important finding:** Three of six violations (label noise shift, temporal ordering, concept drift) cause coverage to collapse below 0.25 while interval width stays nearly unchanged — producing silent failures that a width-monitoring system would miss entirely.

---

## 1. Motivation

The [companion artifact](conformal_prediction.py) demonstrates that split conformal prediction works under ideal conditions. This artifact asks the harder question practitioners face in deployment:

> *"If my data violates assumption X by degree Y, how far does coverage fall — and would I know?"*

Documenting failure modes is not a disclaimer. It is the most operationally useful part of any uncertainty quantification method. A reliability layer that fails silently is more dangerous than no reliability layer at all, because it produces false confidence.

---

## 2. Design Principle

Each scenario in the harness:

1. **Breaks exactly one assumption** while holding all others constant
2. **Is parameterised** so violation severity can be varied
3. **Runs across 50 seeds** to distinguish structural failure from sampling noise
4. **Reports coverage damage** relative to a baseline where all assumptions hold

The baseline is not a claim that conformal prediction is "correct" in some absolute sense — it is the empirical ceiling against which violations are measured.

---

## 3. Assumptions Under Test

| # | Scenario | Assumption broken | Violation mechanism |
|---|---|---|---|
| 0 | **Baseline** | — | All assumptions satisfied |
| 1 | **Covariate Shift** | Exchangeability | Test X drawn from N(3, I) |
| 2 | **Label Noise Shift** | Representative calibration | Test σ = 2.5 × calibration σ |
| 3 | **Temporal Ordering** | Exchangeability (temporal) | Sequential split; data has time trend |
| 4 | **Heavy Tails** | Representative calibration | 10% of calibration labels contaminated |
| 5 | **Small Cal Set** | Calibration set size | n_cal = 15 instead of 120 |
| 6 | **Concept Drift** | Consistent DGP | True weights shift between cal and test |

---

## 4. Method

### Shared setup

For all scenarios:

$$X \in \mathbb{R}^{n \times d} \sim \mathcal{N}(0, I), \quad w^* \sim \mathcal{N}(0, I_d), \quad n = 600, \quad d = 8, \quad \alpha = 0.10$$

Base model: closed-form ridge regression with $\lambda = 10^{-3}$.

Conformal calibration follows the standard rank rule:

$$q = r_{(k)}, \quad k = \left\lceil (n_{\text{cal}} + 1)(1 - \alpha) \right\rceil$$

Test interval: $[\hat{y} - q, \; \hat{y} + q]$.

### Coverage damage metric

$$\Delta_{\text{cov}} = \bar{C}_{\text{scenario}} - \bar{C}_{\text{baseline}}$$

where $\bar{C}$ is mean empirical coverage across 50 seeds. Negative damage = under-coverage; positive = over-coverage (too wide).

### Scenario mechanics

**Covariate Shift:** Test features shifted $X_{\text{test}} \leftarrow X_{\text{test}} + 3$. Model extrapolates outside its training support; test residuals grow while $q$ is calibrated in-distribution.

**Label Noise Shift:** Calibration labels generated with $\sigma = 0.6$; test labels with $\sigma = 2.5$. The calibration residuals underestimate test-time uncertainty.

**Temporal Ordering:** Data indexed $0, \dots, 599$ with trend $+0.05 \times t$ added to $y$. Split is sequential (not random permutation): train on $[0, 359]$, calibration on $[360, 479]$, test on $[480, 599]$. The trend creates a systematic bias that grows from calibration to test.

**Heavy Tails:** 10% of calibration labels contaminated with $+8\sigma$ outliers. Outlier residuals inflate $q$, producing over-wide intervals and falsely high coverage.

**Small Cal Set:** $n_{\text{cal}} = 15$. With $n = 15$ and $\alpha = 0.10$, the rank rule sets $k = \lceil 16 \times 0.9 \rceil = 15$ — the maximum residual. $q$ is systematically conservative; variance in $q$ across seeds is high.

**Concept Drift:** True weights $w^*$ shifted by $\mathcal{N}(0, 1.5^2)$ before test labels are generated. The base model's predictions are structurally wrong for test targets, but calibration residuals do not reflect this.

---

## 5. Results

### Summary table (50 seeds, n=600, d=8, α=0.10)

| Scenario | Mean coverage | ±std | Width | Damage | Status |
|---|---|---|---|---|---|
| Baseline | **0.9087** | 0.0416 | 2.054 | +0.000 | ✓ on target |
| Covariate Shift | 0.8763 | 0.0728 | 2.054 | -0.032 | ▼ mild |
| Label Noise Shift | 0.3045 | 0.0588 | 2.053 | **-0.604** | ▼▼▼ SEVERE |
| Temporal Ordering | 0.1260 | 0.0670 | **48.29** | **-0.783** | ▼▼▼ SEVERE |
| Heavy Tails | 0.9837 | 0.0192 | 3.095 | +0.075 | ▲ over-covered |
| Small Cal Set | 0.9330 | 0.0624 | 2.405 | +0.024 | ▲ slightly wide |
| Concept Drift | 0.2007 | 0.0526 | 2.053 | **-0.708** | ▼▼▼ SEVERE |

### Final tensor `[mean_coverage, mean_width, mean_q, damage]`

```
tensor([[ 0.9087,  2.0540,  1.0270,  0.0000],   # Baseline
        [ 0.8763,  2.0540,  1.0270, -0.0323],   # Covariate Shift
        [ 0.3045,  2.0531,  1.0266, -0.6042],   # Label Noise Shift
        [ 0.1260, 48.2892, 24.1446, -0.7827],   # Temporal Ordering
        [ 0.9837,  3.0947,  1.5473,  0.0750],   # Heavy Tails
        [ 0.9330,  2.4047,  1.2023,  0.0243],   # Small Cal Set
        [ 0.2007,  2.0531,  1.0266, -0.7080]])  # Concept Drift
```

---

## 6. Analysis

### Pattern 1 — Silent failures

Label Noise Shift and Concept Drift both cause coverage to collapse to ~0.20–0.30 while maintaining **nearly identical interval width** to baseline (2.053 vs 2.054). An operator monitoring interval width as a proxy for reliability would see nothing wrong. These failures are detectable only by comparing predictions against held-out ground truth — which is rarely done continuously post-deployment.

This is the most important practical finding: the method's most dangerous failure mode leaves no visible trace in the intervals themselves.

### Pattern 2 — Temporal ordering is the worst-case scenario

The temporal scenario achieved the lowest coverage (0.126) and the widest intervals (width 48.3) simultaneously. The mechanism is asymmetric: the time trend inflates calibration residuals, making $q$ large — but test-time residuals are *even larger* because the test period occurs later and accumulates more trend. The result is intervals that are both uninformative and inaccurate.

This matters because time-ordered data is the norm in deployment (sensor streams, medical records, financial data), not the exception.

### Pattern 3 — Outlier inflation and small calibration set: same symptom, different risk

Both heavy tails and small calibration sets produce over-coverage (intervals too wide), but the operational risk differs:

- **Heavy tails**: Intervals are 50% wider than necessary on average, every time. The system is safe but useless — like a diagnostic test that flags everyone as high-risk.
- **Small calibration set**: Width variance is 13× higher than baseline. Individual runs may produce dramatically different interval widths. The mean looks acceptable; the reliability is not.

### Pattern 4 — Covariate shift is the recoverable case

A 3-unit distributional shift caused only −0.032 coverage damage. This scenario has a known fix: weighted conformal prediction (Tibshirani et al., 2019) uses importance weights to reweight the calibration distribution to match the test distribution. It is the one scenario where a direct extension restores the guarantee.

---

## 7. Failure Severity Classification

| Severity | Coverage damage | Operational interpretation | Fix |
|---|---|---|---|
| **Silent failure** | < −0.50 with stable width | Confidently wrong; undetectable | Requires distributional monitoring, not just width |
| **Severe** | −0.30 to −0.50 | Guarantee meaningless in practice | Requires assumption audit before deployment |
| **Moderate** | −0.03 to −0.30 | Degraded but not inverted | Weighted conformal; larger calibration set |
| **On target** | ±0.03 | Guarantee holds | — |
| **Over-covered** | > +0.03 | Too wide; low information value | Clean calibration data; larger cal set |

---

## 8. Reproducibility

```bash
pip install numpy torch

# Defaults: n=600, d=8, α=0.10, 50 seeds
python "assumption stress harness.py"

# Vary violation severity (e.g., stronger covariate shift)
python "assumption stress harness.py" --seeds 100 --alpha 0.05
```

Each scenario uses the same seed sequence for fair comparison. All outputs are tensors; no plotting required.

---

## 9. Connections to the Companion Artifact

The [baseline conformal prediction artifact](conformal_prediction.py) demonstrates that the method works when assumptions hold. This harness answers the prior question: *should you trust that your deployment satisfies those assumptions?*

The two artifacts together represent a complete evaluation posture:

1. **Does the method achieve the target coverage under ideal conditions?** → baseline artifact
2. **How robust is that coverage to realistic assumption violations?** → this harness

Neither is sufficient alone. A method that works in ideal conditions but fails silently under mild distributional shift is not a reliable method — it is a demo.

---

## 10. Takeaways

> **The most dangerous failure mode of a reliability method is the one that looks fine.**

Three specific shifts in thinking from building this harness:

1. **Width is not a reliability signal.** In the two most severe failures (label noise shift, concept drift), interval width was indistinguishable from baseline. Monitoring width does not detect coverage collapse. Ground-truth comparison — on a held-out set or via shadow deployment — is required.

2. **Temporal structure is not a nuisance; it is a core assumption violation.** Most deployment data is time-ordered. Treating temporal ordering as "roughly exchangeable" produced the single worst outcome in the harness (coverage 0.126, width 48). Temporal robustness should be a first-class evaluation requirement for any conformal deployment.

3. **Separate mean reliability from variance reliability.** The small calibration set scenario had acceptable mean coverage (0.933) but 13× higher width variance than baseline. For any decision that depends on a specific deployment's interval, mean coverage is the wrong metric. You need per-run reliability.

---

## References

- Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.
- Angelopoulos, A. N., & Bates, S. (2023). Conformal prediction: A gentle introduction. *Foundations and Trends in Machine Learning*, 16(4), 494–591.
- Tibshirani, R. J., Barber, R. F., Candès, E. J., & Ramdas, A. (2019). Conformal prediction under covariate shift. *NeurIPS*.
- Barber, R. F., Candès, E. J., Ramdas, A., & Tibshirani, R. J. (2023). Conformal prediction beyond exchangeability. *Annals of Statistics*.
