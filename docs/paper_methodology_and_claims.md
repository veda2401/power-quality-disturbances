# Rigorous Methodology, Algorithmic Verification & Scientific Claims for Open-Set PQD Benchmark

## 1. Algorithmic Verification: OpenMax (Euclidean Logit-Space Variant)

### Canonical vs Variant Specification
Canonical OpenMax ([Bendale & Boult, CVPR 2016](https://arxiv.org/abs/1511.06233)) recalibrates network activations by fitting Weibull distributions to distances from class Mean Activation Vectors (MAVs). Our benchmark implements a self-contained **Euclidean Logit-Space OpenMax Variant**, designed for numerical stability and zero external C-bindings (no `libMR` dependencies).

```
Unnormalized Logits v(x) ∈ ℝ^K
       ↓
Compute Euclidean Distance to MAV: d(x, μ_c) = ||v(x) - μ_c||₂
       ↓
Weibull CDF Weighting for Top-α Classes:
  ω_c(x) = 1.0 - ((α - rank(c)) / α) * Weibull_CDF(d(x, μ_c); shape_c, scale_c)
       ↓
Revised Logits: v̂_c(x) = v_c(x) * ω_c(x)
Synthetic Unknown Logit: v̂₀(x) = ∑_{c=1}^K v_c(x) * (1.0 - ω_c(x))
       ↓
Recalibrated Probabilities via Softmax:
  P̂(y = c | x) = exp(v̂_c(x)) / [ exp(v̂₀(x)) + ∑_{j=1}^K exp(v̂_j(x)) ]
       ↓
Knownness Score: S_OpenMax(x) = 1.0 - P̂(y = 0 | x) = ∑_{c=1}^K P̂(y = c | x)
```

- **Feature / Activation Representation**: Unnormalized pre-softmax logit vectors $v(x) \in \mathbb{R}^K$.
- **Class Mean Calculation (MAV)**: $\mu_c = \frac{1}{|V_c|} \sum_{x \in V_c} v(x)$, where $V_c$ is the set of correctly classified training samples for class $c$ (falling back to all class samples if $|V_c| < 5$).
- **Distance Metric**: Euclidean $\ell_2$-distance $\|v(x) - \mu_c\|_2$.
- **Weibull Distribution Fitting**: Fitted to the largest $\eta = 20$ distances (`tail` size) per class using `scipy.stats.weibull_min.fit(tail, floc=0)`.
- **$\alpha$-Weighting**: $\alpha = 5$ (or $K$ if $K < 5$).
- **Score Direction**: $S(x) = 1.0 - \hat{P}(\text{unknown}) \in [0, 1]$. Higher $\rightarrow$ more likely known.

---

## 2. Energy Score Direction and Unified Scoring Convention

Every detector in this benchmark strictly conforms to a unified scoring convention:
$$\text{Higher Score} \implies \text{More likely Known (In-Distribution)}$$
$$\text{Lower Score} \implies \text{More likely Unknown (Out-of-Distribution)}$$

| Method | Mathematical Definition | Range | Direction Rationale |
| :--- | :--- | :--- | :--- |
| **MSP** | $\max_{c \in \{1..K\}} \text{Softmax}(v(x))_c$ | $[1/K, 1.0]$ | In-distribution inputs produce sharper probability peaks. |
| **Energy** | $+T \cdot \log \sum_{c=1}^K \exp(v_c(x) / T)$ with $T=1.0$ | $(-\infty, +\infty)$ | **Free Energy formulation** (negative Helmholtz energy $-E(x)$). Known inputs produce higher log-partition values; unknowns produce lower energy. |
| **Mahalanobis** | $-\min_{c \in \{1..K\}} (z(x) - \mu_c)^T \mathbf{\Sigma}^{-1} (z(x) - \mu_c)$ | $(-\infty, 0]$ | Negative squared Mahalanobis distance in penultimate feature space. Samples near class centroids have scores close to $0$; outliers have large negative scores. |
| **OpenMax** | $1.0 - \hat{P}(y = 0 \mid x) = \sum_{c=1}^K \hat{P}(y = c \mid x)$ | $[0.0, 1.0]$ | Recalibrated probability sum over known classes. High for knowns, near $0$ for unknowns. |
| **OCSVM-deep** | $\text{decision\_function}(z(x)) = \mathbf{w}^T \phi(z(x)) - \rho$ | $(-\infty, +\infty)$ | Signed distance to RBF kernel decision boundary on penultimate features. Positive for inliers (known), negative for outliers (unknown). |
| **OCSVM-hand** | $\text{decision\_function}(f_{\text{hand}}(x)) = \mathbf{w}^T \phi(f_{\text{hand}}(x)) - \rho$ | $(-\infty, +\infty)$ | Signed distance on 32-dimensional domain-specific handcrafted electrical features (RMS, THD, peak factor, residual energy). |

---

## 3. Strict Frozen Train / Validation / Test Protocol

```
┌─────────────────────────────────────────────────────────────┐
│                    KNOWN TRAINING DATA                      │
│                  (X_train, y_train ∈ Known)                 │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                   TRAIN CLOSED-SET MODEL                    │
│             Cross-Entropy on Known Classes Only             │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                   KNOWN VALIDATION DATA                     │
│                   (X_val, y_val ∈ Known)                    │
├─────────────────────────────────────────────────────────────┤
│ 1. Model Selection: Lowest validation loss checkpoint       │
│ 2. Scorer Calibration: Fit Mahalanobis Σ, OpenMax Weibull   │
│ 3. Threshold Selection: 5th percentile of known val scores  │
│    (θ = Percentile(S_val, 5%) -> 95% validation acceptance) │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    TEST DATA EVALUATION                     │
│    (X_test_known ∈ Known)    +    (X_test_unknown ∈ Unknown)│
├─────────────────────────────────────────────────────────────┤
│ • Closed-Set Accuracy: Evaluated on X_test_known only       │
│ • Threshold-Free Metrics: AUROC, AUPR-U, FPR95, OSCR        │
│ • Operating-Point Metrics: URR & TAR (evaluated at fixed θ) │
│                                                             │
│  * CRITICAL GUARANTEE: Unknown test data is NEVER seen or   │
│    accessed during training, validation, or thresholding.   │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Open-Set Disturbance Taxonomies: Random vs Near vs Far OOD

The 16 classes comprise 1 normal sinusoid, 8 single disturbances, and 7 combined disturbances:

### Split Taxonomy Definitions:
1. **Random Splits (`random0`, `random1`, `random2`)**:
   - 4 unknown classes chosen uniformly at random without replacement (`normal` is always pinned to Known).
   - Simulates arbitrary unseen operational conditions.

2. **Near / Compositional OOD (`near`)**:
   - **Known Classes (12)**: `normal`, `sag`, `swell`, `interruption`, `harmonics`, `osc_transient`, `notch`, `spike`, `flicker`, `flicker_harm`, `flicker_sag`, `harm_osc`.
   - **Unknown Classes (4)**: `sag_harm`, `swell_harm`, `sag_osc`, `swell_osc`.
   - **Semantic Justification**: **Zero novel primitive components**. The individual constituent primitives (`sag`, `swell`, `harm`, `osc`) have all been observed in single-disturbance training classes. However, their simultaneous multi-disturbance co-occurrence was never seen. This represents a highly challenging compositional OOD regime where unknown signals closely mimic combinations of known features.

3. **Far / Novel-Component OOD (`far`)**:
   - **Known Classes (12)**: `normal`, `sag`, `swell`, `interruption`, `harmonics`, `osc_transient`, `spike`, `sag_harm`, `swell_harm`, `sag_osc`, `swell_osc`, `harm_osc`.
   - **Unknown Classes (4)**: `flicker`, `notch`, `flicker_harm`, `flicker_sag`.
   - **Novel Components**: `flicker`, `notch`.
   - **Semantic Justification**: Held-out classes contain entire physical disturbance mechanisms (low-frequency amplitude modulation for flicker, sub-cycle commutation notches) that never appeared in any training class.

---

## 5. Statistical Rigor and Multi-Comparison Corrections

- **Experimental Unit**: The statistical comparisons are conducted across matched $(\text{split} \times \text{model\_seed} \times \text{data\_seed})$ runs ($N = 45$ matched pairs per architecture), ensuring independence of experimental units.
- **Hypothesis Testing**: Two-sided paired Wilcoxon signed-rank tests against the highest-performing detector.
- **Multiple Comparison Correction**: **Holm-Bonferroni step-down procedure** adjusting raw $p$-values to control Family-Wise Error Rate (FWER) at $\alpha = 0.05$.
- **Effect Size Quantification**:
  - **Rank-Biserial Correlation**: $r_{\text{rb}} = \frac{W^+ - W^-}{W^+ + W^-} \in [-1, 1]$.
  - **Cohen's $d_z$**: $d_z = \frac{\bar{D}}{s_D}$ for paired differences.
- **Confidence Intervals**: Non-parametric 95% Bootstrap Confidence Intervals (1000 resamples).

---

## 6. Synthetic Signal Generation Methodology (IEEE 1159-Inspired)

- **Grid Standard**: 50 Hz fundamental power frequency ($f_0 = 50.0$ Hz), sampling rate $f_s = 3200$ Hz (64 samples/cycle), 10 fundamental cycles per window ($N = 640$ samples = 200 ms).
- **Per-Unit Normalization**: Nominal fundamental amplitude $A = 1.0$ pu.
- **Noise Model**: Additive White Gaussian Noise (AWGN) calibrated w.r.t nominal sine reference power $P_{\text{ref}} = 0.5\text{ pu}^2$:
  $$\sigma = \sqrt{\frac{0.5}{10^{\text{SNR}_{\text{dB}} / 10}}}$$
  Noise amplitude does not artificially vary with disturbance severity.
- **Randomization**: Independent randomized draws for amplitude levels, harmonic orders (3rd, 5th, 7th, 9th, 11th), frequency jitter ($\pm 0.1$ Hz), phase offset $\phi \in [0, 2\pi)$, transition start times ($0.5$ to $4.0$ cycles), and damping time constants ($\tau = 2\text{--}8$ ms).
- **Synthetic Disclaimer**: All evaluations are based on IEEE 1159-inspired synthetic mathematical models.

---

## 7. Scientifically Grounded Claims & Discussion

1. **Open-Set PQD Detection is Far From Solved**: While closed-set accuracy on known PQD classes reaches $81.7\%\text{--}84.5\%$, open-set detection AUROC across diverse unknown classes ranges between $0.54\text{--}0.70$ (with OpenMax achieving $0.696 \pm 0.056$ on 1D-CNN and $0.681 \pm 0.073$ on 1D-ResNet). False Positive Rates at 95% TPR remain above $75\text{--}85\%$, confirming that open-set rejection in noisy power systems is an open challenge.
2. **Compositional (Near) OOD is Harder than Novel-Component (Far) OOD**: Multi-disturbance combinations sharing known primitive components are significantly more difficult to reject than disturbances with novel spectral signatures.
3. **Detector Strengths Vary by Operating Regime**:
   - **OpenMax** achieves the highest threshold-free separation (AUROC $\approx 0.68\text{--}0.70$, AUPR-U $\approx 0.41\text{--}0.43$) by explicitly modeling class-tail distributions via Weibull calibration.
   - **Mahalanobis** and **MSP** retain competitive closed-set classification but experience degraded unknown rejection under severe noise (10 dB).
   - **Handcrafted OCSVM** provides an architecture-independent classical baseline but yields lower discriminative power compared to deep representations.
4. **Noise Robustness Across Calibrated SNRs**: All detectors exhibit significant performance degradation as test noise increases from 40 dB down to 10 dB SNR, demonstrating the necessity of evaluating open-set detectors across standardized noise regimes.

---

## 8. Literature Positioning & Collision Check

- **Prior Art**: Previous power systems literature has extensively examined closed-set PQD classification and binary novelty detection (e.g., SVDD, GMM, OCSVM on wavelet/S-transform features).
- **Novelty of this Benchmark**:
  - Standardized, reproducible class-holdout open-set benchmark for PQDs.
  - Explicit formulation and empirical comparison of **Compositional / Near-OOD** vs **Novel-Component / Far-OOD** splits in power quality.
  - Systematic multi-detector comparison (MSP, Energy, Mahalanobis, OpenMax, OCSVM) evaluated across calibrated SNR regimes under a strictly frozen protocol.
