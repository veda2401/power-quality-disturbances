# Open-Set Evaluation Benchmark for Power Quality Disturbance (PQD) Classification

> **Research Question:** Can existing power-quality-disturbance (PQD) deep learning classifiers
> reliably recognize *and reject* disturbance types that were completely absent
> from training?

The contribution is a **standardized open-set evaluation protocol, comprehensive detector comparison, and reproducible benchmark** for power systems. The dataset is generated from mathematical signal definitions in `pqd/generator.py` (IEEE 1159-inspired), requiring no external data downloads.

---

## Quick Start

```bash
pip install -r requirements.txt
python tests/test_smoke.py                 # Sanity checks (seconds)
./run_all.sh                               # Authoritative final benchmark pipeline
```

Individual steps:

```bash
python export_tables.py                                    # Dataset and split tables -> docs/
python run_experiments.py --out results/main               # Main benchmark (90 runs)
python analyze_results.py --results results/main           # Paper tables (.md/.tex/.csv) + statistics
python generate_comprehensive_benchmark_figures.py        # Benchmark figures
python generate_signal_graphs.py                          # Waveform and spectrum gallery
```

---

## Authoritative Benchmark Protocol

### 1. Synthetic Signal Model (IEEE 1159-Inspired)
- **Grid Standard:** Indian 50 Hz power system ($f_0 = 50.0$ Hz), sampling rate $f_s = 3200$ Hz (64 samples/cycle).
- **Window Length:** 10 fundamental cycles $\rightarrow N = 640$ samples per window (200 ms duration).
- **Amplitude Normalization:** Per-unit (pu) with nominal fundamental $A = 1.0$ pu.
- **16 Classes:** 1 normal sinusoid, 8 single disturbances (`sag`, `swell`, `interruption`, `harmonics`, `osc_transient`, `notch`, `spike`, `flicker`), and 7 combined disturbances (`sag_harm`, `swell_harm`, `flicker_harm`, `sag_osc`, `swell_osc`, `flicker_sag`, `harm_osc`).
- **Noise Model:** Additive White Gaussian Noise (AWGN); calibrated against nominal sine power $P_{\text{ref}} = 0.5\text{ pu}^2$ across SNRs: 10 dB, 20 dB, 30 dB, 40 dB.
- **Dataset Generation:** Evaluated across **3 independent synthetic dataset seeds** (`data_seeds = [0, 1, 2]`).

### 2. Class-Holdout Splits (12 Known / 4 Unknown)
`normal` is always pinned to Known. Each split holds out 4 disturbance classes:
- **Random Splits (`random0`, `random1`, `random2`):** Uniform random selection of 4 unknown classes.
- **Near-OOD (`near` - Compositional):** Unknowns are multi-disturbance combinations (`sag_harm`, `swell_harm`, `sag_osc`, `swell_osc`) whose individual constituent primitives (`sag`, `swell`, `harm`, `osc`) were present in training.
- **Far-OOD (`far` - Novel-Component):** Unknowns contain disturbance families (`flicker`, `notch`) completely absent from training.

### 3. Deep Learning Architectures & Training
- **Architectures:** 1D-CNN and 1D-ResNet (32-channel feature representations).
- **Training Setup:** Closed-set cross-entropy on known classes only (120 train / 30 val / 80 test samples per class, 10 epochs, batch size 128, Adam $\text{lr}=10^{-3}$, weight decay $10^{-4}$).
- **Noise Protocol:** Mixed-SNR training (10/20/30/40 dB), evaluated across test SNRs (10, 20, 30, 40 dB).
- **Execution Matrix:** **90 total model runs** ($3\text{ data seeds} \times 3\text{ model seeds } [0,1,2] \times 5\text{ splits} \times 2\text{ architectures}$).

### 4. Unknown-Detection Methods & Scoring Convention
All detectors conform to a unified scoring convention: **Higher Score $\implies$ More Likely Known (In-Distribution)**; **Lower Score $\implies$ More Likely Unknown (Out-of-Distribution)**.

| Method | Specification |
| :--- | :--- |
| **MSP** | Maximum Softmax Probability baseline ($\max_c \text{Softmax}(\mathbf{v})_c$). |
| **Energy** | Free Energy scoring ($+T \cdot \log \sum \exp(v_c / T)$, $T = 1.0$). |
| **Mahalanobis** | Class-conditional Gaussians with shared covariance on penultimate features ($-\min_c (\mathbf{z}-\boldsymbol{\mu}_c)^T \mathbf{\Sigma}^{-1} (\mathbf{z}-\boldsymbol{\mu}_c)$). |
| **OpenMax** | Euclidean Logit-Space Variant: Weibull tail fitting ($\eta = 20$, $\alpha = 5$) on correctly classified training logits; $S = 1.0 - \hat{P}(\text{unknown})$. |
| **OCSVM-deep** | One-Class SVM with RBF kernel on standardized penultimate deep features ($\nu = 0.05$). |
| **OCSVM-hand** | One-Class SVM with RBF kernel on 32 handcrafted electrical features (RMS, THD, peak factor, residual energy; $\nu = 0.05$). |

### 5. Frozen Thresholding & Zero-Leakage Guarantee
- **Threshold Calibration:** Fixed at the 5th percentile of **known validation scores** ($95\%$ known validation acceptance).
- **Data Leakage Guarantee:** Unknown test samples are **never accessed or used** during training, model selection, or threshold calibration.

### 6. Headline Metrics & Statistical Rigor
- **Threshold-Free Metrics:** AUROC, AUPR-U (unknown = positive class), FPR95, OSCR (Open-Set Classification Rate).
- **Operating-Point Metrics:** URR (Unknown Rejection Rate) and TAR (Known Acceptance Rate) at the fixed validation threshold.
- **Statistical Significance:** Paired two-sided Wilcoxon signed-rank tests across $N = 45$ matched runs per architecture with **Holm-Bonferroni step-down correction**, paired effect sizes (Cohen's $d_z$, rank-biserial correlation $r_{\text{rb}}$), and 95% bootstrap confidence intervals.

---

## Authoritative Results Summary (@ 30 dB SNR, Mean $\pm$ Std across 45 runs)

| Architecture | Method | Closed Acc. | AUROC | AUPR-U | FPR95 | OSCR | URR | TAR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1D-CNN** | **OpenMax** | 81.9% $\pm$ 1.6% | **0.696 $\pm$ 0.056** | **0.428 $\pm$ 0.105** | **0.822 $\pm$ 0.164** | **0.588 $\pm$ 0.045** | **0.247** | 0.928 |
| | **Mahalanobis** | **84.5% $\pm$ 1.5%** | 0.655 $\pm$ 0.113 | 0.359 $\pm$ 0.095 | 0.904 $\pm$ 0.048 | 0.547 $\pm$ 0.092 | 0.072 | 0.962 |
| | **OCSVM-deep** | -- | 0.596 $\pm$ 0.154 | 0.409 $\pm$ 0.187 | 0.779 $\pm$ 0.184 | -- | 0.219 | 0.949 |
| | **MSP** | 84.5% $\pm$ 1.5% | 0.595 $\pm$ 0.128 | 0.359 $\pm$ 0.138 | 0.853 $\pm$ 0.123 | 0.543 $\pm$ 0.106 | 0.027 | 0.995 |
| | **Energy** | 84.5% $\pm$ 1.5% | 0.568 $\pm$ 0.154 | 0.343 $\pm$ 0.158 | 0.864 $\pm$ 0.142 | 0.503 $\pm$ 0.128 | 0.027 | 0.992 |
| | **OCSVM-hand** | -- | 0.536 $\pm$ 0.093 | 0.394 $\pm$ 0.082 | 0.791 $\pm$ 0.076 | -- | 0.181 | 0.967 |
| **1D-ResNet**| **OpenMax** | 79.8% $\pm$ 2.1% | **0.681 $\pm$ 0.073** | **0.411 $\pm$ 0.083** | **0.832 $\pm$ 0.110** | **0.554 $\pm$ 0.055** | **0.217** | 0.936 |
| | **Mahalanobis** | **81.7% $\pm$ 1.9%** | 0.636 $\pm$ 0.107 | 0.321 $\pm$ 0.075 | 0.952 $\pm$ 0.032 | 0.517 $\pm$ 0.088 | 0.022 | 0.968 |
| | **MSP** | 81.7% $\pm$ 1.9% | 0.619 $\pm$ 0.125 | 0.359 $\pm$ 0.109 | 0.873 $\pm$ 0.086 | 0.555 $\pm$ 0.099 | 0.010 | 0.997 |
| | **Energy** | 81.7% $\pm$ 1.9% | 0.566 $\pm$ 0.167 | 0.335 $\pm$ 0.126 | 0.882 $\pm$ 0.096 | 0.492 $\pm$ 0.131 | 0.035 | 0.986 |
| | **OCSVM-deep** | -- | 0.553 $\pm$ 0.117 | 0.343 $\pm$ 0.129 | 0.853 $\pm$ 0.121 | -- | 0.135 | 0.955 |
| | **OCSVM-hand** | -- | 0.536 $\pm$ 0.093 | 0.394 $\pm$ 0.082 | 0.791 $\pm$ 0.076 | -- | 0.181 | 0.967 |

---

## Key Limitations & Scientific Claims

- **Synthetic Signals Only:** All evaluations utilize IEEE 1159-inspired mathematical simulation models; validating on recorded real-world substation waveforms is planned future work.
- **Open-Set PQD Detection is Non-Trivial:** While closed-set accuracy exceeds $84\%$, open-set AUROC ranges between $0.65\text{--}0.70$, showing that novel PQD rejection under noise remains an open research challenge.
- **Compositional Hardness:** Unknowns consisting of combinations of known primitive components (Near-OOD) are significantly harder to detect than entirely novel disturbance families (Far-OOD).

