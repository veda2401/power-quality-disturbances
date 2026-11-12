#!/usr/bin/env bash
# Reproducible Pipeline for Authoritative Final Open-Set PQD Benchmark.
# Finished runs are automatically cached in results/main/raw_results.csv.
set -e

# 1. Sanity checks (seconds)
python tests/test_smoke.py

# 2. Dataset / split parameter reproducibility tables -> docs/
python export_tables.py

# 3. Authoritative Benchmark Protocol:
#    - 90 model runs (3 synthetic data seeds x 3 model seeds x 5 splits x 2 deep architectures)
#    - 120 train / 30 val / 80 test samples per class, 10 epochs
#    - Mixed-SNR training (10/20/30/40 dB), evaluated across test SNRs (10/20/30/40 dB)
python run_experiments.py \
    --archs cnn resnet \
    --seeds 0 1 2 \
    --data_seeds 0 1 2 \
    --n_random_splits 3 \
    --n_train 120 \
    --n_val 30 \
    --n_test 80 \
    --epochs 10 \
    --out results/main

# 4. Statistical Analysis & Paper-Ready Tables (Markdown, LaTeX, CSV, Holm-adjusted Wilcoxon)
python analyze_results.py --results results/main --matched_snr 30

# 5. Benchmark Figures & Signal Visualizations
python generate_comprehensive_benchmark_figures.py --results results/main --matched_snr 30
python generate_signal_graphs.py

