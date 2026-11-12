#!/usr/bin/env python
"""Generate rich, publication-ready benchmark analysis figures from raw_results.csv."""
import argparse
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pqd.scoring import METHODS

plt.rcParams.update({
    "font.sans-serif": "DejaVu Sans",
    "font.family": "sans-serif",
    "figure.titlesize": 13,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300
})

METHOD_COLORS = {
    "MSP": "#1f77b4",          # Blue
    "Energy": "#ff7f0e",       # Orange
    "Mahalanobis": "#2ca02c",  # Green
    "OpenMax": "#d62728",      # Red
    "OCSVM-deep": "#9467bd",   # Purple
    "OCSVM-hand": "#8c564b"    # Brown
}


def plot_comprehensive_snr_curves(df, out_dir):
    """Multi-metric SNR robustness curves for CNN and ResNet."""
    metrics = [
        ("auroc", "AUROC", (0.4, 1.02)),
        ("fpr95", "FPR95 (Unknowns Accepted at 95% Known TPR)", (-0.02, 1.02)),
        ("closed_acc", "Closed-Set Accuracy on Known Classes", (0.5, 1.02)),
        ("oscr", "OSCR (Open-Set Classification Rate)", (0.3, 1.02)),
        ("aupr_unk", "AUPR-U (Unknown Precision-Recall)", (0.1, 1.02)),
        ("tar", "TAR (Known Acceptance Rate @ Val Thr)", (0.7, 1.02))
    ]

    for arch, ga in df.groupby("arch"):
        fig, axes = plt.subplots(2, 3, figsize=(17, 9), sharex=True)
        for ax, (m, label, ylim) in zip(axes.flat, metrics):
            for meth in METHODS:
                g = ga[ga.method == meth].groupby("snr_db")[m].agg(["mean", "std"]).dropna(subset=["mean"])
                if g.empty:
                    continue
                ax.errorbar(
                    g.index, g["mean"], yerr=g["std"].fillna(0),
                    marker="o", markersize=6, capsize=3, lw=1.8,
                    color=METHOD_COLORS.get(meth, None), label=meth
                )
            ax.set_title(label, fontweight="bold", fontsize=10)
            ax.set_ylabel(label.split()[0], fontweight="bold")
            ax.set_ylim(ylim)
            ax.grid(True, linestyle="--", alpha=0.4)
            if ax in axes[1, :]:
                ax.set_xlabel("Test Noise Level SNR (dB)", fontweight="bold")

        axes[0, 0].legend(loc="lower right", fontsize=8.5, framealpha=0.9)
        fig.suptitle(f"Open-Set Performance & Noise Robustness vs Test SNR ({arch.upper()} Architecture)\n(Mixed-SNR Training: 10/20/30/40 dB, Tested across 10 to 40 dB)", fontsize=13, fontweight="bold", y=0.98)
        fig.tight_layout(rect=[0, 0.02, 1, 0.95])
        path = os.path.join(out_dir, f"01_snr_robustness_detailed_{arch}.png")
        fig.savefig(path)
        plt.close(fig)
        print(f"Saved: {path}")


def plot_split_type_breakdowns(df, out_dir, matched_snr=30.0):
    """Bar charts comparing Random vs Near-OOD vs Far-OOD splits."""
    ref = df[df.snr_db == matched_snr]
    split_types = ["random", "near", "far"]
    split_type_names = ["Random Splits (Avg)", "Near-OOD (Sub-component Overlap)", "Far-OOD (Novel Disturbance Class)"]

    for arch, ga in ref.groupby("arch"):
        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
        metrics = [("auroc", "AUROC (Higher = Better)"), ("fpr95", "FPR95 (Lower = Better)"), ("oscr", "OSCR (Higher = Better)")]

        for ax_idx, (m, label) in enumerate(metrics):
            ax = axes[ax_idx]
            w = 0.8 / len(METHODS)
            for i, meth in enumerate(METHODS):
                vals = [ga[(ga.method == meth) & (ga.split_type == t)][m].mean() for t in split_types]
                errs = [ga[(ga.method == meth) & (ga.split_type == t)][m].std() for t in split_types]
                # Filter NaNs for methods without classification
                vals = [0.0 if np.isnan(v) else v for v in vals]
                errs = [0.0 if np.isnan(e) else e for e in errs]

                ax.bar(
                    np.arange(len(split_types)) + i * w, vals, w,
                    yerr=errs, label=meth if ax_idx == 0 else "",
                    color=METHOD_COLORS.get(meth, None), capsize=3, edgecolor="black", lw=0.5
                )

            ax.set_xticks(np.arange(len(split_types)) + 0.4 - w / 2)
            ax.set_xticklabels(split_type_names, fontsize=8.5, fontweight="bold")
            ax.set_title(label, fontweight="bold", fontsize=11)
            ax.grid(True, linestyle="--", alpha=0.3, axis="y")
            if m == "auroc":
                ax.axhline(0.5, color="gray", linestyle="--", lw=1.0, alpha=0.7)
                ax.set_ylim(0.4, 1.05)
            elif m == "fpr95":
                ax.set_ylim(0.0, 1.05)
            elif m == "oscr":
                ax.set_ylim(0.0, 1.05)

        axes[0].legend(loc="upper right", fontsize=8.5, ncol=2)
        fig.suptitle(f"Open-Set Detection Hardness by Split Regime ({arch.upper()} @ {matched_snr:.0f} dB SNR)\nNear-OOD vs Far-OOD vs Random Class Partitions", fontsize=13, fontweight="bold", y=0.98)
        fig.tight_layout(rect=[0, 0.02, 1, 0.94])
        path = os.path.join(out_dir, f"02_split_type_hardness_{arch}.png")
        fig.savefig(path)
        plt.close(fig)
        print(f"Saved: {path}")


def plot_detector_boxplots(df, out_dir):
    """Box plots showing distribution and variance of AUROC across all runs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    
    for ax, (arch, ga) in zip([ax1, ax2], df.groupby("arch")):
        data_to_plot = []
        labels = []
        palette = []
        for m in METHODS:
            vals = ga[ga.method == m]["auroc"].dropna().values
            if len(vals) > 0:
                data_to_plot.append(vals)
                labels.append(m)
                palette.append(METHOD_COLORS[m])

        bplot = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, medianprops=dict(color="black", lw=1.5))
        for patch, color in zip(bplot['boxes'], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Overlay jittered data points
        for i, vals in enumerate(data_to_plot):
            x = np.random.normal(i + 1, 0.04, size=len(vals))
            ax.plot(x, vals, 'k.', alpha=0.4, markersize=5)

        ax.set_title(f"AUROC Distribution: {arch.upper()}", fontweight="bold")
        ax.set_ylabel("AUROC", fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.axhline(0.5, color="red", linestyle="--", lw=0.8, label="Random (0.50)")
        ax.set_ylim(0.35, 1.02)

    fig.suptitle("Overall AUROC Variance & Stability Across All Splits, Seeds, and Noise Conditions", fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    path = os.path.join(out_dir, "03_auroc_distribution_boxplots.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/main")
    ap.add_argument("--matched_snr", type=float, default=30.0)
    a = ap.parse_args()

    csv_path = os.path.join(a.results, "raw_results.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} does not exist.")
        return

    df = pd.read_csv(csv_path)
    fig_dir = os.path.join(a.results, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print(f"Generating comprehensive benchmark figures from {csv_path}...")
    plot_comprehensive_snr_curves(df, fig_dir)
    plot_split_type_breakdowns(df, fig_dir, a.matched_snr)
    plot_detector_boxplots(df, fig_dir)
    print(f"All benchmark figures generated successfully in {fig_dir}!")


if __name__ == "__main__":
    main()
