#!/usr/bin/env python
"""Generate noise-mismatch figures comparing mixed-SNR vs single-SNR training."""
import argparse
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


def plot_mismatch_comparison(main_path, mismatch_path, out_dir):
    """Compare performance across test SNRs for Mixed vs Mismatched training."""
    if not (os.path.exists(main_path) and os.path.exists(mismatch_path)):
        print("Both main and mismatch CSVs are required.")
        return

    df_main = pd.read_csv(main_path)
    df_mismatch = pd.read_csv(mismatch_path)
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), sharey=True)

    for ax, arch in zip(axes, ["cnn", "resnet"]):
        gm_main = df_main[df_main.arch == arch]
        gm_mis = df_mismatch[df_mismatch.arch == arch]

        for m in METHODS:
            s_main = gm_main[gm_main.method == m].groupby("snr_db")["auroc"].mean()
            s_mis = gm_mis[gm_mis.method == m].groupby("snr_db")["auroc"].mean()

            # Solid line for mixed-SNR, dashed line for 30 dB mismatch
            ax.plot(s_main.index, s_main.values, marker="o", lw=1.8, color=METHOD_COLORS[m], label=f"{m} (Mixed-SNR)")
            ax.plot(s_mis.index, s_mis.values, marker="s", ls="--", lw=1.5, color=METHOD_COLORS[m], alpha=0.7, label=f"{m} (Train 30dB Only)")

        ax.set_title(f"{arch.upper()} Architecture: Matched vs Mismatched Noise", fontweight="bold")
        ax.set_xlabel("Test Noise Level SNR (dB)", fontweight="bold")
        ax.set_ylabel("AUROC (Higher = Better)", fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.axvline(30, color="gray", linestyle=":", alpha=0.6, label="Training SNR (30 dB)")

    axes[0].legend(loc="lower right", fontsize=7.5, ncol=2)
    fig.suptitle("Impact of Noise Mismatch: Training on Mixed SNRs (10-40 dB) vs Single SNR (30 dB)\n(Demonstrating Robustness Collapse at Heavy 10 dB Noise when Training Without Noise Diversity)", fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    path = os.path.join(out_dir, "01_noise_mismatch_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")

    # Plot Delta AUROC Heatmap (Mixed - Mismatched)
    # Showing gain from mixed training
    delta_data = []
    for arch in ["cnn", "resnet"]:
        gm_main = df_main[df_main.arch == arch]
        gm_mis = df_mismatch[df_mismatch.arch == arch]
        for snr in [10.0, 20.0, 30.0, 40.0]:
            row = {"Arch": arch.upper(), "SNR": f"{snr:.0f} dB"}
            for m in METHODS:
                v_main = gm_main[(gm_main.method == m) & (gm_main.snr_db == snr)]["auroc"].mean()
                v_mis = gm_mis[(gm_mis.method == m) & (gm_mis.snr_db == snr)]["auroc"].mean()
                row[m] = (v_main - v_mis)
            delta_data.append(row)

    delta_df = pd.DataFrame(delta_data)
    delta_mat = delta_df.set_index(["Arch", "SNR"])[[m for m in METHODS]]

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(delta_mat.values, cmap="coolwarm", vmin=-0.1, vmax=0.3, aspect="auto")
    ax.set_xticks(np.arange(len(delta_mat.columns)))
    ax.set_xticklabels(delta_mat.columns, fontsize=10, fontweight="bold")
    ax.set_yticks(np.arange(len(delta_mat.index)))
    ax.set_yticklabels([f"{idx[0]} @ {idx[1]}" for idx in delta_mat.index], fontsize=9.5, fontweight="bold")

    for i in range(len(delta_mat.index)):
        for j in range(len(delta_mat.columns)):
            v = delta_mat.values[i, j]
            ax.text(j, i, f"{v:+.3f}", ha="center", va="center", color="white" if abs(v) > 0.15 else "black", fontweight="bold", fontsize=9)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("$\Delta$ AUROC (Mixed-SNR - Single-SNR)", fontweight="bold")
    ax.set_title("Generalization Gain ($\Delta$ AUROC) of Mixed-SNR Training over Single 30 dB SNR Training\n(Positive Values = Mixed Training Outperforms Single-SNR Model)", fontsize=12, fontweight="bold", pad=12)
    fig.tight_layout()
    path_delta = os.path.join(out_dir, "02_mismatch_gain_delta_heatmap.png")
    fig.savefig(path_delta)
    plt.close(fig)
    print(f"Saved: {path_delta}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="results/main/raw_results.csv")
    ap.add_argument("--mismatch", default="results/mismatch/raw_results.csv")
    ap.add_argument("--out", default="figures/noise_mismatch")
    a = ap.parse_args()
    plot_mismatch_comparison(a.main, a.mismatch, a.out)
