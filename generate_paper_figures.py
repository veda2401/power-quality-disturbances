#!/usr/bin/env python
"""Generate publication-ready academic figures for the PQD Open-Set Recognition Benchmark.

Clean, minimal aesthetic with:
- NO background grid lines (pristine white background).
- NO black error bar lines on top of bar plots.
- Clean line and scatter markers without error bar whiskers.
- Standardized, colorblind-friendly global detector palette.
- Strict adherence to results/main/raw_results.csv.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.manifold import TSNE

from pqd.data import build_dataset, noisy, noisy_train, select_classes
from pqd.generator import CLASS_NAMES, FS, F0, N, CYCLES, generate_one
from pqd.scoring import METHODS, Mahalanobis, OCSVM, OpenMax, score_energy, score_msp
from pqd.splits import make_splits, split_names
from pqd.train import extract, train_model

# -----------------------------------------------------------------------------
# Global Styling & Constants
# -----------------------------------------------------------------------------
PAPER_FIG_DIR = "results/main/figures/paper"
os.makedirs(PAPER_FIG_DIR, exist_ok=True)

# Publication typography & minimal styling (NO GRIDS)
plt.rcParams.update({
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"],
    "font.family": "sans-serif",
    "figure.titlesize": 10.0,
    "axes.titlesize": 9.0,
    "axes.labelsize": 8.0,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.2,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "lines.linewidth": 1.3,
    "lines.markersize": 4.5,
    "axes.linewidth": 0.75,
    "axes.grid": False,          # NO background grid lines
    "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Standardized Global Detector Palette
DETECTOR_COLORS = {
    "MSP": "#1f77b4",          # Classic Steel Blue
    "Energy": "#ff7f0e",       # Deep Amber / Ochre
    "Mahalanobis": "#2ca02c",  # Forest Green
    "OpenMax": "#d62728",      # Crimson Red
    "OCSVM-deep": "#9467bd",   # Slate Purple
    "OCSVM-hand": "#8c564b",   # Charcoal Brown
}


def save_dual_figure(fig, name):
    """Save figure as both vector PDF and 300 DPI PNG with tight bounding box."""
    pdf_path = os.path.join(PAPER_FIG_DIR, f"{name}.pdf")
    png_path = os.path.join(PAPER_FIG_DIR, f"{name}.png")
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {pdf_path} and {png_path}")


# -----------------------------------------------------------------------------
# 1. Open-Set Detector Comparison @ 30 dB SNR (Clean Bars, No Top Lines, No Grid)
# -----------------------------------------------------------------------------
def plot_01_detector_comparison(df):
    """Clean grouped bar chart comparing all 6 detectors at 30 dB SNR across core metrics."""
    ref = df[df.snr_db == 30.0]
    metrics = [
        ("auroc", "AUROC", (0.0, 0.85)),
        ("aupr_unk", "AUPR (Unknown)", (0.0, 0.55)),
        ("fpr95", "FPR95 (at 95% TPR)", (0.0, 1.05)),
        ("oscr", "OSCR", (0.0, 0.75)),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.3), layout="constrained")
    archs = ["cnn", "resnet"]
    arch_names = ["1D-CNN", "1D-ResNet"]
    x = np.arange(len(archs))
    w = 0.125
    offsets = np.linspace(-2.5 * w, 2.5 * w, len(METHODS))

    for ax_idx, (m_col, m_title, y_lim) in enumerate(metrics):
        ax = axes[ax_idx]
        for m_idx, meth in enumerate(METHODS):
            means = []
            for a in archs:
                sub = ref[(ref.arch == a) & (ref.method == meth)][m_col].dropna()
                means.append(sub.mean() if len(sub) > 0 else np.nan)

            if not np.all(np.isnan(means)):
                # Clean bar with NO error bars / black lines on top
                ax.bar(
                    x + offsets[m_idx], means, w,
                    label=meth if ax_idx == 0 else "",
                    color=DETECTOR_COLORS[meth],
                    edgecolor="none"
                )

        ax.set_xticks(x)
        ax.set_xticklabels(arch_names, fontweight="medium")
        ax.set_title(m_title, fontsize=8.5, pad=3)
        ax.set_ylim(y_lim)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.15),
               ncol=6, frameon=False, fontsize=7.2, columnspacing=0.8)

    save_dual_figure(fig, "01_detector_comparison")


# -----------------------------------------------------------------------------
# 2. SNR Robustness Curves (Clean Lines & Markers, No Error Bars, No Grid)
# -----------------------------------------------------------------------------
def plot_02_snr_robustness(df):
    """Multi-metric line plot across test SNRs (10, 20, 30, 40 dB) comparing detectors."""
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.3), layout="constrained")
    metrics = [
        ("auroc", "AUROC", (0.45, 0.80)),
        ("fpr95", "FPR95", (0.70, 1.00)),
        ("oscr", "OSCR", (0.30, 0.75)),
        ("closed_acc", "Closed-Set Acc.", (0.50, 0.95)),
    ]

    for ax_idx, (m_col, m_title, y_lim) in enumerate(metrics):
        ax = axes[ax_idx]
        for meth in METHODS:
            sub = df[(df.arch == "cnn") & (df.method == meth)].groupby("snr_db")[m_col].mean().dropna()
            if sub.empty:
                continue
            ax.plot(
                sub.index, sub.values,
                label=meth if ax_idx == 0 else "",
                color=DETECTOR_COLORS[meth],
                marker="o", markersize=4.0, linewidth=1.3
            )

        ax.set_xlabel("Test SNR (dB)")
        ax.set_title(m_title, fontsize=8.5, pad=3)
        ax.set_xticks([10, 20, 30, 40])
        ax.set_xlim(8.0, 42.0)
        ax.set_ylim(y_lim)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.15),
               ncol=6, frameon=False, fontsize=7.2, columnspacing=0.8)

    save_dual_figure(fig, "02_snr_robustness")


# -----------------------------------------------------------------------------
# 3. Split Regime Hardness (Clean Grouped Bars, No Top Error Bars, No Grid)
# -----------------------------------------------------------------------------
def plot_03_split_regime_hardness(df):
    """Clean bar chart comparing Random vs Near-OOD vs Far-OOD splits at 30 dB SNR."""
    ref = df[df.snr_db == 30.0]
    split_types = ["random", "near", "far"]
    split_labels = ["Random OOD", "Near-OOD\n(Compositional)", "Far-OOD\n(Novel Class)"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.4), layout="constrained")
    x = np.arange(len(split_types))
    w = 0.125
    offsets = np.linspace(-2.5 * w, 2.5 * w, len(METHODS))

    for ax, arch, title in [(ax1, "cnn", "1D-CNN (AUROC)"), (ax2, "resnet", "1D-ResNet (AUROC)")]:
        ga = ref[ref.arch == arch]
        for m_idx, meth in enumerate(METHODS):
            means = [ga[(ga.method == meth) & (ga.split_type == t)]["auroc"].mean() for t in split_types]
            ax.bar(
                x + offsets[m_idx], means, w,
                label=meth if arch == "cnn" else "",
                color=DETECTOR_COLORS[meth],
                edgecolor="none"
            )

        ax.set_xticks(x)
        ax.set_xticklabels(split_labels, fontsize=7.5)
        ax.set_title(title, fontsize=8.5, pad=3)
        ax.set_ylim(0.0, 0.90)
        ax.set_ylabel("AUROC")

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.14),
               ncol=6, frameon=False, fontsize=7.2, columnspacing=0.8)

    save_dual_figure(fig, "03_random_near_far_split_hardness")


# -----------------------------------------------------------------------------
# 4. Architecture Comparison: 1D-CNN vs 1D-ResNet (Clean Bars, No Grid)
# -----------------------------------------------------------------------------
def plot_04_architecture_comparison(df):
    """Clean paired comparison showing CNN vs ResNet consistency across detectors."""
    ref = df[df.snr_db == 30.0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.3), layout="constrained")

    x = np.arange(len(METHODS))
    w = 0.35

    cnn_means = [ref[(ref.arch == "cnn") & (ref.method == m)]["auroc"].mean() for m in METHODS]
    res_means = [ref[(ref.arch == "resnet") & (ref.method == m)]["auroc"].mean() for m in METHODS]

    ax1.bar(x - w/2, cnn_means, w, label="1D-CNN", color="#1f77b4", edgecolor="none")
    ax1.bar(x + w/2, res_means, w, label="1D-ResNet", color="#e6550d", edgecolor="none")
    ax1.set_xticks(x)
    ax1.set_xticklabels(METHODS, rotation=15, ha="right", fontsize=7.5)
    ax1.set_ylabel("AUROC")
    ax1.set_ylim(0.0, 0.80)
    ax1.set_title("(a) Open-Set AUROC (@ 30 dB SNR)", fontsize=8.5)
    ax1.legend(loc="upper right", fontsize=7.2, frameon=False)

    cls_methods = ["MSP", "Energy", "Mahalanobis", "OpenMax"]
    x_cls = np.arange(len(cls_methods))
    cnn_acc = [ref[(ref.arch == "cnn") & (ref.method == m)]["closed_acc"].mean() for m in cls_methods]
    res_acc = [ref[(ref.arch == "resnet") & (ref.method == m)]["closed_acc"].mean() for m in cls_methods]

    ax2.bar(x_cls - w/2, cnn_acc, w, label="1D-CNN", color="#1f77b4", edgecolor="none")
    ax2.bar(x_cls + w/2, res_acc, w, label="1D-ResNet", color="#e6550d", edgecolor="none")
    ax2.set_xticks(x_cls)
    ax2.set_xticklabels(cls_methods, fontsize=7.5)
    ax2.set_ylabel("Closed-Set Accuracy")
    ax2.set_ylim(0.0, 0.95)
    ax2.set_title("(b) Known Closed-Set Accuracy (@ 30 dB SNR)", fontsize=8.5)
    ax2.legend(loc="upper right", fontsize=7.2, frameon=False)

    save_dual_figure(fig, "04_architecture_comparison")


# -----------------------------------------------------------------------------
# 5. Closed-Set Accuracy vs Open-Set AUROC (Clean Scatter, No Grid)
# -----------------------------------------------------------------------------
def plot_05_closed_vs_open_set(df):
    """Clean scatter plot showing closed-set accuracy vs open-set AUROC."""
    ref = df[df.snr_db == 30.0]
    fig, ax = plt.subplots(figsize=(3.4, 2.5), layout="constrained")

    cls_methods = ["MSP", "Energy", "Mahalanobis", "OpenMax"]
    for m in cls_methods:
        sub_cnn = ref[(ref.arch == "cnn") & (ref.method == m)]
        sub_res = ref[(ref.arch == "resnet") & (ref.method == m)]

        ax.scatter(
            sub_cnn["closed_acc"].mean(), sub_cnn["auroc"].mean(),
            color=DETECTOR_COLORS[m], marker="o", s=55, label=f"{m} (CNN)", zorder=5
        )
        ax.scatter(
            sub_res["closed_acc"].mean(), sub_res["auroc"].mean(),
            color=DETECTOR_COLORS[m], marker="s", s=50, facecolors="none",
            edgecolors=DETECTOR_COLORS[m], linewidths=1.5, label=f"{m} (ResNet)", zorder=5
        )

    ax.set_xlabel("Closed-Set Accuracy (Known Classes)")
    ax.set_ylabel("Open-Set Detection AUROC")
    ax.set_xlim(0.78, 0.86)
    ax.set_ylim(0.55, 0.73)
    ax.legend(bbox_to_anchor=(1.03, 1.0), loc="upper left", fontsize=6.8, frameon=False)

    save_dual_figure(fig, "05_closed_vs_open_set")


# -----------------------------------------------------------------------------
# 6. AUROC Distribution Boxplots (Clean Boxplots, No Grid)
# -----------------------------------------------------------------------------
def plot_06_auroc_distribution_boxplots(df):
    """Clean publication boxplots of AUROC variance across all 90 runs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.4), sharey=True, layout="constrained")

    for ax, arch, title in [(ax1, "cnn", "1D-CNN AUROC Distribution"), (ax2, "resnet", "1D-ResNet AUROC Distribution")]:
        ga = df[df.arch == arch]
        data = [ga[ga.method == m]["auroc"].dropna().values for m in METHODS]

        bplot = ax.boxplot(
            data, tick_labels=METHODS, patch_artist=True,
            medianprops=dict(color="black", linewidth=1.2),
            whiskerprops=dict(linewidth=0.8, color="black"),
            capprops=dict(linewidth=0.8, color="black"),
            flierprops=dict(marker=".", markersize=2.5, alpha=0.3)
        )

        for patch, m in zip(bplot["boxes"], METHODS):
            patch.set_facecolor(DETECTOR_COLORS[m])
            patch.set_alpha(0.75)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.6)

        ax.set_title(title, fontsize=8.5)
        ax.set_ylabel("AUROC" if arch == "cnn" else "")
        ax.set_ylim(0.35, 0.90)
        ax.tick_params(axis="x", rotation=15)

    save_dual_figure(fig, "06_auroc_distribution_boxplots")


# -----------------------------------------------------------------------------
# 7. OSCR and ROC Curves (Clean Curves, No Grid)
# -----------------------------------------------------------------------------
def plot_07_oscr_and_roc_curves(rep_data):
    """Side-by-side clean ROC and OSCR curves for representative 1D-CNN."""
    fig, (ax_roc, ax_oscr) = plt.subplots(1, 2, figsize=(6.8, 2.4), layout="constrained")
    sk, su = rep_data["sk"], rep_data["su"]
    yk = rep_data["yk"]

    # (a) ROC Curves
    for m in METHODS:
        s_k = sk[m][0]
        s_u = su[m][0]
        y_true = np.concatenate([np.ones(len(s_k)), np.zeros(len(s_u))])
        scores = np.concatenate([s_k, s_u])

        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, label=f"{m} ({roc_auc:.3f})", color=DETECTOR_COLORS[m], linewidth=1.3)

    ax_roc.set_xlabel("False Positive Rate (FPR)")
    ax_roc.set_ylabel("True Positive Rate (TPR)")
    ax_roc.set_title("(a) Receiver Operating Characteristic (ROC)", fontsize=8.5)
    ax_roc.legend(loc="lower right", fontsize=6.5, frameon=False)

    # (b) OSCR Curves
    for m in METHODS:
        s_k, pred_k = sk[m]
        if pred_k is None:
            continue
        correct_k = (pred_k == yk).astype(float)
        s_u = su[m][0]

        nk, nu = len(s_k), len(s_u)
        order = np.argsort(-s_k)
        neg_sk = -s_k[order]
        ccum = np.concatenate([[0.0], np.cumsum(correct_k[order])])
        neg_su = np.sort(-s_u)

        thr = np.unique(np.concatenate([s_k, s_u]))[::-1]
        n_k = np.searchsorted(neg_sk, -thr, side="right")
        n_u = np.searchsorted(neg_su, -thr, side="right")
        ccr = np.concatenate([[0.0], ccum[n_k] / nk])
        fpr = np.concatenate([[0.0], n_u / nu])

        _trapz = getattr(np, "trapezoid", None) or np.trapz
        area = float(_trapz(ccr, fpr))
        ax_oscr.plot(fpr, ccr, label=f"{m} ({area:.3f})", color=DETECTOR_COLORS[m], linewidth=1.3)

    ax_oscr.set_xlabel("FPR on Unknowns")
    ax_oscr.set_ylabel("CCR on Knowns")
    ax_oscr.set_title("(b) Open-Set Classification Rate (OSCR)", fontsize=8.5)
    ax_oscr.legend(loc="lower right", fontsize=6.5, frameon=False)

    save_dual_figure(fig, "07_oscr_and_roc_curves")


# -----------------------------------------------------------------------------
# 8. Score Distributions KDE (Clean Distributions, No Grid)
# -----------------------------------------------------------------------------
def plot_08_score_distributions_kde(rep_data):
    """2x3 grid of clean score distributions for all 6 detectors."""
    fig, axes = plt.subplots(2, 3, figsize=(6.8, 3.2), layout="constrained")
    sk, su = rep_data["sk"], rep_data["su"]

    for idx, (m, ax) in enumerate(zip(METHODS, axes.flat)):
        s_k = sk[m][0]
        s_u = su[m][0]

        s_min, s_max = min(s_k.min(), s_u.min()), max(s_k.max(), s_u.max())
        if s_max - s_min < 1e-8:
            s_max = s_min + 1.0
        x_eval = np.linspace(s_min, s_max, 200)

        kde_k = stats.gaussian_kde(s_k)(x_eval)
        kde_u = stats.gaussian_kde(s_u)(x_eval)

        ax.plot(x_eval, kde_k, label="Known", color="#2ca02c", linewidth=1.2)
        ax.fill_between(x_eval, kde_k, color="#2ca02c", alpha=0.22)
        ax.plot(x_eval, kde_u, label="Unknown", color="#d62728", linewidth=1.2)
        ax.fill_between(x_eval, kde_u, color="#d62728", alpha=0.22)

        thr = np.percentile(s_k, 5)
        ax.axvline(thr, color="black", linestyle="--", linewidth=0.85, label=r"$\theta_{95}$")

        ax.set_title(m, fontsize=8.0, pad=2)
        ax.tick_params(labelsize=6.5)
        if idx == 0:
            ax.legend(fontsize=6.2, loc="upper left", frameon=False)

    save_dual_figure(fig, "08_score_distributions_kde")


# -----------------------------------------------------------------------------
# 9. Penultimate Feature Latent Space (Clean t-SNE, No Grid)
# -----------------------------------------------------------------------------
def plot_09_latent_feature_tsne(rep_data):
    """Clean 2D t-SNE projection of penultimate deep features."""
    fe_k = rep_data["fe_k"]
    fe_u = rep_data["fe_u"]
    yk = rep_data["yk"]
    known_indices = rep_data["known"]

    np.random.seed(42)
    idx_k = np.random.choice(len(fe_k), size=min(400, len(fe_k)), replace=False)
    idx_u = np.random.choice(len(fe_u), size=min(150, len(fe_u)), replace=False)

    X_all = np.vstack([fe_k[idx_k], fe_u[idx_u]])
    labels_all = np.concatenate([yk[idx_k], np.full(len(idx_u), -1)])

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X_all)

    fig, ax = plt.subplots(figsize=(3.4, 2.8), layout="constrained")
    cmap = matplotlib.colormaps.get_cmap("tab20")

    for k_local, k_orig in enumerate(known_indices):
        mask = (labels_all == k_local)
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], s=10, alpha=0.6, color=cmap(k_local))

    mask_u = (labels_all == -1)
    ax.scatter(X_tsne[mask_u, 0], X_tsne[mask_u, 1], s=25, marker="x", color="#d62728", lw=1.1, label="Unknown (Held-Out)", zorder=10)

    ax.scatter([], [], s=10, color="gray", label="Known (12 Classes)")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.set_title("Penultimate Deep Feature Space (t-SNE)", fontsize=8.5)
    ax.legend(loc="lower right", fontsize=6.8, frameon=False)

    save_dual_figure(fig, "09_latent_feature_tsne")


# -----------------------------------------------------------------------------
# 10. Detector Ranking Heatmap (Clean Heatmap, No Grid)
# -----------------------------------------------------------------------------
def plot_10_detector_ranking_heatmap(df):
    """Perceptually uniform ranking heatmap across all conditions."""
    rk = {}
    for snr in sorted(df.snr_db.unique()):
        for arch in ["cnn", "resnet"]:
            ga = df[(df.snr_db == snr) & (df.arch == arch)]
            p = ga.pivot_table(index=["split", "seed"], columns="method", values="auroc")
            rk[f"{arch.upper()} ({snr:.0f} dB)"] = p.rank(axis=1, ascending=False).mean()
    rank_df = pd.DataFrame(rk).T
    rank_df = rank_df[[m for m in METHODS if m in rank_df.columns]]

    fig, ax = plt.subplots(figsize=(4.8, 2.6), layout="constrained")
    im = ax.imshow(rank_df.values, cmap="YlGn_r", aspect="auto", vmin=1.0, vmax=6.0)

    ax.set_xticks(np.arange(len(rank_df.columns)))
    ax.set_xticklabels(rank_df.columns, rotation=18, ha="right", fontsize=7.5)
    ax.set_yticks(np.arange(len(rank_df.index)))
    ax.set_yticklabels(rank_df.index, fontsize=7.5)

    for i in range(len(rank_df.index)):
        for j in range(len(rank_df.columns)):
            v = rank_df.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="black" if v > 3.2 else "white", fontsize=7.2, fontweight="medium")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Average Rank (1 = Best)", fontsize=7.5)
    cbar.ax.tick_params(labelsize=7.0)
    ax.set_title("Average AUROC Rank Matrix across Noise & Architectures", fontsize=8.5)

    save_dual_figure(fig, "10_detector_ranking_heatmap")


# -----------------------------------------------------------------------------
# 11. Representative PQD Waveforms (Clean Grid, No Gridlines)
# -----------------------------------------------------------------------------
def plot_11_pqd_waveforms():
    """Compact 4x4 grid of all 16 IEEE 1159-inspired PQD disturbance classes without background lines."""
    fig, axes = plt.subplots(4, 4, figsize=(6.8, 4.4), sharex=True, sharey=True, layout="constrained")
    t_ms = (np.arange(N) / FS) * 1000
    rng = np.random.default_rng(42)

    labels = [
        "Normal", "Sag", "Swell", "Interruption",
        "Harmonics", "Osc. Transient", "Notch", "Spike",
        "Flicker", "Sag+Harm", "Swell+Harm", "Flicker+Harm",
        "Sag+Osc", "Swell+Osc", "Flicker+Sag", "Harm+Osc"
    ]

    for idx, (cls_name, ax, label) in enumerate(zip(CLASS_NAMES, axes.flat, labels)):
        x = generate_one(cls_name, rng)
        ax.plot(t_ms, x, color="#1f77b4", linewidth=0.9)
        ax.set_title(f"({chr(97+idx)}) {label}", fontsize=7.5, pad=2)
        ax.set_ylim(-2.2, 2.2)
        if idx % 4 == 0:
            ax.set_ylabel("pu", fontsize=7.0)
        if idx >= 12:
            ax.set_xlabel("Time (ms)", fontsize=7.0)
        ax.tick_params(labelsize=6.5)

    save_dual_figure(fig, "11_pqd_waveforms_sample")


# -----------------------------------------------------------------------------
# 12. Split Taxonomy Matrix (Clean Matrix, No Grid)
# -----------------------------------------------------------------------------
def plot_12_split_taxonomy_matrix():
    """Visual matrix depicting Known (12) vs Unknown (4) classes for each split regime."""
    splits = make_splits(3, 4)
    split_names_list = ["Random-0", "Random-1", "Random-2", "Near-OOD (Comp.)", "Far-OOD (Novel)"]

    mat = np.zeros((len(CLASS_NAMES), len(splits)))
    for s_idx, sp in enumerate(splits):
        for k in sp["known"]:
            mat[k, s_idx] = 1.0
        for u in sp["unknown"]:
            mat[u, s_idx] = 0.0

    fig, ax = plt.subplots(figsize=(4.0, 3.8), layout="constrained")
    cmap = matplotlib.colors.ListedColormap(["#d62728", "#31a354"])
    im = ax.imshow(mat, cmap=cmap, aspect="auto")

    ax.set_xticks(np.arange(len(splits)))
    ax.set_xticklabels(split_names_list, rotation=25, ha="right", fontsize=7.5)
    ax.set_yticks(np.arange(len(CLASS_NAMES)))
    ax.set_yticklabels([c.replace("_", " ") for c in CLASS_NAMES], fontsize=7.5)

    for i in range(len(CLASS_NAMES)):
        for j in range(len(splits)):
            is_known = mat[i, j] == 1.0
            ax.text(j, i, "K" if is_known else "U", ha="center", va="center",
                    color="white", fontsize=7.0, fontweight="bold")

    ax.set_title("Open-Set Class Partition Matrix (K=Known, U=Unknown)", fontsize=8.5)

    save_dual_figure(fig, "12_split_taxonomy_matrix")


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def main():
    raw_csv = "results/main/raw_results.csv"
    if not os.path.exists(raw_csv):
        print(f"Error: {raw_csv} not found.")
        return

    print("Loading authoritative benchmark results from results/main/raw_results.csv...")
    df = pd.read_csv(raw_csv)

    print("\n--- Generating Clean Paper Figures (NO Gridlines, NO Top Bar Lines) ---")
    plot_01_detector_comparison(df)
    plot_02_snr_robustness(df)
    plot_03_split_regime_hardness(df)
    plot_04_architecture_comparison(df)
    plot_05_closed_vs_open_set(df)
    plot_06_auroc_distribution_boxplots(df)
    plot_10_detector_ranking_heatmap(df)
    plot_11_pqd_waveforms()
    plot_12_split_taxonomy_matrix()

    print("\nExtracting representative model data for ROC, OSCR, KDE, and t-SNE figures...")
    data = build_dataset(120, 30, 80, seed=0, cache_dir="data_cache")
    splits = make_splits(1, 4)
    split = [s for s in splits if s["type"] == "near"][0]
    known, unknown = split["known"], split["unknown"]
    K = len(known)

    Xtr, ytr = select_classes(data["X_train"], data["y_train"], known)
    Xva, yva = select_classes(data["X_val"], data["y_val"], known)
    Xk, yk = select_classes(data["X_test"], data["y_test"], known)
    Xu, yu_orig = select_classes(data["X_test"], data["y_test"], unknown, remap=False)

    train_snrs = [10, 20, 30, 40]
    base = [1000, 0]
    Xtr_n = noisy_train(Xtr, train_snrs, base + [0, 0])
    Xva_n = noisy_train(Xva, train_snrs, base + [1, 0])

    model = train_model("cnn", Xtr_n, ytr, Xva_n, yva, K, epochs=10, bs=128, lr=1e-3, wd=1e-4, width=32, device="cpu", seed=0)
    lo_tr, fe_tr = extract(model, Xtr_n, "cpu")
    mah = Mahalanobis().fit(fe_tr, ytr, K)
    om = OpenMax().fit(lo_tr, ytr, K)
    ocd = OCSVM("deep", seed=0).fit(fe_tr, None)
    och = OCSVM("hand", seed=0).fit(None, Xtr_n)

    eval_snr = 30.0
    Xk_n = noisy(Xk, eval_snr, base + [2, int(eval_snr)])
    Xu_n = noisy(Xu, eval_snr, base + [3, int(eval_snr)])
    lo_k, fe_k = extract(model, Xk_n, "cpu")
    lo_u, fe_u = extract(model, Xu_n, "cpu")

    def get_scores(lo, fe, X):
        pred = lo.argmax(1)
        return {
            "MSP": score_msp(lo),
            "Energy": score_energy(lo),
            "Mahalanobis": (mah.score(fe)[0], pred),
            "OpenMax": om.score(lo),
            "OCSVM-deep": ocd.score(fe, X),
            "OCSVM-hand": och.score(fe, X),
        }

    rep_data = {
        "model": model, "split": split, "known": known, "unknown": unknown,
        "Xk_n": Xk_n, "Xu_n": Xu_n, "yk": yk, "yu_orig": yu_orig,
        "lo_k": lo_k, "lo_u": lo_u, "fe_k": fe_k, "fe_u": fe_u,
        "sk": get_scores(lo_k, fe_k, Xk_n),
        "su": get_scores(lo_u, fe_u, Xu_n)
    }

    plot_07_oscr_and_roc_curves(rep_data)
    plot_08_score_distributions_kde(rep_data)
    plot_09_latent_feature_tsne(rep_data)

    print(f"\nAll publication figures successfully updated in {PAPER_FIG_DIR}/")


if __name__ == "__main__":
    main()
