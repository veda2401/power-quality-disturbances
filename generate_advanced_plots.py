#!/usr/bin/env python
"""Generate advanced analysis graphs: ROC curves, PR curves, OSCR curves,
Score distribution KDEs, t-SNE / PCA latent spaces, Confusion Matrix,
Radar/Spider charts, and Ranking Heatmaps.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import scipy.stats as stats

from pqd.data import build_dataset, noisy, noisy_train, select_classes
from pqd.generator import CLASS_NAMES, FS, F0, N
from pqd.scoring import METHODS, Mahalanobis, OCSVM, OpenMax, score_energy, score_msp
from pqd.splits import make_splits, split_names
from pqd.train import extract, train_model

OUTPUT_DIR = "figures/advanced_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.sans-serif": "DejaVu Sans",
    "font.family": "sans-serif",
    "figure.titlesize": 13,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
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


def compute_representative_model():
    """Train a representative model to extract detailed scores, features, and curves."""
    print("Training representative model for deep feature and ROC analysis...")
    data = build_dataset(120, 30, 80, seed=0, cache_dir="data_cache")
    splits = make_splits(1, 4)
    split = [s for s in splits if s["type"] == "near"][0]  # Use 'near' OOD split for challenge
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

    # Evaluate at 30 dB SNR
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

    sk = get_scores(lo_k, fe_k, Xk_n)
    su = get_scores(lo_u, fe_u, Xu_n)

    return {
        "model": model, "split": split, "known": known, "unknown": unknown,
        "Xk_n": Xk_n, "Xu_n": Xu_n, "yk": yk, "yu_orig": yu_orig,
        "lo_k": lo_k, "lo_u": lo_u, "fe_k": fe_k, "fe_u": fe_u,
        "sk": sk, "su": su
    }


def plot_roc_and_pr_curves(rep_data):
    """1. Full Receiver Operating Characteristic (ROC) & Precision-Recall (PR) curves."""
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 5.5))
    sk, su = rep_data["sk"], rep_data["su"]

    for m in METHODS:
        s_k = sk[m][0]
        s_u = su[m][0]
        y_true = np.concatenate([np.ones(len(s_k)), np.zeros(len(s_u))])
        scores = np.concatenate([s_k, s_u])

        # ROC: Known is positive (1), higher score = known
        fpr, tpr, _ = roc_curve(y_true, scores)
        from sklearn.metrics import auc, precision_recall_curve
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, label=f"{m} (AUROC = {roc_auc:.3f})", color=METHOD_COLORS[m], lw=2.0)

        # PR: Unknown is positive (1), lower score = unknown -> use -scores
        precision, recall, _ = precision_recall_curve(1 - y_true, -scores)
        pr_auc = auc(recall, precision)
        ax_pr.plot(recall, precision, label=f"{m} (AUPR-U = {pr_auc:.3f})", color=METHOD_COLORS[m], lw=2.0)

    ax_roc.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.6, label="Random Guess (0.500)")
    ax_roc.set_xlabel("False Positive Rate (Unknowns misclassified as Known)")
    ax_roc.set_ylabel("True Positive Rate (Knowns correctly accepted)")
    ax_roc.set_title("(a) Receiver Operating Characteristic (ROC) Curves", fontweight="bold")
    ax_roc.grid(True, linestyle="--", alpha=0.4)
    ax_roc.legend(loc="lower right", fontsize=8.5)

    ax_pr.set_xlabel("Recall (Fraction of Unknowns detected)")
    ax_pr.set_ylabel("Precision (True Unknowns / Total Flagged)")
    ax_pr.set_title("(b) Precision-Recall (PR) Curves (Unknown = Positive Class)", fontweight="bold")
    ax_pr.grid(True, linestyle="--", alpha=0.4)
    ax_pr.legend(loc="lower left", fontsize=8.5)

    fig.suptitle("Open-Set Detection Performance Trade-Offs @ 30 dB SNR (Near-OOD Split)", fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    path = os.path.join(OUTPUT_DIR, "01_roc_and_pr_curves.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_oscr_curves(rep_data):
    """2. Open-Set Classification Rate (OSCR) curves: Correct Classification Rate vs FPR."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sk, su = rep_data["sk"], rep_data["su"]
    yk = rep_data["yk"]

    for m in METHODS:
        s_k, pred_k = sk[m]
        if pred_k is None:
            continue  # Skip OCSVMs (no classifier)
        correct_k = (pred_k == yk).astype(float)
        s_u = su[m][0]

        # Compute OSCR curve points
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
        ax.plot(fpr, ccr, label=f"{m} (OSCR = {area:.3f})", color=METHOD_COLORS[m], lw=2.2)

    ax.set_xlabel("False Positive Rate on Unknowns (FPR)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Correct Classification Rate on Knowns (CCR)", fontsize=10, fontweight="bold")
    ax.set_title("Open-Set Classification Rate (OSCR) Curves\n(Simultaneous Closed-Set Classification & Unknown Rejection)", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right", fontsize=9.5)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "02_oscr_curves.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_score_distributions_kde(rep_data):
    """3. Kernel Density Estimation (KDE) of Known vs Unknown Score Distributions."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=False)
    sk, su = rep_data["sk"], rep_data["su"]

    for idx, (m, ax) in enumerate(zip(METHODS, axes.flat)):
        s_k = sk[m][0]
        s_u = su[m][0]

        # Standardize scores for cleaner visualization
        s_min, s_max = min(s_k.min(), s_u.min()), max(s_k.max(), s_u.max())
        x_eval = np.linspace(s_min, s_max, 300)

        kde_k = stats.gaussian_kde(s_k)(x_eval)
        kde_u = stats.gaussian_kde(s_u)(x_eval)

        ax.plot(x_eval, kde_k, label="Known In-Distribution", color="#2ca02c", lw=2.0)
        ax.fill_between(x_eval, kde_k, color="#2ca02c", alpha=0.25)
        ax.plot(x_eval, kde_u, label="Unknown Out-of-Dist", color="#d62728", lw=2.0)
        ax.fill_between(x_eval, kde_u, color="#d62728", alpha=0.25)

        # 5th percentile threshold
        thr = np.percentile(s_k, 5)
        ax.axvline(thr, color="black", linestyle="--", lw=1.2, label=f"95% Val Thr ({thr:.2f})")

        ax.set_title(f"Method: {m}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Knownness Score")
        ax.set_ylabel("Probability Density")
        ax.grid(True, linestyle="--", alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7.5, loc="upper left")

    fig.suptitle("Known vs. Unknown Score Separation Distributions across All 6 Detectors\n(Higher Score = Identified as In-Distribution Known Disturbance)", fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    path = os.path.join(OUTPUT_DIR, "03_score_distributions_kde.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_confusion_matrix_knowns(rep_data):
    """4. Normalized Confusion Matrix of Closed-Set Classification on Known Classes."""
    yk = rep_data["yk"]
    pred_k = rep_data["sk"]["MSP"][1]
    known_indices = rep_data["known"]
    known_names = [CLASS_NAMES[i] for i in known_indices]

    cm = confusion_matrix(yk, pred_k, normalize="true")
    fig, ax = plt.subplots(figsize=(10, 8.5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[n.replace('_', '\n') for n in known_names])
    disp.plot(ax=ax, cmap="Blues", values_format=".2f", colorbar=True)
    
    ax.set_title("Closed-Set Classification Confusion Matrix on Known Disturbance Classes\n(1D-CNN @ 30 dB SNR, Overall Accuracy = {:.1f}%)".format(100.0 * np.mean(yk == pred_k)), fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Disturbance Class", fontweight="bold")
    ax.set_ylabel("True Disturbance Class", fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "04_confusion_matrix_closed_set.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_tsne_and_pca_latent_space(rep_data):
    """5. t-SNE and PCA 2D projections of the penultimate feature representations."""
    fe_k = rep_data["fe_k"]
    fe_u = rep_data["fe_u"]
    yk = rep_data["yk"]
    yu = rep_data["yu_orig"]
    known_indices = rep_data["known"]
    unknown_indices = rep_data["unknown"]

    # Subsample for crisp visualization
    np.random.seed(42)
    idx_k = np.random.choice(len(fe_k), size=min(600, len(fe_k)), replace=False)
    idx_u = np.random.choice(len(fe_u), size=min(200, len(fe_u)), replace=False)

    X_all = np.vstack([fe_k[idx_k], fe_u[idx_u]])
    labels_all = np.concatenate([yk[idx_k], np.full(len(idx_u), -1)])  # -1 for Unknown

    # Compute PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_all)

    # Compute t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X_all)

    fig, (ax_tsne, ax_pca) = plt.subplots(1, 2, figsize=(16, 7))

    # Color palette for known classes
    cmap = matplotlib.colormaps.get_cmap("tab20")

    for k_local, k_orig in enumerate(known_indices):
        mask = (labels_all == k_local)
        name = CLASS_NAMES[k_orig]
        ax_tsne.scatter(X_tsne[mask, 0], X_tsne[mask, 1], s=25, alpha=0.7, color=cmap(k_local), label=f"Known: {name}")
        ax_pca.scatter(X_pca[mask, 0], X_pca[mask, 1], s=25, alpha=0.7, color=cmap(k_local))

    # Plot unknown classes as high-contrast red/black stars
    mask_u = (labels_all == -1)
    ax_tsne.scatter(X_tsne[mask_u, 0], X_tsne[mask_u, 1], s=60, marker="X", color="#d62728", edgecolor="black", lw=0.5, label="Unknown (Held-Out OOD)", zorder=10)
    ax_pca.scatter(X_pca[mask_u, 0], X_pca[mask_u, 1], s=60, marker="X", color="#d62728", edgecolor="black", lw=0.5, label="Unknown (Held-Out OOD)", zorder=10)

    ax_tsne.set_title("(a) 2D t-SNE Manifold of Penultimate Features", fontweight="bold")
    ax_tsne.set_xlabel("t-SNE Dimension 1")
    ax_tsne.set_ylabel("t-SNE Dimension 2")
    ax_tsne.grid(True, linestyle="--", alpha=0.3)

    ax_pca.set_title(f"(b) 2D PCA Latent Space (Expl. Var: {pca.explained_variance_ratio_.sum()*100:.1f}%)", fontweight="bold")
    ax_pca.set_xlabel("Principal Component 1")
    ax_pca.set_ylabel("Principal Component 2")
    ax_pca.grid(True, linestyle="--", alpha=0.3)

    # Place legend below
    handles, labels = ax_tsne.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=5, fontsize=8.5)

    fig.suptitle("Latent Feature Representations of 1D-CNN: Known Clusters vs Out-of-Distribution Unknown Disturbance Classes", fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    path = os.path.join(OUTPUT_DIR, "05_latent_feature_tsne_pca.png")
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


def plot_radar_and_rankings(raw_csv_path="results/main/raw_results.csv"):
    """6. Multi-Metric Radar/Spider Chart & Method Ranking Heatmap."""
    if not os.path.exists(raw_csv_path):
        print(f"Results CSV not found at {raw_csv_path}, skipping radar chart.")
        return

    df = pd.read_csv(raw_csv_path)
    ref = df[df.snr_db == 30.0]

    # Metrics for radar chart (normalized to 0..1 scale, higher is better)
    # We invert FPR95 -> (1 - FPR95)
    metrics_list = ["auroc", "aupr_unk", "1_minus_fpr95", "oscr", "urr", "tar"]
    metric_labels = ["AUROC", "AUPR (Unknown)", "1 - FPR95", "OSCR", "URR", "TAR"]

    ref = ref.copy()
    ref["1_minus_fpr95"] = 1.0 - ref["fpr95"]

    radar_data = {}
    for m in METHODS:
        gm = ref[ref.method == m]
        vals = []
        for met in ["auroc", "aupr_unk", "1_minus_fpr95", "oscr", "urr", "tar"]:
            v = gm[met].mean()
            vals.append(v if not np.isnan(v) else 0.0)
        radar_data[m] = vals

    # Plot Radar Chart
    angles = np.linspace(0, 2 * np.pi, len(metrics_list), endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for m in METHODS:
        vals = radar_data[m] + radar_data[m][:1]
        ax.plot(angles, vals, color=METHOD_COLORS[m], linewidth=2, label=m)
        ax.fill(angles, vals, color=METHOD_COLORS[m], alpha=0.1)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), metric_labels, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=9)
    ax.set_title("Multi-Metric Spider / Radar Profile of Detectors @ 30 dB SNR\n(Higher is Better across All Axes)", fontsize=12, fontweight="bold", pad=20)
    fig.tight_layout()
    path_radar = os.path.join(OUTPUT_DIR, "06_radar_chart_methods.png")
    fig.savefig(path_radar, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path_radar}")

    # Plot Average Rank Heatmap across SNRs
    rk = {}
    for snr, g in df.groupby("snr_db"):
        for arch, ga in g.groupby("arch"):
            p = ga.pivot_table(index=["split", "seed"], columns="method", values="auroc")
            rk[f"{arch.upper()} @ {snr:.0f}dB"] = p.rank(axis=1, ascending=False).mean()
    rank_df = pd.DataFrame(rk).T
    rank_df = rank_df[[m for m in METHODS if m in rank_df.columns]]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    im = ax.imshow(rank_df.values, cmap="YlGn_r", aspect="auto")
    ax.set_xticks(np.arange(len(rank_df.columns)))
    ax.set_xticklabels(rank_df.columns, fontsize=10, fontweight="bold")
    ax.set_yticks(np.arange(len(rank_df.index)))
    ax.set_yticklabels(rank_df.index, fontsize=9.5, fontweight="bold")

    for i in range(len(rank_df.index)):
        for j in range(len(rank_df.columns)):
            v = rank_df.values[i, j]
            ax.text(j, i, f"Rank {v:.2f}", ha="center", va="center", color="black" if v > 3.0 else "white", fontweight="bold", fontsize=9)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Average Rank (1 = Best)", fontweight="bold")
    ax.set_title("Detector Ranking Matrix by AUROC across Architecture & SNR Conditions\n(Over All Evaluation Splits and Model Seeds)", fontsize=12, fontweight="bold", pad=12)
    fig.tight_layout()
    path_rank = os.path.join(OUTPUT_DIR, "07_detector_ranking_heatmap.png")
    fig.savefig(path_rank)
    plt.close(fig)
    print(f"Saved: {path_rank}")


if __name__ == "__main__":
    print("Generating advanced open-set evaluation figures...")
    rep_data = compute_representative_model()
    plot_roc_and_pr_curves(rep_data)
    plot_oscr_curves(rep_data)
    plot_score_distributions_kde(rep_data)
    plot_confusion_matrix_knowns(rep_data)
    plot_tsne_and_pca_latent_space(rep_data)
    print("Advanced ROC, OSCR, KDE, CM, and t-SNE figures generated successfully!")
