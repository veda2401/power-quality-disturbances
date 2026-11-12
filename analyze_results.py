#!/usr/bin/env python
"""Turn results/raw_results.csv into paper-ready tables (Markdown + LaTeX)
and figures.

    python analyze_results.py --results results --matched_snr 30
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from pqd.scoring import METHODS

MET = {"closed_acc": "Closed acc.", "auroc": "AUROC", "aupr_unk": "AUPR-U", "fpr95": "FPR95",
       "oscr": "OSCR", "urr": "URR", "tar": "TAR"}
LOWER_IS_BETTER = {"fpr95"}


def ms(x, d=3):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return "--" if len(x) == 0 else f"{x.mean():.{d}f} ± {x.std(ddof=1) if len(x) > 1 else 0:.{d}f}"


def to_latex(df, caption, label):
    cols = "l" * df.index.nlevels + "c" * len(df.columns)
    lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\resizebox{\columnwidth}{!}{", rf"\begin{{tabular}}{{{cols}}}", r"\toprule"]
    head = [str(n) if n else "" for n in df.index.names] + [str(c) for c in df.columns]
    lines.append(" & ".join(head).replace("_", r"\_") + r" \\")
    lines.append(r"\midrule")
    for idx, row in df.iterrows():
        idx = idx if isinstance(idx, tuple) else (idx,)
        cells = [str(v) for v in idx] + [str(v) for v in row.values]
        lines.append(" & ".join(cells).replace("±", r"$\pm$").replace("_", r"\_") + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines)


def save(df, name, out, caption):
    df.reset_index().to_markdown(os.path.join(out, f"{name}.md"), index=False)
    with open(os.path.join(out, f"{name}.tex"), "w") as f:
        f.write(to_latex(df, caption, f"tab:{name}"))
    df.to_csv(os.path.join(out, f"{name}.csv"))


def table(df, index, metrics):
    rows = {}
    for key, g in df.groupby(index, sort=False):
        rows[key] = {MET[m]: ms(g[m]) for m in metrics}
    t = pd.DataFrame.from_dict(rows, orient="index")
    t.index = pd.MultiIndex.from_tuples([k if isinstance(k, tuple) else (k,) for k in t.index], names=index)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--matched_snr", type=float, default=30.0, help="SNR used for the main tables")
    a = ap.parse_args()
    df = pd.read_csv(os.path.join(a.results, "raw_results.csv"))
    out = os.path.join(a.results, "tables")
    os.makedirs(out, exist_ok=True)
    order = {m: i for i, m in enumerate(METHODS)}
    df["m_order"] = df.method.map(order)
    df = df.sort_values(["arch", "m_order"]).drop(columns="m_order")
    ref = df[df.snr_db == a.matched_snr]
    n_runs = ref.drop_duplicates(["arch", "split", "seed"]).groupby("arch").size()
    print("runs per architecture:", {k: int(v) for k, v in n_runs.items()})

    # 1. main table (pooled over all splits and seeds)
    save(table(ref, ["arch", "method"], list(MET)), "table_main", out,
         f"Open-set results at {a.matched_snr:.0f} dB SNR, mean $\\pm$ std over all splits and seeds.")

    # 2. by split type
    save(table(ref, ["arch", "split_type", "method"], ["auroc", "fpr95", "oscr", "urr"]),
         "table_by_split_type", out, "Results by split type at the matched SNR.")

    # 3. by SNR (robustness)
    for m in ["auroc", "fpr95", "oscr", "closed_acc"]:
        t = df.pivot_table(index=["arch", "method"], columns="snr_db", values=m,
                           aggfunc=lambda x: ms(x))
        t.columns = [f"{c:.0f} dB" for c in t.columns]
        save(t, f"table_snr_{m}", out, f"{MET[m]} versus test SNR.")
    thr = df.pivot_table(index=["arch", "method"], columns="snr_db", values="tar", aggfunc=lambda x: ms(x))
    thr.columns = [f"{c:.0f} dB" for c in thr.columns]
    save(thr, "table_snr_tar", out, "Known acceptance rate at the fixed (validation) threshold versus test SNR.")

    # 4. average ranks by AUROC per SNR
    rk = {}
    for snr, g in df.groupby("snr_db"):
        for arch, ga in g.groupby("arch"):
            p = ga.pivot_table(index=["split", "seed"], columns="method", values="auroc")
            rk[(arch, f"{snr:.0f} dB")] = p.rank(axis=1, ascending=False).mean().round(2)
    ranks = pd.DataFrame(rk).T
    ranks.index = pd.MultiIndex.from_tuples(ranks.index, names=["arch", "SNR"])
    ranks = ranks[[m for m in METHODS if m in ranks.columns]]
    save(ranks, "table_avg_rank_auroc", out, "Average rank by AUROC (1 = best), over splits and seeds.")

    # 5. Paired Wilcoxon signed-rank vs best method with Holm correction, effect size, and 95% CI
    # Experimental Unit: Matched (data_seed x split x seed) runs (not individual signal instances)
    run_idx = ["data_seed", "split", "seed"] if "data_seed" in ref.columns else ["split", "seed"]
    rows = []
    for arch, ga in ref.groupby("arch"):
        p = ga.pivot_table(index=run_idx, columns="method", values="auroc")
        best = p.mean().idxmax()
        comp_methods = [m for m in p.columns if m != best]
        
        # Collect raw p-values for Holm-Bonferroni step-down
        raw_pvals = []
        stats_data = []
        for m in comp_methods:
            diffs = (p[best] - p[m]).dropna().values
            n_pairs = len(diffs)
            nz_diffs = diffs[diffs != 0]
            if len(nz_diffs) >= 5:
                try:
                    res = wilcoxon(diffs, alternative="two-sided")
                    pv = float(res.pvalue)
                    w_stat = float(res.statistic)
                    # Rank-Biserial correlation
                    # Total sum of ranks T = n*(n+1)/2
                    n_nz = len(nz_diffs)
                    T = n_nz * (n_nz + 1) / 2.0
                    # scipy returns min(W+, W-)
                    ranks = np.argsort(np.argsort(np.abs(nz_diffs))) + 1.0
                    w_plus = np.sum(ranks[nz_diffs > 0])
                    w_minus = np.sum(ranks[nz_diffs < 0])
                    r_biserial = (w_plus - w_minus) / T
                except Exception:
                    pv, r_biserial = float("nan"), float("nan")
            else:
                pv, r_biserial = float("nan"), float("nan")

            # Cohen's dz
            mean_diff = np.mean(diffs)
            std_diff = np.std(diffs, ddof=1) if len(diffs) > 1 else 1e-6
            cohen_dz = mean_diff / (std_diff + 1e-12)

            # Bootstrap 95% CI for mean difference
            np.random.seed(42)
            boot_means = [np.mean(np.random.choice(diffs, size=len(diffs), replace=True)) for _ in range(1000)]
            ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

            raw_pvals.append(pv)
            stats_data.append({
                "arch": arch, "best": best, "vs": m,
                "mean diff": f"{mean_diff:+.3f}",
                "95% CI": f"[{ci_low:+.3f}, {ci_high:+.3f}]",
                "Cohen dz": f"{cohen_dz:+.2f}",
                "r_rb": f"{r_biserial:+.2f}" if not np.isnan(r_biserial) else "--",
                "p_raw": pv,
                "n pairs": n_pairs
            })

        # Holm-Bonferroni adjustment
        valid_indices = [i for i, pv in enumerate(raw_pvals) if not np.isnan(pv)]
        sorted_indices = sorted(valid_indices, key=lambda i: raw_pvals[i])
        m_tests = len(sorted_indices)
        
        holm_pvals = {i: raw_pvals[i] for i in range(len(raw_pvals))}
        running_max = 0.0
        for rank_idx, orig_i in enumerate(sorted_indices):
            k = m_tests - rank_idx
            adj = min(1.0, raw_pvals[orig_i] * k)
            adj = max(adj, running_max)  # ensure monotonicity
            running_max = adj
            holm_pvals[orig_i] = adj

        for i, srow in enumerate(stats_data):
            p_raw = srow.pop("p_raw")
            p_adj = holm_pvals[i]
            srow["p (Wilcoxon)"] = f"{p_raw:.4f}" if not np.isnan(p_raw) else "--"
            srow["p (Holm-adj)"] = f"{p_adj:.4f}" if not np.isnan(p_adj) else "--"
            rows.append(srow)

    if rows:
        stat_df = pd.DataFrame(rows).set_index(["arch", "best", "vs"])
        save(stat_df, "table_wilcoxon", out,
             "Paired Wilcoxon signed-rank tests with Holm-Bonferroni correction and effect sizes on AUROC versus best detector. Experimental unit: matched (split x seed) run.")

    # figures
    fig_dir = os.path.join(a.results, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    for arch, ga in df.groupby("arch"):
        fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
        for ax, (m, lab) in zip(axes, [("auroc", "AUROC"), ("fpr95", "FPR95"), ("closed_acc", "Closed-set accuracy")]):
            for meth in METHODS:
                g = ga[ga.method == meth].groupby("snr_db")[m].agg(["mean", "std"]).dropna(subset=["mean"])
                if g.empty:
                    continue
                ax.errorbar(g.index, g["mean"], yerr=g["std"].fillna(0), marker="o", capsize=2, label=meth)
            ax.set_xlabel("Test SNR (dB)"); ax.set_ylabel(lab); ax.grid(alpha=.3)
        axes[0].legend(fontsize=7)
        fig.suptitle(f"Robustness to noise ({arch})"); fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"snr_robustness_{arch}.png"), dpi=200); plt.close(fig)

        r = ga[ga.snr_db == a.matched_snr]
        types = ["random", "near", "far"]
        fig, ax = plt.subplots(figsize=(7, 3.8))
        w = 0.8 / len(METHODS)
        for i, meth in enumerate(METHODS):
            vals = [r[(r.method == meth) & (r.split_type == t)].auroc.mean() for t in types]
            errs = [r[(r.method == meth) & (r.split_type == t)].auroc.std() for t in types]
            ax.bar(np.arange(3) + i * w, vals, w, yerr=np.nan_to_num(errs), label=meth, capsize=2)
        ax.set_xticks(np.arange(3) + 0.4 - w / 2); ax.set_xticklabels(types)
        ax.set_ylabel("AUROC"); ax.set_title(f"AUROC by split type at {a.matched_snr:.0f} dB ({arch})")
        ax.axhline(0.5, color="gray", ls="--", lw=.8); ax.legend(fontsize=7, ncol=2); fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"auroc_by_split_type_{arch}.png"), dpi=200); plt.close(fig)
    print(f"tables -> {out}\nfigures -> {fig_dir}")


if __name__ == "__main__":
    main()
