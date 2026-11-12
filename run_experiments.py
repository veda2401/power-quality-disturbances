#!/usr/bin/env python
"""Run the open-set PQD benchmark.

    python run_experiments.py --quick          # ~1-2 min smoke test
    python run_experiments.py                  # full benchmark (see README for runtime)

Results are appended to <out>/raw_results.csv after every finished run, so an
interrupted job can simply be restarted (finished runs are skipped).
"""
import argparse
import json
import os
import platform
import time

import numpy as np
import pandas as pd
import sklearn
import torch

from pqd import __version__
from pqd.data import build_dataset, noisy, noisy_train, select_classes
from pqd.generator import CLASS_NAMES, PARAMS, FS, F0, N
from pqd.metrics import open_set_metrics
from pqd.scoring import METHODS, Mahalanobis, OCSVM, OpenMax, score_energy, score_msp
from pqd.splits import make_splits, split_names
from pqd.train import extract, train_model

KIND = {"train": 0, "val": 1, "test_known": 2, "test_unknown": 3}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--archs", nargs="+", default=["cnn", "resnet"], choices=["cnn", "resnet"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="model initialization seeds")
    p.add_argument("--data_seeds", nargs="+", type=int, default=[0, 1, 2], help="synthetic dataset generation seeds")
    p.add_argument("--n_random_splits", type=int, default=3, help="random known/unknown splits (near+far are always added)")
    p.add_argument("--n_unknown", type=int, default=4, help="unknown classes per random split")
    p.add_argument("--n_train", type=int, default=120, help="train samples per class (default: 120)")
    p.add_argument("--n_val", type=int, default=30, help="validation samples per class (default: 30)")
    p.add_argument("--n_test", type=int, default=80, help="test samples per class (default: 80)")
    p.add_argument("--epochs", type=int, default=10, help="training epochs (default: 10)")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--train_snrs", nargs="+", type=float, default=[10, 20, 30, 40],
                   help="SNR(s) in dB of training/validation data. Several values = mixed-SNR training "
                        "(default protocol); a single value (e.g. 30) = noise-mismatch protocol")
    p.add_argument("--test_snrs", nargs="+", type=float, default=[10, 20, 30, 40])
    p.add_argument("--out", default="results/main")
    p.add_argument("--data_dir", default="data_cache")
    p.add_argument("--device", default="auto")
    p.add_argument("--fresh", action="store_true", help="delete existing raw_results.csv first")
    p.add_argument("--quick", action="store_true", help="tiny configuration for a smoke test")
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args()
    if a.quick:
        a.n_train, a.n_val, a.n_test, a.epochs = 60, 20, 40, 3
        a.seeds, a.data_seeds, a.n_random_splits, a.archs, a.test_snrs = [0], [0], 1, ["cnn"], [10.0, 30.0]
    return a


def all_scores(logits, feats, X, mah, om, ocd, och):
    """method -> (known_score, predicted class or None)."""
    pred = logits.argmax(1)
    return {
        "MSP": score_msp(logits),
        "Energy": score_energy(logits),
        "Mahalanobis": (mah.score(feats)[0], pred),
        "OpenMax": om.score(logits),
        "OCSVM-deep": ocd.score(feats, X),
        "OCSVM-hand": och.score(feats, X),
    }


def run_one(arch, split, split_idx, seed, data, data_seed, a, device):
    known, unknown = split["known"], split["unknown"]
    K = len(known)
    Xtr, ytr = select_classes(data["X_train"], data["y_train"], known)
    Xva, yva = select_classes(data["X_val"], data["y_val"], known)
    Xk, yk = select_classes(data["X_test"], data["y_test"], known)
    Xu, _ = select_classes(data["X_test"], data["y_test"], unknown, remap=False)
    base = [data_seed * 10000 + 1000 + split_idx, seed]

    Xtr_n = noisy_train(Xtr, a.train_snrs, base + [KIND["train"], 0])
    Xva_n = noisy_train(Xva, a.train_snrs, base + [KIND["val"], 0])

    model = train_model(arch, Xtr_n, ytr, Xva_n, yva, K, epochs=a.epochs, bs=a.batch_size,
                        lr=a.lr, wd=a.weight_decay, width=a.width, device=device,
                        seed=seed, verbose=a.verbose)
    lo_tr, fe_tr = extract(model, Xtr_n, device)
    lo_va, fe_va = extract(model, Xva_n, device)

    mah = Mahalanobis().fit(fe_tr, ytr, K)
    om = OpenMax().fit(lo_tr, ytr, K)
    ocd = OCSVM("deep", seed=seed).fit(fe_tr, None)
    och = OCSVM("hand", seed=seed).fit(None, Xtr_n)

    val_scores = all_scores(lo_va, fe_va, Xva_n, mah, om, ocd, och)   # thresholds: known val only

    rows = []
    for snr in a.test_snrs:
        Xk_n = noisy(Xk, snr, base + [KIND["test_known"], int(snr)])
        Xu_n = noisy(Xu, snr, base + [KIND["test_unknown"], int(snr)])
        lo_k, fe_k = extract(model, Xk_n, device)
        lo_u, fe_u = extract(model, Xu_n, device)
        sk = all_scores(lo_k, fe_k, Xk_n, mah, om, ocd, och)
        su = all_scores(lo_u, fe_u, Xu_n, mah, om, ocd, och)
        for m in METHODS:
            s_k, pred_k = sk[m]
            correct = None if pred_k is None else (pred_k == yk)
            r = open_set_metrics(s_k, su[m][0], val_scores[m][0], correct)
            r["closed_acc"] = float("nan") if correct is None else float(correct.mean())
            r.update(data_seed=data_seed, arch=arch, split=split["name"], split_type=split["type"], seed=seed,
                     snr_db=snr, train_snrs="-".join(str(int(v)) for v in a.train_snrs),
                     method=m, n_known=K, n_unknown=len(unknown),
                     n_test_known=len(Xk), n_test_unknown=len(Xu))
            rows.append(r)
    return rows


def main():
    a = parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    os.makedirs(a.out, exist_ok=True)
    csv_path = os.path.join(a.out, "raw_results.csv")
    if a.fresh and os.path.exists(csv_path):
        os.remove(csv_path)

    splits = make_splits(a.n_random_splits, a.n_unknown)

    with open(os.path.join(a.out, "config.json"), "w") as f:
        json.dump({
            "args": vars(a),
            "versions": {"pqd_openset": __version__, "python": platform.python_version(),
                          "numpy": np.__version__, "torch": torch.__version__,
                          "sklearn": sklearn.__version__},
            "signal": {"fs_hz": FS, "f0_hz": F0, "samples": N},
            "classes": CLASS_NAMES,
            "splits": [dict(s, known_names=split_names(s)[0], unknown_names=split_names(s)[1]) for s in splits],
        }, f, indent=2)

    done = set()
    if os.path.exists(csv_path):
        cols = ["data_seed", "arch", "split", "seed"]
        old = pd.read_csv(csv_path)
        if "data_seed" in old.columns:
            old_subset = old[cols].drop_duplicates()
            done = set(map(tuple, old_subset.values.tolist()))
        else:
            old_subset = old[["arch", "split", "seed"]].drop_duplicates()
            done = set((0, r[0], r[1], r[2]) for r in old_subset.values.tolist())

    jobs = [(d_seed, arch, i, s, seed) for d_seed in a.data_seeds for arch in a.archs for i, s in enumerate(splits) for seed in a.seeds]
    print(f"device={device}  total_runs={len(jobs)}  already_done={len(done)}  data_seeds={a.data_seeds}  model_seeds={a.seeds}")
    t_all = time.time()
    
    current_data = None
    current_d_seed = None

    for n, (d_seed, arch, i, s, seed) in enumerate(jobs, 1):
        if (d_seed, arch, s["name"], seed) in done:
            continue
        if current_d_seed != d_seed:
            current_data = build_dataset(a.n_train, a.n_val, a.n_test, seed=d_seed, cache_dir=a.data_dir)
            current_d_seed = d_seed

        t0 = time.time()
        rows = run_one(arch, s, i, seed, current_data, d_seed, a, device)
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)
        ref = 30.0 if 30.0 in a.test_snrs else max(a.test_snrs)
        m = df[df.snr_db == ref]
        best = m.sort_values("auroc", ascending=False).iloc[0]
        print(f"[{n}/{len(jobs)}] data_seed={d_seed} {arch:6s} {s['name']:9s} seed={seed}  {time.time() - t0:5.0f}s  "
              f"@{ref:.0f}dB closed-acc={m[m.method == 'MSP'].closed_acc.iloc[0]:.3f}  "
              f"best AUROC={best.auroc:.3f} ({best.method})", flush=True)
    print(f"finished in {(time.time() - t_all) / 60:.1f} min -> {csv_path}")


if __name__ == "__main__":
    main()
