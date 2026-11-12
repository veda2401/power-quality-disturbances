"""Sanity checks.  Run:  python tests/test_smoke.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pqd.generator import CLASS_NAMES, N, NUM_CLASSES, generate_one, add_noise
from pqd.data import build_dataset
from pqd.metrics import oscr, open_set_metrics
from pqd.scoring import Mahalanobis, OpenMax, score_energy, score_msp
from pqd.splits import make_splits


def test_generator():
    for name in CLASS_NAMES:
        x = generate_one(name, np.random.default_rng(0))
        assert x.shape == (N,) and np.isfinite(x).all(), name
        assert np.allclose(x, generate_one(name, np.random.default_rng(0))), "not deterministic"
    print("generator ok")


def test_dataset_reproducible():
    a = build_dataset(3, 2, 2, seed=5)
    b = build_dataset(3, 2, 2, seed=5)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert a["X_train"].shape == (3 * NUM_CLASSES, N)
    n = add_noise(a["X_train"], 20, np.random.default_rng(0))
    assert abs(np.mean((n - a["X_train"]) ** 2) - 0.5 / 100) < 2e-3   # 20 dB vs 0.5 pu^2
    print("dataset ok")


def test_splits():
    sp = make_splits(3)
    assert len(sp) == 5
    near = [s for s in sp if s["type"] == "near"][0]
    far = [s for s in sp if s["type"] == "far"][0]
    assert near["novel_components"] == [], near
    assert set(far["novel_components"]) == {"flicker", "notch"}, far
    print("splits ok")


def test_metrics():
    rng = np.random.default_rng(0)
    # perfectly separable, all correct -> OSCR = 1, AUROC = 1
    sk, su = np.ones(50), np.zeros(50)
    assert abs(oscr(sk, np.ones(50), su) - 1.0) < 1e-9
    m = open_set_metrics(sk + rng.normal(0, 1e-3, 50), su + rng.normal(0, 1e-3, 50), sk)
    assert m["auroc"] == 1.0 and m["fpr95"] == 0.0 and m["urr"] == 1.0
    # identical distributions -> AUROC ~ 0.5
    a, b = rng.normal(size=4000), rng.normal(size=4000)
    assert abs(open_set_metrics(a, b, a)["auroc"] - 0.5) < 0.03
    # correctness caps OSCR: nothing correct -> 0
    assert oscr(sk, np.zeros(50), su) == 0.0
    print("metrics ok")


def test_scorers():
    rng = np.random.default_rng(0)
    K, D = 4, 16
    labels = rng.integers(0, K, 600)
    centers = rng.normal(0, 4, (K, D))
    feats = centers[labels] + rng.normal(0, 1, (600, D))
    logits = feats[:, :K]
    mah = Mahalanobis().fit(feats, labels, K)
    om = OpenMax().fit(logits, labels, K)
    far = rng.normal(0, 1, (200, D)) + 15                     # far-away 'unknown' cloud
    s_k, _ = mah.score(feats)
    s_u, _ = mah.score(far)
    assert s_k.mean() > s_u.mean()
    for s in (om.score(logits)[0], score_msp(logits)[0], score_energy(logits)[0]):
        assert np.isfinite(s).all()
    print("scorers ok")


if __name__ == "__main__":
    test_generator()
    test_dataset_reproducible()
    test_splits()
    test_metrics()
    test_scorers()
    print("ALL TESTS PASSED")
