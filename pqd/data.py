"""Dataset construction, caching and noise handling."""
import os
import numpy as np
from .generator import CLASS_NAMES, NUM_CLASSES, N, NOISE_REFERENCE_POWER, generate_one, add_noise

GEN_VERSION = "1.0"
_SPLIT_ID = {"train": 0, "val": 1, "test": 2}


def build_dataset(n_train, n_val, n_test, seed=0, cache_dir=None):
    """Generate a clean dataset with n_* samples per class for every class.

    Every (split, class) pair has its own RNG stream, so the data does not
    depend on generation order.  Returns dict with X_train, y_train, ...
    """
    fname = f"pqd_v{GEN_VERSION}_seed{seed}_{n_train}_{n_val}_{n_test}.npz"
    path = os.path.join(cache_dir, fname) if cache_dir else None
    if path and os.path.exists(path):
        z = np.load(path)
        return {k: z[k] for k in z.files}

    out = {}
    for split, n in (("train", n_train), ("val", n_val), ("test", n_test)):
        X = np.empty((NUM_CLASSES * n, N), np.float32)
        y = np.repeat(np.arange(NUM_CLASSES), n).astype(np.int64)
        for c, name in enumerate(CLASS_NAMES):
            rng = np.random.default_rng([seed, _SPLIT_ID[split], c])
            for i in range(n):
                X[c * n + i] = generate_one(name, rng)
        out[f"X_{split}"], out[f"y_{split}"] = X, y

    if path:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez_compressed(path, **out)
    return out


def select_classes(X, y, class_ids, remap=True):
    """Keep only samples of class_ids.  If remap, labels become 0..K-1 in the
    order of class_ids."""
    mask = np.isin(y, class_ids)
    Xs, ys = X[mask], y[mask]
    if remap:
        lut = {c: i for i, c in enumerate(class_ids)}
        ys = np.array([lut[v] for v in ys], dtype=np.int64)
    return Xs, ys


def noisy(X, snr_db, seed_parts):
    """Deterministic noisy copy of X.  seed_parts: list of ints for the RNG."""
    if snr_db is None:
        return X.copy()
    return add_noise(X, snr_db, np.random.default_rng(seed_parts))


def noisy_mixed(X, snr_list, seed_parts):
    """Noisy copy where every sample gets an SNR drawn uniformly from snr_list."""
    rng = np.random.default_rng(seed_parts)
    snrs = rng.choice(np.asarray(snr_list, float), size=len(X))
    sigma = np.sqrt(NOISE_REFERENCE_POWER / 10.0 ** (snrs / 10.0))[:, None]
    return (X + rng.normal(0.0, 1.0, X.shape) * sigma).astype(np.float32)


def noisy_train(X, snr_list, seed_parts):
    """Single SNR -> fixed-SNR noise; several SNRs -> mixed-SNR noise."""
    if len(snr_list) == 1:
        return noisy(X, snr_list[0], seed_parts)
    return noisy_mixed(X, snr_list, seed_parts)
