"""Hand-crafted features for the classical One-Class SVM baseline
(no neural network involved, mirroring older PQD novelty-detection work)."""
import numpy as np
from scipy.stats import kurtosis, skew
from .generator import N, SPC, CYCLES, F0, FS

_HARM = list(range(1, 14))                 # fundamental .. 13th
_HALF = SPC // 2


def _rms(x, axis=-1):
    return np.sqrt(np.mean(x ** 2, axis=axis))


def handcrafted_features(X):
    """X: (n, N) -> (n, d) feature matrix (d = 32)."""
    X = np.asarray(X, np.float64)
    n = len(X)
    feats = []

    rms = _rms(X)
    peak = np.abs(X).max(1)
    feats += [rms, peak, peak / (rms + 1e-9), kurtosis(X, axis=1), skew(X, axis=1)]

    cyc = _rms(X.reshape(n, CYCLES, SPC), axis=2)            # per-cycle RMS
    feats += [cyc.min(1), cyc.max(1), cyc.std(1)]
    half = _rms(X.reshape(n, 2 * CYCLES, _HALF), axis=2)     # per-half-cycle RMS
    feats += [half.min(1), half.max(1), half.std(1),
              np.abs(np.diff(half, axis=1)).max(1)]

    spec = np.abs(np.fft.rfft(X, axis=1)) / (N / 2)
    bin0 = int(round(F0 * N / FS))                           # bin of the fundamental (10)
    fund = spec[:, bin0] + 1e-9
    for h in _HARM:
        feats.append(spec[:, h * bin0] / fund)
    hi = spec[:, 14 * bin0:]
    feats += [np.sqrt((spec[:, 2 * bin0:14 * bin0:bin0] ** 2).sum(1)) / fund,   # THD proxy
              np.sqrt((hi ** 2).sum(1)) / fund]                                  # HF content

    # residual after least-squares fit of the fundamental
    t = np.arange(N) / FS
    B = np.stack([np.sin(2 * np.pi * F0 * t), np.cos(2 * np.pi * F0 * t)], 1)
    coef = np.linalg.lstsq(B, X.T, rcond=None)[0]
    res = X - (B @ coef).T
    feats += [_rms(res), np.abs(res).max(1), kurtosis(res, axis=1)]

    d = np.abs(np.diff(X, axis=1))
    feats += [d.max(1), d.mean(1)]
    return np.stack(feats, axis=1).astype(np.float32)
