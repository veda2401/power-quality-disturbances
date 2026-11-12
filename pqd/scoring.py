"""Unknown-detection scorers.  Every scorer returns
    known_score : (n,) higher = more likely a known class
    pred        : (n,) predicted known-class index, or None if the method has
                  no classifier (One-Class SVM)
Thresholds are never tuned on unknown data."""
import numpy as np
from scipy.special import logsumexp, softmax
from scipy.stats import weibull_min
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from .features import handcrafted_features

METHODS = ["MSP", "Energy", "Mahalanobis", "OpenMax", "OCSVM-deep", "OCSVM-hand"]


# ---------------------------------------------------------------- softmax based
def score_msp(logits):
    """Maximum Softmax Probability (MSP, Hendrycks & Gimpel, 2017).
    Convention: higher score -> more likely known (in-distribution)."""
    return softmax(logits, axis=1).max(axis=1), logits.argmax(1)


def score_energy(logits, T=1.0):
    """Energy-based OOD scoring (Liu et al., NeurIPS 2020).
    Original Energy E(x) = -T * logsumexp(logits / T) is lower for in-distribution.
    Convention: We return Free Energy score (-E(x) = +T * logsumexp(logits / T)),
    so that higher score -> more likely known, lower score -> more likely unknown."""
    return T * logsumexp(logits / T, axis=1), logits.argmax(1)


# ------------------------------------------------------------------ Mahalanobis
class Mahalanobis:
    """Class-conditional Gaussians with a shared covariance on penultimate
    features (Lee et al., NeurIPS 2018 -- simple version: no input pre-processing
    and no multi-layer ensemble).
    Convention: Score = - min_c (z - mu_c)^T Sigma^{-1} (z - mu_c).
    Higher score (closer to 0) -> more likely known."""
    def fit(self, feats, labels, n_classes):
        self.mu = np.stack([feats[labels == c].mean(0) for c in range(n_classes)])
        centered = feats - self.mu[labels]
        cov = centered.T @ centered / len(feats)
        cov += np.eye(cov.shape[0]) * 1e-3 * np.trace(cov) / cov.shape[0]   # ridge
        self.prec = np.linalg.pinv(cov)
        return self

    def score(self, feats):
        d = np.empty((len(feats), len(self.mu)))
        for c, m in enumerate(self.mu):
            diff = feats - m
            d[:, c] = np.einsum("ij,jk,ik->i", diff, self.prec, diff)
        return -d.min(1), d.argmin(1)


# --------------------------------------------------------------------- OpenMax
class OpenMax:
    """OpenMax (Bendale & Boult, CVPR 2016) -- Euclidean Logit-Space Variant.

    Algorithm specification & variant notes:
    1. Feature Representation: Operates on unnormalized logit vectors v(x) in R^K.
    2. Mean Activation Vector (MAV): Per-class MAV mu_c is computed from correctly
       classified training samples (falling back to all class samples if < 5 correct).
    3. Distance Metric: Euclidean distance ||v(x) - mu_c||_2 in logit space.
    4. Weibull Fitting: Fits a 2-parameter Weibull distribution (shape, scale; loc=0)
       to the tail (largest `tail` distances) of training samples to their MAV using
       scipy.stats.weibull_min (avoiding external C-library libMR bindings).
    5. alpha-Ranking Weighting: For input x, top `alpha` predicted classes receive
       a calibrated Weibull CDF weight:
           omega_c(x) = 1.0 - ((alpha - rank) / alpha) * Weibull_CDF(dist(x, mu_c)).
    6. Recalibrated Probabilities & Unknown Class:
           revised_logits_c = v_c(x) * omega_c(x)
           unknown_logit = sum_{c=1}^K v_c(x) * (1 - omega_c(x))
           P = Softmax([unknown_logit, revised_logits_1, ..., revised_logits_K])
    7. Knownness Score:
           known_score = 1.0 - P(unknown) = sum_{c=1}^K P_c(x).
       Convention: higher score -> more likely known.
    """
    def __init__(self, tail=20, alpha=5):
        self.tail, self.alpha = tail, alpha

    def fit(self, logits, labels, n_classes):
        self.K = n_classes
        self.alpha = min(self.alpha, n_classes)
        pred = logits.argmax(1)
        self.mav, self.wb = [], []
        for c in range(n_classes):
            v = logits[(labels == c) & (pred == c)]
            if len(v) < 5:                       # fallback: use all samples of the class
                v = logits[labels == c]
            mav = v.mean(0)
            dist = np.linalg.norm(v - mav, axis=1)
            tail = np.sort(dist)[-min(self.tail, len(dist)):]
            try:
                shape, _, scale = weibull_min.fit(tail, floc=0)
                if not (np.isfinite(shape) and np.isfinite(scale) and scale > 0):
                    raise ValueError
            except Exception:
                shape, scale = 1.0, float(tail.mean()) + 1e-6
            self.mav.append(mav)
            self.wb.append((shape, scale))
        self.mav = np.stack(self.mav)
        return self

    def score(self, logits):
        n = len(logits)
        dist = np.linalg.norm(logits[:, None, :] - self.mav[None], axis=2)   # (n, K)
        W = np.ones((n, self.K))
        ranks = np.argsort(-logits, axis=1)
        rows = np.arange(n)
        for r in range(self.alpha):
            j = ranks[:, r]
            cdf = np.empty(n)
            for c in np.unique(j):
                m = j == c
                cdf[m] = weibull_min.cdf(dist[m, c], self.wb[c][0], loc=0, scale=self.wb[c][1])
            W[rows, j] = 1.0 - ((self.alpha - r) / self.alpha) * cdf
        revised = logits * W
        unk = (logits * (1.0 - W)).sum(1, keepdims=True)
        p = softmax(np.concatenate([unk, revised], axis=1), axis=1)
        return 1.0 - p[:, 0], revised.argmax(1)


# ------------------------------------------------------------------ One-Class SVM
class OCSVM:
    """One-Class SVM on standardized features, trained on known classes only
    (no class labels).  `kind` = 'deep' (penultimate features of the trained
    network) or 'hand' (hand-crafted features from the raw waveform)."""
    def __init__(self, kind, nu=0.05, max_train=3000, seed=0):
        self.kind, self.nu, self.max_train, self.seed = kind, nu, max_train, seed

    def _f(self, feats, X):
        return feats if self.kind == "deep" else handcrafted_features(X)

    def fit(self, feats, X):
        F = self._f(feats, X)
        if len(F) > self.max_train:
            idx = np.random.default_rng(self.seed).choice(len(F), self.max_train, replace=False)
            F = F[idx]
        self.sc = StandardScaler().fit(F)
        self.m = OneClassSVM(kernel="rbf", gamma="scale", nu=self.nu).fit(self.sc.transform(F))
        return self

    def score(self, feats, X):
        return self.m.decision_function(self.sc.transform(self._f(feats, X))), None
