"""Open-set metrics.  Convention: every detector produces a 'known-ness'
score (higher = more likely a known class)."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

_trapz = getattr(np, "trapezoid", None) or np.trapz   # numpy 1.x / 2.x


def oscr(s_known, correct_known, s_unknown):
    """Open-Set Classification Rate (Dhamija et al., 2018): area under the
    curve of correct-classification rate (CCR) versus false-positive rate on
    unknowns, sweeping the acceptance threshold."""
    s_known = np.asarray(s_known, float)
    s_unknown = np.asarray(s_unknown, float)
    correct = np.asarray(correct_known, float)
    nk, nu = len(s_known), len(s_unknown)

    order = np.argsort(-s_known)
    neg_sk = -s_known[order]                                  # ascending
    ccum = np.concatenate([[0.0], np.cumsum(correct[order])])
    neg_su = np.sort(-s_unknown)                              # ascending

    thr = np.unique(np.concatenate([s_known, s_unknown]))[::-1]   # descending
    n_k = np.searchsorted(neg_sk, -thr, side="right")   # known with score >= thr
    n_u = np.searchsorted(neg_su, -thr, side="right")   # unknown with score >= thr
    ccr = np.concatenate([[0.0], ccum[n_k] / nk])
    fpr = np.concatenate([[0.0], n_u / nu])
    return float(_trapz(ccr, fpr))


def open_set_metrics(s_known, s_unknown, s_known_val, correct_known=None):
    """All detection metrics for one (run, SNR, method).

    s_known_val : known-class *validation* scores.  The rejection threshold is
                  fixed from these only (5th percentile -> 95% of known
                  validation samples accepted); no unknown data is used.
    """
    s_known = np.asarray(s_known, float)
    s_unknown = np.asarray(s_unknown, float)
    y = np.concatenate([np.ones(len(s_known)), np.zeros(len(s_unknown))])
    s = np.concatenate([s_known, s_unknown])

    fpr, tpr, _ = roc_curve(y, s)
    fpr95 = float(fpr[min(np.searchsorted(tpr, 0.95, side="left"), len(fpr) - 1)])
    thr = np.percentile(s_known_val, 5)

    out = {
        "auroc": float(roc_auc_score(y, s)),
        "aupr_unk": float(average_precision_score(1 - y, -s)),   # unknown = positive
        "aupr_known": float(average_precision_score(y, s)),      # known = positive
        "fpr95": fpr95,                                          # unknowns accepted at 95% known TPR
        "urr": float(np.mean(s_unknown < thr)),                  # unknown rejection rate at val threshold
        "tar": float(np.mean(s_known >= thr)),                   # known acceptance rate at val threshold
        "oscr": float("nan"),
    }
    if correct_known is not None:
        out["oscr"] = oscr(s_known, correct_known, s_unknown)
    return out
