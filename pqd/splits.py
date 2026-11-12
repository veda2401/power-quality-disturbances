"""Known / unknown class splits.

Split types
  random : unknown classes drawn at random (several draws)
  near   : unknown classes are combinations of components that are all known
           (e.g. sag+harmonics when sag and harmonics are both known)
  far    : unknown classes contain components that never appear in the known
           classes (flicker family and notch)

'normal' is always known.  `novel_components` records which disturbance
components exist only in the unknown classes (empty for a purely 'near' split).
"""
import numpy as np
from .generator import CLASS_NAMES, COMPONENTS, NUM_CLASSES

NEAR_UNKNOWN = ["sag_harm", "swell_harm", "sag_osc", "swell_osc"]
FAR_UNKNOWN = ["flicker", "flicker_harm", "flicker_sag", "notch"]


def _ids(names):
    return [CLASS_NAMES.index(n) for n in names]


def _make(name, split_type, unknown_ids):
    unknown = sorted(int(i) for i in unknown_ids)
    known = [i for i in range(NUM_CLASSES) if i not in unknown]
    kc = set().union(*[COMPONENTS[CLASS_NAMES[i]] for i in known])
    uc = set().union(*[COMPONENTS[CLASS_NAMES[i]] for i in unknown])
    return {
        "name": name,
        "type": split_type,
        "known": known,
        "unknown": unknown,
        "novel_components": sorted(uc - kc),
    }


def make_splits(n_random=3, n_unknown=4, base_seed=1234, include_near_far=True):
    splits = []
    non_normal = [i for i in range(NUM_CLASSES) if CLASS_NAMES[i] != "normal"]
    for r in range(n_random):
        rng = np.random.default_rng(base_seed + r)
        unk = rng.choice(non_normal, size=n_unknown, replace=False)
        splits.append(_make(f"random{r}", "random", unk))
    if include_near_far:
        splits.append(_make("near", "near", _ids(NEAR_UNKNOWN)))
        splits.append(_make("far", "far", _ids(FAR_UNKNOWN)))
    for s in splits:
        check_split(s)
    return splits


def check_split(s):
    assert not set(s["known"]) & set(s["unknown"]), "known/unknown overlap"
    assert CLASS_NAMES.index("normal") in s["known"], "'normal' must be known"
    assert len(s["known"]) + len(s["unknown"]) == NUM_CLASSES


def split_names(s):
    return ([CLASS_NAMES[i] for i in s["known"]], [CLASS_NAMES[i] for i in s["unknown"]])
