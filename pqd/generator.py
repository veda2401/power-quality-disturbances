"""Synthetic PQD generator (IEEE 1159-inspired).

PARAMS is the single source of truth: the generator reads it and
export_tables.py writes it into the dataset-parameter table for the paper.

Signal model: 50 Hz system (India), fs = 3200 Hz, 10 cycles -> 640 samples.
All amplitudes are in per-unit (pu) of the nominal voltage.
"""
import numpy as np

FS = 3200                    # sampling rate (Hz)
F0 = 50.0                    # nominal frequency (Hz)
CYCLES = 10                  # cycles per window
SPC = int(round(FS / F0))    # samples per cycle (64)
N = SPC * CYCLES             # samples per window (640)

# (class name, components).  Components of combined classes are what makes
# "near" unknowns possible: a combined class built from known components.
_CLASSES = [
    ("normal",        ()),
    ("sag",           ("sag",)),
    ("swell",         ("swell",)),
    ("interruption",  ("interruption",)),
    ("harmonics",     ("harm",)),
    ("osc_transient", ("osc",)),
    ("notch",         ("notch",)),
    ("spike",         ("spike",)),
    ("flicker",       ("flicker",)),
    ("sag_harm",      ("sag", "harm")),
    ("swell_harm",    ("swell", "harm")),
    ("flicker_harm",  ("flicker", "harm")),
    ("sag_osc",       ("sag", "osc")),
    ("swell_osc",     ("swell", "osc")),
    ("flicker_sag",   ("flicker", "sag")),
    ("harm_osc",      ("harm", "osc")),
]
CLASS_NAMES = [c[0] for c in _CLASSES]
COMPONENTS = {c[0]: frozenset(c[1]) for c in _CLASSES}
NUM_CLASSES = len(CLASS_NAMES)

# Ranges are (low, high) and sampled uniformly unless stated otherwise.
PARAMS = {
    "fundamental": {
        "amplitude_pu": (0.98, 1.02),
        "freq_jitter_hz": (-0.1, 0.1),
        "phase_rad": (0.0, 2 * np.pi),
    },
    "sag": {
        "level_pu": (0.1, 0.9),          # remaining voltage during event
        "start_cycle": (0.5, 4.0),
        "duration_cycles": (1.5, 5.0),
    },
    "swell": {
        "level_pu": (1.1, 1.8),
        "start_cycle": (0.5, 4.0),
        "duration_cycles": (1.5, 5.0),
    },
    "interruption": {
        "level_pu": (0.0, 0.1),
        "start_cycle": (0.5, 4.0),
        "duration_cycles": (1.5, 5.0),
    },
    "harmonics": {
        "orders": [3, 5, 7, 9, 11],
        "n_orders": (2, 4),              # integers, inclusive
        "amplitude_pu": (0.03, 0.12),    # relative to fundamental, per order
    },
    "osc_transient": {
        "amplitude_pu": (0.2, 0.8),
        "freq_hz": (300.0, 900.0),
        "tau_s": (0.002, 0.008),         # decay time constant
        "start_s": (0.02, 0.14),
    },
    "notch": {
        "depth_pu": (0.2, 0.6),
        "width_samples": (1, 3),         # integers, inclusive
        "position_in_cycle_samples": (8, 56),  # integers, inclusive
    },
    "spike": {
        "n_spikes": (1, 3),              # integers, inclusive
        "amplitude_pu": (0.5, 2.0),      # sign is random
        "width_samples": (1, 2),         # integers, inclusive
    },
    "flicker": {
        "depth": (0.05, 0.2),
        "freq_hz": (5.0, 25.0),
    },
}

NOISE_REFERENCE_POWER = 0.5   # power of a 1 pu sine; SNR is defined w.r.t. this


def _u(rng, pair):
    return rng.uniform(pair[0], pair[1])


def _i(rng, pair):
    return int(rng.integers(pair[0], pair[1] + 1))


def generate_one(class_name, rng):
    """Generate one clean (noise-free) window of the given class."""
    comps = COMPONENTS[class_name]
    P = PARAMS
    t = np.arange(N) / FS
    A = _u(rng, P["fundamental"]["amplitude_pu"])
    f = F0 + _u(rng, P["fundamental"]["freq_jitter_hz"])
    phi = _u(rng, P["fundamental"]["phase_rad"])
    x = A * np.sin(2 * np.pi * f * t + phi)

    if "harm" in comps:
        hp = P["harmonics"]
        k = _i(rng, hp["n_orders"])
        orders = rng.choice(hp["orders"], size=k, replace=False)
        for h in orders:
            a = _u(rng, hp["amplitude_pu"])
            x = x + A * a * np.sin(2 * np.pi * h * f * t + _u(rng, (0.0, 2 * np.pi)))

    for key in ("sag", "swell", "interruption"):
        if key in comps:
            p = P[key]
            level = _u(rng, p["level_pu"])
            s = int(_u(rng, p["start_cycle"]) * SPC)
            e = min(N, s + int(_u(rng, p["duration_cycles"]) * SPC))
            env = np.ones(N)
            env[s:e] = level
            x = x * env

    if "flicker" in comps:
        p = P["flicker"]
        m = _u(rng, p["depth"])
        fm = _u(rng, p["freq_hz"])
        x = x * (1.0 + m * np.sin(2 * np.pi * fm * t + _u(rng, (0.0, 2 * np.pi))))

    if "osc" in comps:
        p = P["osc_transient"]
        B = _u(rng, p["amplitude_pu"])
        fn = _u(rng, p["freq_hz"])
        tau = _u(rng, p["tau_s"])
        t0 = _u(rng, p["start_s"])
        m = t >= t0
        x[m] = x[m] + B * np.exp(-(t[m] - t0) / tau) * np.sin(2 * np.pi * fn * (t[m] - t0))

    if "notch" in comps:
        p = P["notch"]
        d = _u(rng, p["depth_pu"])
        w = _i(rng, p["width_samples"])
        p0 = _i(rng, p["position_in_cycle_samples"])
        for c in range(CYCLES):
            sl = slice(p0 + c * SPC, min(N, p0 + c * SPC + w))
            x[sl] = x[sl] - d * A * np.sign(x[sl])

    if "spike" in comps:
        p = P["spike"]
        for _ in range(_i(rng, p["n_spikes"])):
            pos = int(rng.integers(0, N - 2))
            w = _i(rng, p["width_samples"])
            amp = _u(rng, p["amplitude_pu"]) * (1 if rng.random() < 0.5 else -1)
            x[pos:pos + w] = x[pos:pos + w] + amp * A

    return x.astype(np.float32)


def add_noise(X, snr_db, rng):
    """Add white Gaussian noise.  SNR is defined against a nominal 1 pu sine
    (power 0.5) so that the noise level does not change with the disturbance."""
    sigma = np.sqrt(NOISE_REFERENCE_POWER / (10.0 ** (snr_db / 10.0)))
    return (X + rng.normal(0.0, sigma, X.shape)).astype(np.float32)
