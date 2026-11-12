#!/usr/bin/env python
"""Generate comprehensive signal, waveform, spectrum, and split visualization graphs."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from pqd.generator import (
    CLASS_NAMES, COMPONENTS, PARAMS, FS, F0, N, CYCLES, SPC,
    generate_one, add_noise
)
from pqd.splits import make_splits, split_names

OUTPUT_DIR = "figures/signals_and_waveforms"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set clean aesthetic styling
plt.rcParams.update({
    "font.sans-serif": "DejaVu Sans",
    "font.family": "sans-serif",
    "figure.titlesize": 14,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300
})

def plot_all_16_waveforms():
    """1. High-resolution 4x4 grid of all 16 PQD classes."""
    fig, axes = plt.subplots(4, 4, figsize=(16, 11), sharex=True, sharey=True)
    t_ms = (np.arange(N) / FS) * 1000  # in ms (0 to 200 ms for 10 cycles)
    rng = np.random.default_rng(42)
    
    # Human-friendly class titles
    titles = {
        "normal": "1. Normal Sinusoid",
        "sag": "2. Voltage Sag",
        "swell": "3. Voltage Swell",
        "interruption": "4. Interruption",
        "harmonics": "5. Harmonics (3, 5, 7...)",
        "osc_transient": "6. Oscillatory Transient",
        "notch": "7. Voltage Notch",
        "spike": "8. Impulsive Spike",
        "flicker": "9. Voltage Flicker",
        "sag_harm": "10. Sag + Harmonics",
        "swell_harm": "11. Swell + Harmonics",
        "flicker_harm": "12. Flicker + Harmonics",
        "sag_osc": "13. Sag + Osc. Transient",
        "swell_osc": "14. Swell + Osc. Transient",
        "flicker_sag": "15. Flicker + Sag",
        "harm_osc": "16. Harmonics + Osc. Transient"
    }

    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf", "#3366cc", "#dc3912",
        "#ff9900", "#109618", "#990099", "#0099c6"
    ]

    for idx, (cls_name, ax) in enumerate(zip(CLASS_NAMES, axes.flat)):
        x = generate_one(cls_name, rng)
        ax.plot(t_ms, x, color=colors[idx], lw=1.2, label="Signal")
        ax.set_title(titles.get(cls_name, cls_name), fontsize=10, fontweight="bold", pad=4)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.axhline(0, color="black", lw=0.5, alpha=0.5)
        ax.set_ylim(-2.5, 2.5)
        if idx % 4 == 0:
            ax.set_ylabel("Amplitude (pu)")
        if idx >= 12:
            ax.set_xlabel("Time (ms)")

    fig.suptitle("Synthetic Power Quality Disturbance (PQD) Dataset: All 16 Disturbance Classes\n(10 Cycles @ 50 Hz, $f_s = 3200$ Hz, $N=640$ samples)", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    path = os.path.join(OUTPUT_DIR, "01_all_16_pqd_waveforms_clean.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_noise_levels():
    """2. Compare SNR levels (Clean, 40 dB, 30 dB, 20 dB, 10 dB) across representative classes."""
    selected_classes = ["normal", "sag_harm", "osc_transient", "flicker", "spike"]
    snrs = [float("inf"), 40.0, 30.0, 20.0, 10.0]
    snr_labels = ["Clean ($\infty$ dB)", "40 dB SNR", "30 dB SNR", "20 dB SNR", "10 dB SNR"]
    
    t_ms = (np.arange(N) / FS) * 1000
    rng = np.random.default_rng(101)
    
    fig, axes = plt.subplots(len(selected_classes), len(snrs), figsize=(17, 10), sharex=True, sharey=True)
    
    for r_idx, cls in enumerate(selected_classes):
        base_x = generate_one(cls, rng)
        for c_idx, snr in enumerate(snrs):
            ax = axes[r_idx, c_idx]
            if np.isinf(snr):
                x_noisy = base_x
            else:
                x_noisy = add_noise(base_x[np.newaxis, :], snr, rng)[0]
            
            color = "#1f77b4" if np.isinf(snr) else ("#2ca02c" if snr >= 30 else ("#ff7f0e" if snr >= 20 else "#d62728"))
            ax.plot(t_ms, x_noisy, color=color, lw=1.0)
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.set_ylim(-2.6, 2.6)
            
            if r_idx == 0:
                ax.set_title(snr_labels[c_idx], fontsize=11, fontweight="bold")
            if c_idx == 0:
                ax.set_ylabel(f"{cls.replace('_', ' ').title()}\n(pu)", fontweight="bold")
            if r_idx == len(selected_classes) - 1:
                ax.set_xlabel("Time (ms)")

    fig.suptitle("Additive White Gaussian Noise (AWGN) Robustness Across SNR Regimes\n(Indian Grid: 50 Hz fundamental, noise power calibrated to 1.0 pu sine)", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    path = os.path.join(OUTPUT_DIR, "02_noise_snr_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_frequency_spectra():
    """3. Fast Fourier Transform (FFT) Power Spectra for key classes."""
    fig, axes = plt.subplots(4, 4, figsize=(16, 11), sharex=True)
    freqs = np.fft.rfftfreq(N, 1.0 / FS)
    rng = np.random.default_rng(202)

    for idx, (cls_name, ax) in enumerate(zip(CLASS_NAMES, axes.flat)):
        x = generate_one(cls_name, rng)
        # Apply Hanning window and compute magnitude spectrum
        win = np.hanning(N)
        X_fft = np.abs(np.fft.rfft(x * win)) / (N / 2)
        
        ax.plot(freqs, X_fft, color="#2b5c8f", lw=1.2)
        ax.set_title(cls_name.replace("_", " ").title(), fontsize=10, fontweight="bold")
        ax.set_xlim(0, 1000)  # Show up to 1000 Hz
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.axvline(50, color="red", linestyle=":", alpha=0.6, label="50Hz Fundamental" if idx==0 else "")
        if "harm" in cls_name or cls_name == "harmonics":
            ax.set_facecolor("#fcf8e3")  # subtle highlight for harmonic classes
        if idx % 4 == 0:
            ax.set_ylabel("Magnitude (pu)")
        if idx >= 12:
            ax.set_xlabel("Frequency (Hz)")

    fig.suptitle("Single-Sided Frequency Spectra (FFT) of 16 PQD Classes\n(Revealing Fundamental @ 50 Hz, Odd Harmonics @ 150/250/350 Hz, and Transient Bands)", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    path = os.path.join(OUTPUT_DIR, "03_frequency_spectra_fft.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_spectrograms():
    """4. Time-Frequency STFT Spectrograms for transient and dynamic disturbances."""
    sample_classes = ["normal", "harmonics", "osc_transient", "notch", "spike", "flicker", "sag_harm", "swell_osc"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True, sharey=True)
    rng = np.random.default_rng(777)

    for idx, (cls, ax) in enumerate(zip(sample_classes, axes.flat)):
        x = generate_one(cls, rng)
        f, t, Sxx = signal.spectrogram(x, fs=FS, nperseg=64, noverlap=56, nfft=256, window='hann')
        t_ms = t * 1000
        
        im = ax.pcolormesh(t_ms, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='inferno')
        ax.set_title(cls.replace("_", " ").title(), fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1000)
        ax.grid(False)
        if idx % 4 == 0:
            ax.set_ylabel("Frequency (Hz)")
        if idx >= 4:
            ax.set_xlabel("Time (ms)")

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Power Spectral Density (dB/Hz)", fontsize=10)

    fig.suptitle("Short-Time Fourier Transform (STFT) Spectrograms for Selected PQD Events", fontsize=14, fontweight="bold", y=0.98)
    path = os.path.join(OUTPUT_DIR, "04_time_frequency_spectrograms.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_split_composition():
    """5. Visual heatmap showing Known vs Unknown classes in the 5 benchmark splits."""
    sp = make_splits(3, n_unknown=4)
    split_labels = [f"Random Split 0", "Random Split 1", "Random Split 2", "Near-OOD Split", "Far-OOD Split"]
    
    matrix = np.zeros((len(sp), len(CLASS_NAMES)))
    for row_idx, s in enumerate(sp):
        known_idx, unk_idx = s["known"], s["unknown"]
        for k in known_idx:
            matrix[row_idx, k] = 1  # Known = 1 (Green)
        for u in unk_idx:
            matrix[row_idx, u] = 0  # Unknown = 0 (Red/Orange)

    fig, ax = plt.subplots(figsize=(14, 5.5))
    cmap = matplotlib.colors.ListedColormap(['#d9534f', '#5cb85c'])
    bounds = [-0.5, 0.5, 1.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect='auto')
    ax.set_xticks(np.arange(len(CLASS_NAMES)))
    ax.set_xticklabels([c.replace('_', '\n') for c in CLASS_NAMES], fontsize=8, fontweight='bold')
    ax.set_yticks(np.arange(len(sp)))
    ax.set_yticklabels(split_labels, fontsize=10, fontweight='bold')

    # Draw grid
    ax.set_xticks(np.arange(-.5, len(CLASS_NAMES), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(sp), 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)

    # Annotate cells
    for i in range(len(sp)):
        for j in range(len(CLASS_NAMES)):
            status = "KNOWN" if matrix[i, j] == 1 else "UNKNOWN"
            color = "white"
            ax.text(j, i, status, ha="center", va="center", color=color, fontsize=7.5, fontweight="bold")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#5cb85c', edgecolor='black', label='Known (Training & In-Distribution Closed-Set)'),
        Patch(facecolor='#d9534f', edgecolor='black', label='Unknown (Held-Out Out-of-Distribution / Open-Set)')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=10)

    ax.set_title("Open-Set Benchmark Evaluation Splits (12 Known vs 4 Unknown Classes per Split)\nRandom Splits vs Semantic Near-OOD (Component Overlap) vs Far-OOD (Novel Disturbance Families)", fontsize=12, fontweight="bold", pad=12)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "05_split_composition_matrix.png")
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    print("Generating comprehensive signal and waveform graphs...")
    plot_all_16_waveforms()
    plot_noise_levels()
    plot_frequency_spectra()
    plot_spectrograms()
    plot_split_composition()
    print("All signal graphs generated successfully!")
