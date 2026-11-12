# Synthetic Dataset & Open-Set Split Definitions

**Signal Architecture & Sampling:**
- Synthetic dataset inspired by IEEE 1159 disturbance models.
- Sampling frequency $f_s = 3200$ Hz, nominal power system fundamental $f_0 = 50$ Hz (Indian 50 Hz grid standard).
- Window duration: 10 fundamental cycles $\rightarrow N = 640$ samples per window.
- Voltage units: Normalized per-unit (pu) with nominal fundamental amplitude $A = 1.0$ pu.
- Noise: Additive White Gaussian Noise (AWGN); calibrated against nominal sine power $P_{\text{ref}} = 0.5\text{ pu}^2$.

## 1. Class Definitions (16 Classes)

|   id | class         | components     |
|-----:|:--------------|:---------------|
|    0 | normal        | -              |
|    1 | sag           | sag            |
|    2 | swell         | swell          |
|    3 | interruption  | interruption   |
|    4 | harmonics     | harm           |
|    5 | osc_transient | osc            |
|    6 | notch         | notch          |
|    7 | spike         | spike          |
|    8 | flicker       | flicker        |
|    9 | sag_harm      | harm + sag     |
|   10 | swell_harm    | harm + swell   |
|   11 | flicker_harm  | flicker + harm |
|   12 | sag_osc       | osc + sag      |
|   13 | swell_osc     | osc + swell    |
|   14 | flicker_sag   | flicker + sag  |
|   15 | harm_osc      | harm + osc     |

## 2. Generator Parameter Ranges

| disturbance   | parameter                 | range / values   |
|:--------------|:--------------------------|:-----------------|
| fundamental   | amplitude_pu              | [0.98, 1.02]     |
| fundamental   | freq_jitter_hz            | [-0.1, 0.1]      |
| fundamental   | phase_rad                 | [0, 6.283]       |
| sag           | level_pu                  | [0.1, 0.9]       |
| sag           | start_cycle               | [0.5, 4]         |
| sag           | duration_cycles           | [1.5, 5]         |
| swell         | level_pu                  | [1.1, 1.8]       |
| swell         | start_cycle               | [0.5, 4]         |
| swell         | duration_cycles           | [1.5, 5]         |
| interruption  | level_pu                  | [0, 0.1]         |
| interruption  | start_cycle               | [0.5, 4]         |
| interruption  | duration_cycles           | [1.5, 5]         |
| harmonics     | orders                    | [3, 5, 7, 9, 11] |
| harmonics     | n_orders                  | [2, 4]           |
| harmonics     | amplitude_pu              | [0.03, 0.12]     |
| osc_transient | amplitude_pu              | [0.2, 0.8]       |
| osc_transient | freq_hz                   | [300, 900]       |
| osc_transient | tau_s                     | [0.002, 0.008]   |
| osc_transient | start_s                   | [0.02, 0.14]     |
| notch         | depth_pu                  | [0.2, 0.6]       |
| notch         | width_samples             | [1, 3]           |
| notch         | position_in_cycle_samples | [8, 56]          |
| spike         | n_spikes                  | [1, 3]           |
| spike         | amplitude_pu              | [0.5, 2]         |
| spike         | width_samples             | [1, 2]           |
| flicker       | depth                     | [0.05, 0.2]      |
| flicker       | freq_hz                   | [5, 25]          |

## 3. Open-Set Benchmark Splits (12 Known / 4 Unknown)

| split   | type   | known classes (12)                                                                                                       | unknown classes (4)                                | novel components                         | rationale / hardness                                                                                                                                                      |
|:--------|:-------|:-------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------|:-----------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| random0 | random | normal, sag, swell, interruption, harmonics, osc_transient, spike, flicker, sag_harm, swell_harm, flicker_harm, harm_osc | notch, sag_osc, swell_osc, flicker_sag             | notch                                    | Random uniform selection of 4 disturbance classes held out; 'normal' always kept known.                                                                                   |
| random1 | random | normal, sag, swell, harmonics, notch, spike, flicker, sag_harm, swell_harm, flicker_harm, sag_osc, swell_osc             | interruption, osc_transient, flicker_sag, harm_osc | interruption                             | Random uniform selection of 4 disturbance classes held out; 'normal' always kept known.                                                                                   |
| random2 | random | normal, sag, swell, interruption, osc_transient, notch, spike, flicker, swell_harm, flicker_harm, sag_osc, flicker_sag   | harmonics, sag_harm, swell_osc, harm_osc           | none (compositional / component sharing) | Random uniform selection of 4 disturbance classes held out; 'normal' always kept known.                                                                                   |
| near    | near   | normal, sag, swell, interruption, harmonics, osc_transient, notch, spike, flicker, flicker_harm, flicker_sag, harm_osc   | sag_harm, swell_harm, sag_osc, swell_osc           | none (compositional / component sharing) | Compositional OOD: All held-out classes are multi-disturbance combinations whose individual primitives (sag, swell, harm, osc) are present in the known training classes. |
| far     | far    | normal, sag, swell, interruption, harmonics, osc_transient, spike, sag_harm, swell_harm, sag_osc, swell_osc, harm_osc    | notch, flicker, flicker_harm, flicker_sag          | flicker, notch                           | Novel-Component OOD: Held-out classes contain disturbance families (flicker, notch) completely absent from training.                                                      |
