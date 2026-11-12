#!/usr/bin/env python
"""Write the dataset-parameter table and the split-definition table (the
'reproducibility tables' of the paper) to docs/.

    python export_tables.py
"""
import os
import numpy as np
import pandas as pd

from pqd.generator import CLASS_NAMES, COMPONENTS, CYCLES, F0, FS, N, PARAMS
from pqd.splits import make_splits, split_names


def main(out="docs"):
    os.makedirs(out, exist_ok=True)

    rows = []
    for group, d in PARAMS.items():
        for k, v in d.items():
            v = v if isinstance(v, list) else f"[{v[0]:.4g}, {v[1]:.4g}]"
            rows.append({"disturbance": group, "parameter": k, "range / values": v})
    params = pd.DataFrame(rows)
    params.to_csv(os.path.join(out, "dataset_parameters.csv"), index=False)

    classes = pd.DataFrame({"id": range(len(CLASS_NAMES)), "class": CLASS_NAMES,
                            "components": [" + ".join(sorted(COMPONENTS[c])) or "-" for c in CLASS_NAMES]})
    classes.to_csv(os.path.join(out, "classes.csv"), index=False)

    sp = make_splits(3)
    srows = []
    for s in sp:
        k_names, u_names = split_names(s)
        rationale = ""
        if s["type"] == "random":
            rationale = "Random uniform selection of 4 disturbance classes held out; 'normal' always kept known."
        elif s["type"] == "near":
            rationale = "Compositional OOD: All held-out classes are multi-disturbance combinations whose individual primitives (sag, swell, harm, osc) are present in the known training classes."
        elif s["type"] == "far":
            rationale = "Novel-Component OOD: Held-out classes contain disturbance families (flicker, notch) completely absent from training."

        srows.append({
            "split": s["name"],
            "type": s["type"],
            "known classes (12)": ", ".join(k_names),
            "unknown classes (4)": ", ".join(u_names),
            "novel components": ", ".join(s["novel_components"]) or "none (compositional / component sharing)",
            "rationale / hardness": rationale
        })
    splits = pd.DataFrame(srows)
    splits.to_csv(os.path.join(out, "splits.csv"), index=False)

    with open(os.path.join(out, "dataset_and_splits.md"), "w") as f:
        f.write(f"# Synthetic Dataset & Open-Set Split Definitions\n\n"
                f"**Signal Architecture & Sampling:**\n"
                f"- Synthetic dataset inspired by IEEE 1159 disturbance models.\n"
                f"- Sampling frequency $f_s = {FS}$ Hz, nominal power system fundamental $f_0 = {F0:g}$ Hz (Indian 50 Hz grid standard).\n"
                f"- Window duration: {CYCLES} fundamental cycles $\\rightarrow N = {N}$ samples per window.\n"
                f"- Voltage units: Normalized per-unit (pu) with nominal fundamental amplitude $A = 1.0$ pu.\n"
                f"- Noise: Additive White Gaussian Noise (AWGN); calibrated against nominal sine power $P_{{\\text{{ref}}}} = 0.5\\text{{ pu}}^2$.\n\n"
                f"## 1. Class Definitions (16 Classes)\n\n" + classes.to_markdown(index=False) + "\n\n"
                f"## 2. Generator Parameter Ranges\n\n" + params.to_markdown(index=False) + "\n\n"
                f"## 3. Open-Set Benchmark Splits (12 Known / 4 Unknown)\n\n" + splits.to_markdown(index=False) + "\n")
    print(f"wrote {out}/dataset_and_splits.md (+ csv files)")


if __name__ == "__main__":
    main()
