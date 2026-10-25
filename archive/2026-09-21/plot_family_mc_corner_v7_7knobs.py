"""
Same corner plot as plot_family_mc_corner.py, but from the v7 (12-knob,
weighted-solve) family search results, showing ONLY the original 7
knobs (Q1/Q2/Q4/Q5/Q6/Q7 B_SC + WF1 Exit dB) for direct visual
comparison against the v1 reference plot -- per the user's request
2026-08-14. Q3 and B1-B4 Exit dB values exist in the v7 data but are
simply not plotted here (this is a like-for-like comparison, not a
claim that they don't matter).

Run:
    cd AgentSearch
    python3 plot_family_mc_corner_v7_7knobs.py
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_FILE = "Results/fp2_family_mc_v7_results.csv"
OUTPUT_DIR = "Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CHI2_THRESHOLD_2SIGMA_DF4 = 9.715641142541578
KNOBS = ['Q1_B_SC', 'Q2_B_SC', 'Q4_B_SC', 'Q5_B_SC', 'Q6_B_SC', 'Q7_B_SC', 'WF1_Exit_dB']
LABELS = ['Q1', 'Q2', 'Q4', 'Q5', 'Q6', 'Q7', 'WF1 dB']

rows = list(csv.DictReader(open(RESULTS_FILE)))
passing = [r for r in rows if r["loss"] not in (None, "") and float(r["loss"]) <= CHI2_THRESHOLD_2SIGMA_DF4]
passing = [{k: float(r[k]) for k in KNOBS + ["loss"]} for r in passing]
print(f"{len(passing)} E2-passing candidates (v7, 12-knob search, showing 7 original knobs).")

data = {k: np.array([r[k] for r in passing]) * 100 for k in KNOBS}
losses = np.array([r["loss"] for r in passing])
best_idx = int(np.argmin(losses))
print(f"Best: loss={losses[best_idx]:.3f}")

n = len(KNOBS)
fig, axes = plt.subplots(n, n, figsize=(2.2 * n, 2.2 * n))
vmax = CHI2_THRESHOLD_2SIGMA_DF4

for i in range(n):
    for j in range(n):
        ax = axes[i, j]
        if j > i:
            ax.axis("off")
            continue
        if i == j:
            ax.hist(data[KNOBS[i]], bins=12, color="tab:green", edgecolor="black", alpha=0.8)
            ax.axvline(data[KNOBS[i]][best_idx], color="red", linestyle="--", linewidth=1.2)
        else:
            sc = ax.scatter(data[KNOBS[j]], data[KNOBS[i]], c=losses, cmap="viridis_r",
                             vmin=losses.min(), vmax=vmax, s=18)
            ax.plot(data[KNOBS[j]][best_idx], data[KNOBS[i]][best_idx], "r*",
                    markersize=14, markeredgecolor="white", markeredgewidth=0.6)
        if i == n - 1:
            ax.set_xlabel(LABELS[j])
        else:
            ax.set_xticklabels([])
        if j == 0 and i != 0:
            ax.set_ylabel(LABELS[i])
        elif j == 0 and i == 0:
            ax.set_ylabel(LABELS[i])
        if j != 0:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=7)

fig.colorbar(sc, ax=axes, location="right", shrink=0.6, label=f"loss (capped at {vmax:.2f})", pad=0.02)
fig.suptitle(f"v7 (12-knob search): {len(passing)} E2-passing candidates, 7 original knobs shown [%]\n"
             f"(red = best found; Q3/B1-B4 not shown here, for comparison against the v1 reference)", fontsize=12)
outfile = os.path.join(OUTPUT_DIR, "family_mc_corner_v7_7knobs.png")
fig.savefig(outfile, dpi=150, bbox_inches="tight")
print(f"wrote {outfile}")
