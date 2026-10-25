"""
Corner plot of the 7 FP3 knobs (Q8-11 B_SC, B5/B6/WF2 Exit dB) across the
51 E2-passing candidates from fp3_family_mc.py -- same pairwise
scatter+histogram style as plot_family_mc_corner_v1_v7_full.py, but for
a single unified FP3 search (no v1/v7 split needed here). Colored by
loss; candidates within the tighter 1-sigma (df=4) chi2<=4.7196 outlined
in black, matching the convention used throughout the FP2 work.

Run:
    cd AgentSearch
    python3 plot_fp3_family_mc_corner.py
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_FILE = "Results/fp3_family_mc_results.csv"
OUTPUT_DIR = "Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CHI2_1SIGMA_DF4 = 4.7196
CHI2_2SIGMA_DF4 = 9.715641142541578

FP3_KNOBS = ['Q8_B_SC', 'Q9_B_SC', 'Q10_B_SC', 'Q11_B_SC', 'B5_Exit_dB', 'B6_Exit_dB', 'WF2_Exit_dB']
LABELS = ['Q8', 'Q9', 'Q10', 'Q11', 'B5', 'B6', 'WF2']

# User's manually-derived best fit (Q6-2.7%, WF1+0.25%, TA_Y+0.1mm are FP2-side/
# not part of this 7-knob FP3 axis set, so only the FP3 knobs it touched are
# shown here; Q8/Q10/B5/B6 are exactly 0 -- not part of the manual solution,
# not "found to be 0").
MANUAL_BEST = {'Q8_B_SC': 0.0, 'Q9_B_SC': 0.0949, 'Q10_B_SC': 0.0, 'Q11_B_SC': -0.0067,
               'B5_Exit_dB': 0.0, 'B6_Exit_dB': 0.0, 'WF2_Exit_dB': -0.00077}
manual_vals = np.array([MANUAL_BEST[k] * 100 for k in FP3_KNOBS])

rows = list(csv.DictReader(open(RESULTS_FILE)))
passing = [r for r in rows if r["status"] == "e2_passed"]
print(f"{len(passing)} E2-passing candidates")

data = np.array([[float(r[k]) * 100 for k in FP3_KNOBS] for r in passing])  # -> %
losses = np.array([float(r["loss"]) for r in passing])
within_1sigma = losses <= CHI2_1SIGMA_DF4
best_idx = np.argmin(losses)
print(f"{within_1sigma.sum()}/{len(passing)} also within 1-sigma (chi2<={CHI2_1SIGMA_DF4})")
print(f"Best: loss={losses[best_idx]:.3f}")

n = len(FP3_KNOBS)
fig, axes = plt.subplots(n, n, figsize=(16, 16))
norm = plt.Normalize(vmin=losses.min(), vmax=min(losses.max(), CHI2_2SIGMA_DF4))
cmap = plt.colormaps["viridis_r"]

for i in range(n):
    for j in range(n):
        ax = axes[i, j]
        if j > i:
            ax.axis("off")
            continue
        if i == j:
            ax.hist(data[:, i], bins=15, color="tab:blue", alpha=0.7)
            ax.axvline(data[best_idx, i], color="red", linestyle="--", linewidth=1)
            ax.axvline(manual_vals[i], color="tab:green", linestyle="--", linewidth=1.5)
        else:
            colors = cmap(norm(losses))
            edge = np.where(within_1sigma, "black", "none")
            ax.scatter(data[:, j], data[:, i], c=colors, s=25, edgecolors=edge, linewidths=0.8)
            ax.scatter(data[best_idx, j], data[best_idx, i], marker="*", s=200, color="red",
                       edgecolors="black", zorder=5)
            ax.scatter(manual_vals[j], manual_vals[i], marker="D", s=90, color="tab:green",
                       edgecolors="black", zorder=6)
        if i == n - 1:
            ax.set_xlabel(LABELS[j])
        if j == 0 and i != 0:
            ax.set_ylabel(LABELS[i])
        elif j == 0 and i == 0:
            ax.set_ylabel(LABELS[i])

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
fig.colorbar(sm, cax=cbar_ax, label=f"loss (capped at {CHI2_2SIGMA_DF4:.2f})")

fig.suptitle(f"FP3 family: {len(passing)} E2-passing candidates, all 7 knobs [%]\n"
             f"black outline = also within 1-sigma ({within_1sigma.sum()}/{len(passing)}); "
             f"red star = best; green diamond = manual best fit",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 0.91, 0.97])
outfile = os.path.join(OUTPUT_DIR, "fp3_family_mc_corner.png")
fig.savefig(outfile, dpi=150)
print(f"\nwrote {outfile}")
