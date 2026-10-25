"""
Combined corner plot: v1's 45 E2-passing candidates (7-knob search) and
v7's 46 E2-passing candidates (12-knob search, weighted+regularized
solve), both shown on the same 7-knob axes (Q1/Q2/Q4/Q5/Q6/Q7 B_SC +
WF1 Exit dB) -- both are "good" solutions under the same 2-sigma
(chi2_df4=9.72) loss threshold, per the user's request 2026-08-14.
Different marker shapes distinguish the two runs (circle=v1,
triangle=v7); both colored by their own loss on a shared scale.

Run:
    cd AgentSearch
    python3 plot_family_mc_corner_v1_v7_combined.py
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

V1_FILE = "Results/fp2_family_mc_results.csv"
V7_FILE = "Results/fp2_family_mc_v7_results.csv"
OUTPUT_DIR = "Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CHI2_THRESHOLD_2SIGMA_DF4 = 9.715641142541578
KNOBS = ['Q1_B_SC', 'Q2_B_SC', 'Q4_B_SC', 'Q5_B_SC', 'Q6_B_SC', 'Q7_B_SC', 'WF1_Exit_dB']
LABELS = ['Q1', 'Q2', 'Q4', 'Q5', 'Q6', 'Q7', 'WF1 dB']


def load_passing(path):
    rows = list(csv.DictReader(open(path)))
    passing = [r for r in rows if r["loss"] not in (None, "") and float(r["loss"]) <= CHI2_THRESHOLD_2SIGMA_DF4]
    return [{k: float(r[k]) for k in KNOBS + ["loss"]} for r in passing]


v1 = load_passing(V1_FILE)
v7 = load_passing(V7_FILE)
print(f"v1: {len(v1)} E2-passing candidates.  v7: {len(v7)} E2-passing candidates.")

data_v1 = {k: np.array([r[k] for r in v1]) * 100 for k in KNOBS}
data_v7 = {k: np.array([r[k] for r in v7]) * 100 for k in KNOBS}
losses_v1 = np.array([r["loss"] for r in v1])
losses_v7 = np.array([r["loss"] for r in v7])

all_losses = np.concatenate([losses_v1, losses_v7])
vmin, vmax = all_losses.min(), CHI2_THRESHOLD_2SIGMA_DF4

global_best_is_v7 = losses_v7.min() < losses_v1.min()
if global_best_is_v7:
    best_idx, best_data, best_loss = int(np.argmin(losses_v7)), data_v7, losses_v7.min()
else:
    best_idx, best_data, best_loss = int(np.argmin(losses_v1)), data_v1, losses_v1.min()
print(f"Overall best: loss={best_loss:.3f} ({'v7' if global_best_is_v7 else 'v1'})")

n = len(KNOBS)
fig, axes = plt.subplots(n, n, figsize=(2.2 * n, 2.2 * n))

for i in range(n):
    for j in range(n):
        ax = axes[i, j]
        if j > i:
            ax.axis("off")
            continue
        if i == j:
            bins = np.linspace(min(data_v1[KNOBS[i]].min(), data_v7[KNOBS[i]].min()),
                                max(data_v1[KNOBS[i]].max(), data_v7[KNOBS[i]].max()), 16)
            ax.hist(data_v1[KNOBS[i]], bins=bins, color="tab:blue", alpha=0.5, label="v1 (7-knob)")
            ax.hist(data_v7[KNOBS[i]], bins=bins, color="tab:orange", alpha=0.5, label="v7 (12-knob)")
            ax.axvline(best_data[KNOBS[i]][best_idx], color="red", linestyle="--", linewidth=1.2)
            if i == 0:
                ax.legend(fontsize=6, loc="upper right")
        else:
            sc1 = ax.scatter(data_v1[KNOBS[j]], data_v1[KNOBS[i]], c=losses_v1, cmap="viridis_r",
                              vmin=vmin, vmax=vmax, s=16, marker="o", edgecolors="tab:blue", linewidths=0.5)
            sc2 = ax.scatter(data_v7[KNOBS[j]], data_v7[KNOBS[i]], c=losses_v7, cmap="viridis_r",
                              vmin=vmin, vmax=vmax, s=20, marker="^", edgecolors="tab:orange", linewidths=0.5)
            ax.plot(best_data[KNOBS[j]][best_idx], best_data[KNOBS[i]][best_idx], "r*",
                    markersize=14, markeredgecolor="white", markeredgewidth=0.6)
        if i == n - 1:
            ax.set_xlabel(LABELS[j])
        else:
            ax.set_xticklabels([])
        if j == 0:
            ax.set_ylabel(LABELS[i])
        if j != 0:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=7)

fig.colorbar(sc2, ax=axes, location="right", shrink=0.6, label=f"loss (capped at {vmax:.2f})", pad=0.02)
fig.suptitle(f"v1 (circle, {len(v1)} candidates) + v7 (triangle, {len(v7)} candidates), same 7 knobs [%]\n"
             f"both E2-passing under the same 2-sigma threshold; red star = overall best", fontsize=12)
outfile = os.path.join(OUTPUT_DIR, "family_mc_corner_v1_v7_combined.png")
fig.savefig(outfile, dpi=150, bbox_inches="tight")
print(f"wrote {outfile}")
