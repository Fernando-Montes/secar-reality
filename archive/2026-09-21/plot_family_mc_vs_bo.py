"""
Overlays the family MC search (fp2_family_mc_results.csv, E2-stage
points, colored by loss) on top of the original BO search
(fp2_bo_search_results.csv -- the v1/flawed-prior run, 381 points, shown
in gray in the background) for direct visual comparison, same Q6/Q2/Q5
pairwise projections and the same +-25% axis range as the original BO
plot (plot_bo_search.py's bo_pairwise.png), per the user's request
2026-08-11.

Run:
    cd AgentSearch
    python3 plot_family_mc_vs_bo.py
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

BO_RESULTS_FILE = "Results/fp2_bo_search_results.csv"
MC_RESULTS_FILE = "Results/fp2_family_mc_results.csv"
OUTPUT_DIR = "Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
AXIS_RANGE = 25  # +-25%, matches BO's search BOUNDS

bo_rows = list(csv.DictReader(open(BO_RESULTS_FILE)))
bo_rows = [{k: (float(v) if k != "source" else v) for k, v in r.items()} for r in bo_rows]

mc_rows = []
with open(MC_RESULTS_FILE, "r", newline="") as f:
    for row in csv.DictReader(f):
        if row["loss"] in (None, ""):
            continue
        mc_rows.append(row)
mc_rows = [{k: (float(v) if k not in ("status",) and v not in (None, "") else v) for k, v in r.items()} for r in mc_rows]

mc_losses = np.array([r["loss"] for r in mc_rows])
mc_best = mc_rows[int(np.argmin(mc_losses))]
bo_losses = np.array([r["loss"] for r in bo_rows])
bo_best = bo_rows[int(np.argmin(bo_losses))]

print(f"BO: {len(bo_rows)} points, best loss={bo_best['loss']:.2f} at "
      f"Q6={bo_best['Q6_B_SC']*100:+.2f}% Q2={bo_best['Q2_B_SC']*100:+.2f}% Q5={bo_best['Q5_B_SC']*100:+.2f}%")
print(f"MC: {len(mc_rows)} points, best loss={mc_best['loss']:.2f} at "
      f"Q6={mc_best['Q6_B_SC']*100:+.2f}% Q2={mc_best['Q2_B_SC']*100:+.2f}% Q5={mc_best['Q5_B_SC']*100:+.2f}%")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
pairs = [("Q6_B_SC", "Q2_B_SC"), ("Q6_B_SC", "Q5_B_SC"), ("Q2_B_SC", "Q5_B_SC")]
CHI2_THRESHOLD_2SIGMA_DF4 = 9.715641142541578
vmax = CHI2_THRESHOLD_2SIGMA_DF4  # real E2 acceptance threshold, not a percentile
mc_passing = mc_losses <= CHI2_THRESHOLD_2SIGMA_DF4
# BO's loss spans 4 orders of magnitude (17 to 228,000) -- a linear scale would
# make almost everything black except the very best handful of points, so use
# a log norm instead: lower loss -> whiter, higher loss -> darker gray, with
# useful gradation across the whole range rather than just the extreme tail.
bo_norm = LogNorm(vmin=bo_losses.min(), vmax=bo_losses.max())
for ax, (xk, yk) in zip(axes, pairs):
    bx = np.array([r[xk] for r in bo_rows]) * 100
    by = np.array([r[yk] for r in bo_rows]) * 100
    ax.scatter(bx, by, c=bo_losses, cmap="Greys", norm=bo_norm,
               s=20, alpha=0.8, zorder=1, label="BO (v1)")

    mx = np.array([r[xk] for r in mc_rows]) * 100
    my = np.array([r[yk] for r in mc_rows]) * 100
    sc = ax.scatter(mx, my, c=mc_losses, cmap="viridis_r", vmin=mc_losses.min(), vmax=vmax, s=25, zorder=2,
                     edgecolors="none", label="Family MC")
    ax.scatter(mx[mc_passing], my[mc_passing], facecolors="none", edgecolors="black",
               linewidths=1.0, s=25, zorder=3)

    ax.plot(bo_best[xk] * 100, bo_best[yk] * 100, "b*", markersize=18,
            markeredgecolor="white", markeredgewidth=0.8, zorder=3, label="BO best")
    ax.plot(mc_best[xk] * 100, mc_best[yk] * 100, "r*", markersize=18,
            markeredgecolor="white", markeredgewidth=0.8, zorder=3, label="MC best")

    ax.set_xlim(-AXIS_RANGE, AXIS_RANGE)
    ax.set_ylim(-AXIS_RANGE, AXIS_RANGE)
    ax.set_xlabel(f"{xk} [%]")
    ax.set_ylabel(f"{yk} [%]")
    fig.colorbar(sc, ax=ax, label=f"MC loss (capped at real E2 threshold {vmax:.2f})")

axes[0].legend(fontsize=7, loc="upper right")
fig.suptitle("Family MC (color) overlaid on BO v1 search (gray) -- same Q6/Q2/Q5 axes/range")
fig.tight_layout()
outfile = os.path.join(OUTPUT_DIR, "family_mc_vs_bo_pairwise.png")
fig.savefig(outfile, dpi=150)
print(f"wrote {outfile}")
