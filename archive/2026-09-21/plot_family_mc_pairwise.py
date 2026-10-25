"""
Same style as plot_bo_v2_search.py's pairwise plot (Q6/Q2/Q5 projections
colored by loss), but from fp2_family_mc_results.csv -- lets us check
whether the low-loss region the MC's E2 stage is finding lines up with
where the earlier BO search found its low-loss region.

Only rows with an actual computed loss (i.e. that reached the E2 stage --
norm_rejected rows never ran COSY and have no loss) are plotted.

Run:
    cd AgentSearch
    python3 plot_family_mc_pairwise.py
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_FILE = "Results/fp2_family_mc_results.csv"
OUTPUT_DIR = "Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CHI2_THRESHOLD_2SIGMA_DF4 = 9.715641142541578

rows = []
with open(RESULTS_FILE, "r", newline="") as f:
    for row in csv.DictReader(f):
        if row["loss"] in (None, ""):
            continue
        rows.append(row)

rows = [{k: (float(v) if k not in ("status",) and v not in (None, "") else v) for k, v in r.items()} for r in rows]
losses = np.array([r["loss"] for r in rows])
best = rows[int(np.argmin(losses))]
n_pass = int(np.sum(losses <= CHI2_THRESHOLD_2SIGMA_DF4))
print(f"{len(rows)} points with a computed loss ({n_pass} actually pass E2 <= {CHI2_THRESHOLD_2SIGMA_DF4:.2f}). "
      f"Best: loss={best['loss']:.2f} at "
      f"Q6={best['Q6_B_SC']*100:+.2f}% Q2={best['Q2_B_SC']*100:+.2f}% Q5={best['Q5_B_SC']*100:+.2f}%")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
pairs = [("Q6_B_SC", "Q2_B_SC"), ("Q6_B_SC", "Q5_B_SC"), ("Q2_B_SC", "Q5_B_SC")]
vmax = CHI2_THRESHOLD_2SIGMA_DF4  # real E2 acceptance threshold, not a percentile
passing = losses <= CHI2_THRESHOLD_2SIGMA_DF4
for ax, (xk, yk) in zip(axes, pairs):
    x = np.array([r[xk] for r in rows]) * 100
    y = np.array([r[yk] for r in rows]) * 100
    sc = ax.scatter(x, y, c=losses, cmap="viridis_r", vmin=losses.min(), vmax=vmax, s=25,
                     edgecolors="none")
    # outline the points that actually pass E2 so they're unambiguous, not just "relatively good"
    ax.scatter(x[passing], y[passing], facecolors="none", edgecolors="black", linewidths=1.0, s=25)
    ax.plot(best[xk] * 100, best[yk] * 100, "r*", markersize=20, markeredgecolor="white", markeredgewidth=0.8)
    ax.set_xlabel(f"{xk} [%]")
    ax.set_ylabel(f"{yk} [%]")
    fig.colorbar(sc, ax=ax, label=f"loss (capped at real E2 threshold {vmax:.2f})")
fig.suptitle(f"Family MC-evaluated points (E2 stage), pairwise projections\n"
             f"black-outlined = actually passes E2 ({n_pass}/{len(rows)}); red star = best found")
fig.tight_layout()
outfile = os.path.join(OUTPUT_DIR, "family_mc_pairwise.png")
fig.savefig(outfile, dpi=150)
print(f"wrote {outfile}")
