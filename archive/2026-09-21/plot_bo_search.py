"""
Diagnostics for fp2_bo_search_results.csv:
  1. bo_convergence.png   -- running-best loss vs. evaluation index
  2. bo_pairwise.png      -- Q6/Q2/Q5 pairwise scatter, colored by loss,
                             best point marked
Run: python3 plot_bo_search.py
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_FILE = "Results/fp2_bo_search_results.csv"
OUTPUT_DIR = "Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

rows = list(csv.DictReader(open(RESULTS_FILE)))
rows = [{k: (float(v) if k != "source" else v) for k, v in r.items()} for r in rows]

# Order: warm start first (as originally appended), then BO points in the
# order they were evaluated (append order == evaluation order in the file).
losses = np.array([r["loss"] for r in rows])
running_best = np.minimum.accumulate(losses)
best = rows[int(np.argmin(losses))]

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(range(len(losses)), losses, "o", markersize=3, alpha=0.4, label="each evaluation")
ax.plot(range(len(losses)), running_best, "-", color="tab:red", linewidth=2, label="running best")
ax.axvline(30, color="gray", linestyle=":", label="end of OAT warm start")
ax.set_yscale("log")
ax.set_xlabel("evaluation index (warm start, then BO batches)")
ax.set_ylabel("loss (log scale)")
ax.set_title(f"BO convergence -- best loss={best['loss']:.2f} at "
             f"Q6={best['Q6_B_SC']*100:+.2f}% Q2={best['Q2_B_SC']*100:+.2f}% Q5={best['Q5_B_SC']*100:+.2f}%")
ax.legend()
fig.tight_layout()
outfile = os.path.join(OUTPUT_DIR, "bo_convergence.png")
fig.savefig(outfile, dpi=150)
print(f"wrote {outfile}")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
pairs = [("Q6_B_SC", "Q2_B_SC"), ("Q6_B_SC", "Q5_B_SC"), ("Q2_B_SC", "Q5_B_SC")]
# Cap the color scale so a few very bad points don't wash out the good region.
vmax = np.percentile(losses, 60)
for ax, (xk, yk) in zip(axes, pairs):
    x = np.array([r[xk] for r in rows]) * 100
    y = np.array([r[yk] for r in rows]) * 100
    sc = ax.scatter(x, y, c=losses, cmap="viridis_r", vmin=losses.min(), vmax=vmax, s=25)
    ax.plot(best[xk] * 100, best[yk] * 100, "r*", markersize=20, markeredgecolor="white", markeredgewidth=0.8)
    ax.set_xlabel(f"{xk} [%]")
    ax.set_ylabel(f"{yk} [%]")
    fig.colorbar(sc, ax=ax, label="loss (capped at 60th pct for contrast)")
fig.suptitle("BO-evaluated points, pairwise projections (red star = best found)")
fig.tight_layout()
outfile = os.path.join(OUTPUT_DIR, "bo_pairwise.png")
fig.savefig(outfile, dpi=150)
print(f"wrote {outfile}")
