"""
Diagnostics for fp2_bo_search_v2_results.csv (corrected 2-term loss, no Q6
prior). History has three phases in file order:
  1. OAT warm start        (30 pts,  1D lines through origin)
  2. V1_rescored_seed      (381 pts, v1's joint-combo history re-scored
                             under this loss -- merged in 2026-08-10 after
                             phase-1-only BO stalled at loss=598)
  3. BO_v2                 (fresh GP-driven joint evaluations)
Produces:
  1. bo_v2_convergence.png -- running-best loss vs. evaluation index
  2. bo_v2_pairwise.png    -- Q6/Q2/Q5 pairwise scatter, colored by loss,
                               best point marked
Run: python3 plot_bo_v2_search.py
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_FILE = "Results/fp2_bo_search_v2_results.csv"
OUTPUT_DIR = "Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

rows = list(csv.DictReader(open(RESULTS_FILE)))
sources = [r["source"] for r in rows]
rows = [{k: (float(v) if k != "source" else v) for k, v in r.items()} for r in rows]

losses = np.array([r["loss"] for r in rows])
running_best = np.minimum.accumulate(losses)
best = rows[int(np.argmin(losses))]

# phase boundaries, from the source column (first index where each phase starts)
phase_starts = {}
for i, s in enumerate(sources):
    if s not in phase_starts:
        phase_starts[s] = i

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(len(losses)), losses, "o", markersize=3, alpha=0.4, label="each evaluation")
ax.plot(range(len(losses)), running_best, "-", color="tab:red", linewidth=2, label="running best")
for name, idx in sorted(phase_starts.items(), key=lambda kv: kv[1]):
    if idx > 0:
        ax.axvline(idx, color="gray", linestyle=":")
    ax.text(idx, ax.get_ylim()[1] if False else losses.max(), name, rotation=90,
            fontsize=7, va="top", ha="right", color="gray")
ax.set_yscale("log")
ax.set_xlabel("evaluation index (file order: OAT warm start, V1 rescored seed, fresh BO)")
ax.set_ylabel("loss (log scale)")
ax.set_title(f"BO v2 convergence -- best loss={best['loss']:.2f} at "
             f"Q6={best['Q6_B_SC']*100:+.2f}% Q2={best['Q2_B_SC']*100:+.2f}% Q5={best['Q5_B_SC']*100:+.2f}%")
ax.legend(loc="upper right")
fig.tight_layout()
outfile = os.path.join(OUTPUT_DIR, "bo_v2_convergence.png")
fig.savefig(outfile, dpi=150)
print(f"wrote {outfile}")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
pairs = [("Q6_B_SC", "Q2_B_SC"), ("Q6_B_SC", "Q5_B_SC"), ("Q2_B_SC", "Q5_B_SC")]
# Cap the color scale so the sea of bad points doesn't wash out the (rare) good region.
vmax = np.percentile(losses, 20)
for ax, (xk, yk) in zip(axes, pairs):
    x = np.array([r[xk] for r in rows]) * 100
    y = np.array([r[yk] for r in rows]) * 100
    sc = ax.scatter(x, y, c=losses, cmap="viridis_r", vmin=losses.min(), vmax=vmax, s=20)
    ax.plot(best[xk] * 100, best[yk] * 100, "r*", markersize=20, markeredgecolor="white", markeredgewidth=0.8)
    ax.set_xlabel(f"{xk} [%]")
    ax.set_ylabel(f"{yk} [%]")
    fig.colorbar(sc, ax=ax, label="loss (capped at 20th pct for contrast)")
fig.suptitle("BO v2 (M12+M16-only loss): pairwise projections, colored by loss (red star = best found)")
fig.tight_layout()
outfile = os.path.join(OUTPUT_DIR, "bo_v2_pairwise.png")
fig.savefig(outfile, dpi=150)
print(f"wrote {outfile}")
