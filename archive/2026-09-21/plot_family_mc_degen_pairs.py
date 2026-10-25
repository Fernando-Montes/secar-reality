"""
Q7-vs-Q6, Q5-vs-Q4, Q2-vs-Q1 pairwise projections from the family MC
results, colored by loss -- checking for degeneracy/coupling between
these specific adjacent-quad pairs.

Run:
    cd AgentSearch
    python3 plot_family_mc_degen_pairs.py
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MC_RESULTS_FILE = "Results/fp2_family_mc_results.csv"
OUTPUT_DIR = "Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CHI2_THRESHOLD_2SIGMA_DF4 = 9.715641142541578

mc_rows = []
with open(MC_RESULTS_FILE, "r", newline="") as f:
    for row in csv.DictReader(f):
        if row["loss"] in (None, ""):
            continue
        mc_rows.append(row)
mc_rows = [{k: (float(v) if k not in ("status",) and v not in (None, "") else v) for k, v in r.items()} for r in mc_rows]

losses = np.array([r["loss"] for r in mc_rows])
best = mc_rows[int(np.argmin(losses))]
n_pass = int(np.sum(losses <= CHI2_THRESHOLD_2SIGMA_DF4))
print(f"{len(mc_rows)} points ({n_pass} actually pass E2). Best: loss={best['loss']:.2f} at "
      f"Q1={best['Q1_B_SC']*100:+.2f}% Q2={best['Q2_B_SC']*100:+.2f}% Q4={best['Q4_B_SC']*100:+.2f}% "
      f"Q5={best['Q5_B_SC']*100:+.2f}% Q6={best['Q6_B_SC']*100:+.2f}% Q7={best['Q7_B_SC']*100:+.2f}%")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
pairs = [("Q6_B_SC", "Q7_B_SC"), ("Q4_B_SC", "Q5_B_SC"), ("Q1_B_SC", "Q2_B_SC")]
vmax = CHI2_THRESHOLD_2SIGMA_DF4
passing = losses <= CHI2_THRESHOLD_2SIGMA_DF4
for ax, (xk, yk) in zip(axes, pairs):
    x = np.array([r[xk] for r in mc_rows]) * 100
    y = np.array([r[yk] for r in mc_rows]) * 100
    sc = ax.scatter(x, y, c=losses, cmap="viridis_r", vmin=losses.min(), vmax=vmax, s=25, edgecolors="none")
    ax.scatter(x[passing], y[passing], facecolors="none", edgecolors="black", linewidths=1.0, s=25)
    ax.plot(best[xk] * 100, best[yk] * 100, "r*", markersize=20, markeredgecolor="white", markeredgewidth=0.8)
    ax.set_xlabel(f"{xk} [%]")
    ax.set_ylabel(f"{yk} [%]")
    fig.colorbar(sc, ax=ax, label=f"loss (capped at real E2 threshold {vmax:.2f})")
fig.suptitle(f"Family MC: Q6/Q7, Q4/Q5, Q1/Q2 pairwise projections\n"
             f"black-outlined = actually passes E2 ({n_pass}/{len(mc_rows)}); red star = best found")
fig.tight_layout()
outfile = os.path.join(OUTPUT_DIR, "family_mc_degen_pairs.png")
fig.savefig(outfile, dpi=150)
print(f"wrote {outfile}")
