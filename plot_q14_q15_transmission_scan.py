"""
Plots the Q14/Q15 transmission grid scan (q14_q15_transmission_scan.py) as
a 2D heatmap, per follow-up to the 2026-09-21 transmission-maximization
scan -- shows transmission is dominated by Q14 (Q15 effect is weak) and
that the tested +-10% range does not contain the optimum (best point sits
at the Q14=-10% edge).

Run:
    cd AgentSearch
    python3 plot_q14_q15_transmission_scan.py
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_FILE = "Results/q14_q15_transmission_scan_results.csv"
OUTPUT_DIR = "Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

rows = list(csv.DictReader(open(RESULTS_FILE)))
q14_vals = sorted(set(float(r["Q14_B_SC"]) for r in rows))
q15_vals = sorted(set(float(r["Q15_B_SC"]) for r in rows))

grid = np.full((len(q15_vals), len(q14_vals)), np.nan)
for r in rows:
    i = q15_vals.index(float(r["Q15_B_SC"]))
    j = q14_vals.index(float(r["Q14_B_SC"]))
    grid[i, j] = float(r["transmission_pct"])

best_row = max(rows, key=lambda r: float(r["transmission_pct"]))

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(grid, origin="lower", cmap="viridis", aspect="auto",
                extent=[min(q14_vals) * 100 - 2.5, max(q14_vals) * 100 + 2.5,
                        min(q15_vals) * 100 - 2.5, max(q15_vals) * 100 + 2.5])
for i, q15 in enumerate(q15_vals):
    for j, q14 in enumerate(q14_vals):
        ax.text(q14 * 100, q15 * 100, f"{grid[i, j]:.1f}", ha="center", va="center",
                color="white", fontsize=8)
ax.scatter([float(best_row["Q14_B_SC"]) * 100], [float(best_row["Q15_B_SC"]) * 100],
           marker="*", s=300, color="red", edgecolor="black", zorder=5,
           label=f"best in grid: Q14={float(best_row['Q14_B_SC'])*100:+.0f}%, "
                 f"Q15={float(best_row['Q15_B_SC'])*100:+.0f}% -> {float(best_row['transmission_pct']):.1f}%")
ax.set_xlabel("Q14_B_SC [%]")
ax.set_ylabel("Q15_B_SC [%]")
ax.set_title("Transmission to DSSD [%] vs Q14/Q15\n"
             "(default stepped_uniform beam, coarse 5x5 grid)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=9)
fig.colorbar(im, ax=ax, label="Transmission [%]")
fig.tight_layout()
outfile = os.path.join(OUTPUT_DIR, "q14_q15_transmission_scan.png")
fig.savefig(outfile, dpi=150, bbox_inches="tight")
print(f"wrote {outfile}")
print(f"\nBest in grid: Q14={float(best_row['Q14_B_SC'])*100:+.1f}%  "
      f"Q15={float(best_row['Q15_B_SC'])*100:+.1f}%  "
      f"transmission={float(best_row['transmission_pct']):.2f}%")

# Q14 marginal trend (average over Q15) to make the dominant-variable
# asymmetry explicit
print("\nQ14 marginal (mean transmission over all tested Q15, per Q14):")
for q14 in q14_vals:
    vals = [float(r["transmission_pct"]) for r in rows if float(r["Q14_B_SC"]) == q14]
    print(f"  Q14={q14*100:+.1f}%: mean={np.mean(vals):.2f}%  (range {min(vals):.2f}-{max(vals):.2f}%)")
