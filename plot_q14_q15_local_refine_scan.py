"""
Plots the Q14/Q15 local refinement scan (q14_q15_local_refine_scan.py) as
a 2D heatmap, per follow-up to the 2026-09-21 transmission-maximization
scan series. Shows the Q14=-15%/Q15=+5% "peak" found by the earlier 1D
extension (which fixed Q15) was misleading -- once both Q14 and Q15 are
swept together, transmission keeps climbing toward the Q14=-18%,
Q15=-10% corner of this grid (96.83%), i.e. the true optimum is still
outside the tested range, further in that direction.

Run:
    cd AgentSearch
    python3 plot_q14_q15_local_refine_scan.py
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_FILE = "Results/q14_q15_local_refine_scan_results.csv"
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

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(grid, origin="lower", cmap="viridis", aspect="auto",
                extent=[min(q14_vals) * 100 - 0.5, max(q14_vals) * 100 + 0.5,
                        min(q15_vals) * 100 - 2.5, max(q15_vals) * 100 + 2.5])
for i, q15 in enumerate(q15_vals):
    for j, q14 in enumerate(q14_vals):
        if not np.isnan(grid[i, j]):
            ax.text(q14 * 100, q15 * 100, f"{grid[i, j]:.1f}", ha="center", va="center",
                    color="white", fontsize=7)
ax.scatter([float(best_row["Q14_B_SC"]) * 100], [float(best_row["Q15_B_SC"]) * 100],
           marker="*", s=300, color="red", edgecolor="black", zorder=5,
           label=f"best in grid: Q14={float(best_row['Q14_B_SC'])*100:+.0f}%, "
                 f"Q15={float(best_row['Q15_B_SC'])*100:+.0f}% -> {float(best_row['transmission_pct']):.2f}%")
ax.set_xlabel("Q14_B_SC [%]")
ax.set_ylabel("Q15_B_SC [%]")
ax.set_title("Local refinement: transmission to DSSD [%] vs Q14/Q15\n"
             "(default stepped_uniform beam) -- optimum still at grid edge")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=9)
fig.colorbar(im, ax=ax, label="Transmission [%]")
fig.tight_layout()
outfile = os.path.join(OUTPUT_DIR, "q14_q15_local_refine_scan.png")
fig.savefig(outfile, dpi=150, bbox_inches="tight")
print(f"wrote {outfile}")
print(f"\nBest in grid: Q14={float(best_row['Q14_B_SC'])*100:+.1f}%  "
      f"Q15={float(best_row['Q15_B_SC'])*100:+.1f}%  "
      f"transmission={float(best_row['transmission_pct']):.2f}%")
