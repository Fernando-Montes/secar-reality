"""
Plots the Q14/Q15 corner extension scan (q14_q15_corner_extension_scan.py)
as a 2D heatmap, per follow-up to the 2026-09-21 transmission-maximization
scan series. Shows transmission plateaus at 96.83% (183/189 rays) across a
whole ridge of (Q14,Q15) combinations rather than a single sharp peak --
evidence that the remaining lost rays are blocked somewhere unrelated to
Q14/Q15.

Run:
    cd AgentSearch
    python3 plot_q14_q15_corner_extension_scan.py
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_FILE = "Results/q14_q15_corner_extension_scan_results.csv"
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

best_val = max(float(r["transmission_pct"]) for r in rows)
best_rows = [r for r in rows if float(r["transmission_pct"]) == best_val]

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(grid, origin="lower", cmap="viridis", aspect="auto",
                extent=[min(q14_vals) * 100 - 1, max(q14_vals) * 100 + 1,
                        min(q15_vals) * 100 - 2.5, max(q15_vals) * 100 + 2.5])
for i, q15 in enumerate(q15_vals):
    for j, q14 in enumerate(q14_vals):
        if not np.isnan(grid[i, j]):
            ax.text(q14 * 100, q15 * 100, f"{grid[i, j]:.1f}", ha="center", va="center",
                    color="white", fontsize=7)
ax.scatter([float(r["Q14_B_SC"]) * 100 for r in best_rows],
           [float(r["Q15_B_SC"]) * 100 for r in best_rows],
           marker="*", s=250, color="red", edgecolor="black", zorder=5,
           label=f"plateau ({len(best_rows)} points tied at {best_val:.2f}%)")
ax.set_xlabel("Q14_B_SC [%]")
ax.set_ylabel("Q15_B_SC [%]")
ax.set_title("Corner extension: transmission to DSSD [%] vs Q14/Q15\n"
             "(default stepped_uniform beam) -- flattens into a ridge, no longer a single peak")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=9)
fig.colorbar(im, ax=ax, label="Transmission [%]")
fig.tight_layout()
outfile = os.path.join(OUTPUT_DIR, "q14_q15_corner_extension_scan.png")
fig.savefig(outfile, dpi=150, bbox_inches="tight")
print(f"wrote {outfile}")
print(f"\n{len(best_rows)} points tied at the plateau max of {best_val:.2f}%:")
for r in best_rows:
    print(f"  Q14={float(r['Q14_B_SC'])*100:+.1f}%  Q15={float(r['Q15_B_SC'])*100:+.1f}%")
