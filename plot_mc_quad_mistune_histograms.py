"""
Histograms of the 5 tracked metrics across mc_quad_mistune_scan.py's
completed trials (all 15 quads mistuned simultaneously, each
Uniform[-1%,+1%] around its own operating point) -- per user request
2026-09-21. Safe to run at any time, including while the generator is
still running in another terminal (only reads completed/flushed rows).

DSSD mean X and range X use the 50cm x 50cm hypothetical aperture
columns, consistent with the quad ranking work earlier the same day.

Run:
    cd AgentSearch
    python3 plot_mc_quad_mistune_histograms.py
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_FILE = "Results/mc_quad_mistune_scan_results.csv"
OUTPUT_DIR = "Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

rows = list(csv.DictReader(open(RESULTS_FILE)))
n_trials = len(rows)
print(f"{n_trials} completed trials found.")

PANELS = [
    ("transmission_pct", "Transmission target->DSSD [%]", "tab:blue"),
    ("fp2_resolution", "FP2 resolution |ME(1,7)|/WV", "tab:orange"),
    ("fp3_resolution", "FP3 resolution |ME(1,7)|/WV", "tab:green"),
    ("dssd_mean_x_50cm_mm", "DSSD mean X [mm] (50x50cm)", "tab:red"),
    ("dssd_range_x_50cm_mm", "DSSD X range (dX) [mm] (50x50cm)", "tab:purple"),
]

fig, axes = plt.subplots(len(PANELS), 1, figsize=(8, 3 * len(PANELS)))
for ax, (key, xlabel, color) in zip(axes, PANELS):
    vals = np.array([float(r[key]) for r in rows])
    ax.hist(vals, bins=40, color=color, edgecolor="black", linewidth=0.3)
    ax.axvline(vals.mean(), color="black", linestyle="--", linewidth=1,
               label=f"mean={vals.mean():.2f}, std={vals.std():.2f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Trials")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    print(f"{key}: mean={vals.mean():.3f}  std={vals.std():.3f}  "
          f"min={vals.min():.3f}  max={vals.max():.3f}")

fig.suptitle(f"15-quad mistune Monte Carlo (n={n_trials} trials)\n"
             f"each quad ~ Uniform[-1%,+1%] around its own operating point", fontsize=12)
fig.tight_layout(h_pad=1.5)
outfile = os.path.join(OUTPUT_DIR, "mc_quad_mistune_histograms.png")
fig.savefig(outfile, dpi=150)
print(f"\nwrote {outfile}")
