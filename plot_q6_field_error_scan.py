"""
Plots all 5 quantities from q6_field_error_scan.py vs Q6_B_SC, per user
request 2026-09-21 -- one figure, 5 panels. Refined same day: finer
0.1% steps (21 points instead of 5), and mean/width reverted back to
DSSD itself (was briefly VD1879 earlier the same session). The last two
panels (DSSD mean X, DSSD X range) now use the hypothetical 50cm x 50cm
DSSD aperture columns (dssd_mean_x_50cm_mm / dssd_range_x_50cm_mm)
instead of the real (much smaller) DSSD aperture, per the same-day
follow-up request -- transmission (panel 1) is unchanged, still the real
DSSD aperture.

Run:
    cd AgentSearch
    python3 plot_q6_field_error_scan.py
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_FILE = "Results/q6_field_error_scan_results.csv"
OUTPUT_DIR = "Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

rows = list(csv.DictReader(open(RESULTS_FILE)))
rows.sort(key=lambda r: float(r["Q6_B_SC"]))
q6_pct = np.array([float(r["Q6_B_SC"]) * 100 for r in rows])

panels = [
    ("transmission_pct", "Transmission target->DSSD [%]", "tab:blue"),
    ("fp2_resolution", "FP2 resolution |ME(1,7)|/WV", "tab:orange"),
    ("fp3_resolution", "FP3 resolution |ME(1,7)|/WV", "tab:green"),
    ("dssd_mean_x_50cm_mm", "DSSD mean X [mm]\n(50x50cm)", "tab:red"),
    ("dssd_range_x_50cm_mm", "DSSD X range [mm]\n(50x50cm)", "tab:purple"),
]

fig, axes = plt.subplots(len(panels), 1, figsize=(8.5, 3 * len(panels)), sharex=True)
for ax, (key, ylabel, color) in zip(axes, panels):
    y = np.array([float(r[key]) for r in rows])
    ax.plot(q6_pct, y, "o-", color=color, markersize=4)
    ax.axvline(0, color="grey", linestyle=":", linewidth=1)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(alpha=0.3)

axes[-1].set_xlabel("Q6_B_SC field error [%]")
fig.suptitle("Q6 field-error scan on top of optimized Q14=-20%, Q15=-20% (+-1%, 0.1% steps,\n"
             "default stepped_uniform beam) -- mean/width at DSSD assuming a 50cm x 50cm aperture", fontsize=12)
fig.tight_layout(h_pad=1.5)
outfile = os.path.join(OUTPUT_DIR, "q6_field_error_scan.png")
fig.savefig(outfile, dpi=150)
print(f"wrote {outfile}")
