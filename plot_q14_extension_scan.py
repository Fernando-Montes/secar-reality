"""
Plots the Q14 1D extension scan (q14_extension_scan.py, Q15 fixed at +5%)
-- shows transmission to DSSD peaks near Q14=-15% (89.42%) then falls off
on both sides, per follow-up to the 2026-09-21 transmission-maximization
scan.

Run:
    cd AgentSearch
    python3 plot_q14_extension_scan.py
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_FILE = "Results/q14_extension_scan_results.csv"
OUTPUT_DIR = "Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

rows = list(csv.DictReader(open(RESULTS_FILE)))
rows.sort(key=lambda r: float(r["Q14_B_SC"]))
q14_pct = np.array([float(r["Q14_B_SC"]) * 100 for r in rows])
trans = np.array([float(r["transmission_pct"]) for r in rows])

best = rows[int(np.argmax(trans))]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(q14_pct, trans, "o-", color="tab:blue")
ax.scatter([float(best["Q14_B_SC"]) * 100], [float(best["transmission_pct"])],
           marker="*", s=300, color="red", zorder=5,
           label=f"peak: Q14={float(best['Q14_B_SC'])*100:+.0f}% -> {float(best['transmission_pct']):.2f}%")
ax.set_xlabel("Q14_B_SC [%]")
ax.set_ylabel("Transmission to DSSD [%]")
ax.set_title(f"Q14 extension scan (Q15 fixed at +5%, default stepped_uniform beam)")
ax.grid(alpha=0.3)
ax.legend(fontsize=9)
fig.tight_layout()
outfile = os.path.join(OUTPUT_DIR, "q14_extension_scan.png")
fig.savefig(outfile, dpi=150)
print(f"wrote {outfile}")
print(f"\nPeak: Q14={float(best['Q14_B_SC'])*100:+.1f}%  transmission={float(best['transmission_pct']):.2f}%")
