"""
Corrected version of plot_family_mc_e3_shift_hist.py: histogram of the
REAL, wide-scan-verified Q6 shift (true_shift_wide_scan, from
fp2_family_e3_wide_summary.csv -- every fit's vertex confirmed to fall
inside the scanned range, unlike the narrow 7-point fits) across all 45
E2-passing candidates.

Run:
    cd AgentSearch
    python3 plot_family_e3_wide_shift_hist.py
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SUMMARY_FILE = "Results/fp2_family_e3_wide_summary.csv"
OUTPUT_DIR = "Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
E3_SHIFT_CENTER = -0.0088
E3_TOLERANCE = 0.004

rows = list(csv.DictReader(open(SUMMARY_FILE)))
shifts = [float(r["true_shift_wide_scan"]) * 100 for r in rows if r["true_shift_wide_scan"] != ""]
print(f"{len(shifts)} candidates with a trustworthy (in-range) wide-scan fit.")
print(f"shift range: [{min(shifts):.2f}%, {max(shifts):.2f}%]  mean={np.mean(shifts):.2f}%  median={np.median(shifts):.2f}%")

fig, ax = plt.subplots(figsize=(9, 6))
ax.hist(shifts, bins=15, color="tab:green", edgecolor="black", alpha=0.8)
ax.axvline(E3_SHIFT_CENTER * 100, color="tab:red", linestyle="--",
           label=f"E3-predicted shift ({E3_SHIFT_CENTER*100:+.2f}%)")
ax.axvspan((E3_SHIFT_CENTER - E3_TOLERANCE) * 100, (E3_SHIFT_CENTER + E3_TOLERANCE) * 100,
           color="tab:red", alpha=0.15, label="acceptance tolerance (+/-0.4%)")
ax.axvline(0, color="gray", linestyle=":", label="no shift (candidate's own Q6)")
ax.set_xlabel("true_shift_wide_scan [%]  (real, wide-scan-verified fits)")
ax.set_ylabel("count")
ax.set_title(f"REAL Q6 shift needed to minimize VD1638 spot, across {len(shifts)} E2-passing candidates\n"
             f"(wide, verified scans -- not the narrow 7-point extrapolations)")
ax.legend(fontsize=9, loc="upper left")
fig.tight_layout()
outfile = os.path.join(OUTPUT_DIR, "family_e3_wide_shift_hist.png")
fig.savefig(outfile, dpi=150)
print(f"wrote {outfile}")
