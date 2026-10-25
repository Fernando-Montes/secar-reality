"""
Same histogram as plot_family_e3_wide_shift_hist.py, but combining BOTH
v1 (45 candidates, 7-knob search) and v7 (46 candidates, 12-knob search)
wide-scan-verified E3 shifts -- per the user's request 2026-08-14, to
see the full 91-candidate picture across both independent searches.

Run:
    cd AgentSearch
    python3 plot_family_e3_wide_shift_hist_v1_v7.py
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

V1_SUMMARY_FILE = "Results/fp2_family_e3_wide_summary.csv"
V7_SUMMARY_FILE = "Results/fp2_family_e3_wide_summary_v7.csv"
OUTPUT_DIR = "Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
E3_SHIFT_CENTER = -0.0088
E3_TOLERANCE = 0.004


def load_shifts(path):
    rows = list(csv.DictReader(open(path)))
    return [float(r["true_shift_wide_scan"]) * 100 for r in rows if r["true_shift_wide_scan"] != ""]


shifts_v1 = load_shifts(V1_SUMMARY_FILE)
shifts_v7 = load_shifts(V7_SUMMARY_FILE)
all_shifts = shifts_v1 + shifts_v7
print(f"v1: {len(shifts_v1)} candidates.  v7: {len(shifts_v7)} candidates.  combined: {len(all_shifts)}")
print(f"v1: mean={np.mean(shifts_v1):.2f}%  median={np.median(shifts_v1):.2f}%  "
      f"range=[{min(shifts_v1):.2f}, {max(shifts_v1):.2f}]")
print(f"v7: mean={np.mean(shifts_v7):.2f}%  median={np.median(shifts_v7):.2f}%  "
      f"range=[{min(shifts_v7):.2f}, {max(shifts_v7):.2f}]")
print(f"combined: mean={np.mean(all_shifts):.2f}%  median={np.median(all_shifts):.2f}%  "
      f"range=[{min(all_shifts):.2f}, {max(all_shifts):.2f}]")

fig, ax = plt.subplots(figsize=(9, 6))
lo = min(all_shifts + [E3_SHIFT_CENTER * 100 - E3_TOLERANCE * 100 * 3])
hi = max(all_shifts)
bins = np.linspace(lo, hi, 26)
ax.hist(shifts_v1, bins=bins, color="tab:blue", edgecolor="black", alpha=0.6, label=f"v1 (7-knob, {len(shifts_v1)})")
ax.hist(shifts_v7, bins=bins, color="tab:red", edgecolor="black", alpha=0.6, label=f"v7 (12-knob, {len(shifts_v7)})")
ax.axvline(E3_SHIFT_CENTER * 100, color="tab:red", linestyle="--",
           label=f"E3-predicted shift ({E3_SHIFT_CENTER*100:+.2f}%)")
ax.axvspan((E3_SHIFT_CENTER - E3_TOLERANCE) * 100, (E3_SHIFT_CENTER + E3_TOLERANCE) * 100,
           color="tab:red", alpha=0.15, label="acceptance tolerance (+/-0.4%)")
ax.axvline(0, color="gray", linestyle=":", label="no shift (candidate's own Q6)")
ax.set_xlabel("true_shift_wide_scan [%]  (real, wide-scan-verified fits)")
ax.set_ylabel("count")
ax.set_title(f"REAL Q6 shift needed to minimize VD1638 spot, v1+v7 combined ({len(all_shifts)} candidates)\n"
             f"(wide, verified scans -- two independent search methodologies)")
ax.legend(fontsize=9, loc="upper left")
fig.tight_layout()
outfile = os.path.join(OUTPUT_DIR, "family_e3_wide_shift_hist_v1_v7.png")
fig.savefig(outfile, dpi=150)
print(f"wrote {outfile}")
