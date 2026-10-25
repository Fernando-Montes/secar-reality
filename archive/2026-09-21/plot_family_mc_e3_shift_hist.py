"""
Histogram of e3_fitted_shift (from the narrow 7-point quadratic fits
already stored in fp2_family_mc_results.csv) across all candidates that
actually pass E2 (loss <= 9.72). Uses the CSV values as-is -- no new
COSY runs -- even though we now know those narrow-window fits tend to
extrapolate (see fp2_e3_check_mc_candidate.py), to see whether there's
still a consistent pattern across all 45 candidates despite individual
fits being unreliable in magnitude.

Run:
    cd AgentSearch
    python3 plot_family_mc_e3_shift_hist.py
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_FILE = "Results/fp2_family_mc_results.csv"
OUTPUT_DIR = "Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CHI2_THRESHOLD_2SIGMA_DF4 = 9.715641142541578
E3_SHIFT_CENTER = -0.0088
E3_TOLERANCE = 0.004

rows = list(csv.DictReader(open(RESULTS_FILE)))
e2_passed = [r for r in rows if r["loss"] not in (None, "") and float(r["loss"]) <= CHI2_THRESHOLD_2SIGMA_DF4]
shifts = [float(r["e3_fitted_shift"]) * 100 for r in e2_passed if r["e3_fitted_shift"] not in (None, "")]
n_missing = len(e2_passed) - len(shifts)

print(f"{len(e2_passed)} E2-passing candidates, {len(shifts)} with a fitted E3 shift "
      f"({n_missing} had no interior parabola minimum / errored).")
print(f"shift range: [{min(shifts):.2f}%, {max(shifts):.2f}%]  mean={np.mean(shifts):.2f}%  median={np.median(shifts):.2f}%")

fig, ax = plt.subplots(figsize=(9, 6))
ax.hist(shifts, bins=20, color="tab:blue", edgecolor="black", alpha=0.8)
ax.axvline(E3_SHIFT_CENTER * 100, color="tab:red", linestyle="--",
           label=f"E3-predicted shift ({E3_SHIFT_CENTER*100:+.2f}%)")
ax.axvspan((E3_SHIFT_CENTER - E3_TOLERANCE) * 100, (E3_SHIFT_CENTER + E3_TOLERANCE) * 100,
           color="tab:red", alpha=0.15, label="acceptance tolerance (+/-0.4%)")
ax.axvline(0, color="gray", linestyle=":", label="no shift (candidate's own Q6)")
ax.set_xlabel("e3_fitted_shift [%]  (from narrow 7-point fits, per fp2_family_mc_results.csv)")
ax.set_ylabel("count")
ax.set_title(f"Distribution of fitted Q6 shift across {len(shifts)} E2-passing MC candidates")
ax.legend(fontsize=9, loc="best")
fig.tight_layout()
outfile = os.path.join(OUTPUT_DIR, "family_mc_e3_shift_hist.png")
fig.savefig(outfile, dpi=150)
print(f"wrote {outfile}")
