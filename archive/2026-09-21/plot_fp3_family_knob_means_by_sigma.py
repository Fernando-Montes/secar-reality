"""
Per-knob mean +/- observed range for the FP3 family (fp3_family_mc.py),
split by acceptance sigma -- same style as
plot_family_knob_means_by_sigma.py's FP2 version, adapted for FP3's 7
knobs and single unified search (no v1/v7 split needed).

Run:
    cd AgentSearch
    python3 plot_fp3_family_knob_means_by_sigma.py
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_FILE = "Results/fp3_family_mc_results.csv"
OUTPUT_DIR = "Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CHI2_1SIGMA_DF4 = 4.7196
CHI2_2SIGMA_DF4 = 9.715641142541578

FP3_KNOBS = ['Q8_B_SC', 'Q9_B_SC', 'Q10_B_SC', 'Q11_B_SC', 'B5_Exit_dB', 'B6_Exit_dB', 'WF2_Exit_dB']
LABELS = ['Q8', 'Q9', 'Q10', 'Q11', 'B5', 'B6', 'WF2']

rows = list(csv.DictReader(open(RESULTS_FILE)))
passing = [r for r in rows if r["status"] == "e2_passed"]
losses = np.array([float(r["loss"]) for r in passing])
data = np.array([[float(r[k]) * 100 for k in FP3_KNOBS] for r in passing])  # -> %

mask_1s = losses <= CHI2_1SIGMA_DF4
mask_2s = losses <= CHI2_2SIGMA_DF4  # true for all, since e2_passed already means <=2sigma
data_1s, data_2s = data[mask_1s], data[mask_2s]
print(f"2-sigma: {len(data_2s)} candidates.  1-sigma: {len(data_1s)} candidates.")

mean_1s, min_1s, max_1s = data_1s.mean(axis=0), data_1s.min(axis=0), data_1s.max(axis=0)
mean_2s, min_2s, max_2s = data_2s.mean(axis=0), data_2s.min(axis=0), data_2s.max(axis=0)

x = np.arange(len(FP3_KNOBS))
offset = 0.12

fig, ax = plt.subplots(figsize=(10, 7))
ax.errorbar(x - offset, mean_2s, yerr=[mean_2s - min_2s, max_2s - mean_2s],
            fmt='o', color="tab:orange", capsize=4, markersize=6,
            label=f"2-sigma accepted ({len(data_2s)} candidates)")
ax.errorbar(x + offset, mean_1s, yerr=[mean_1s - min_1s, max_1s - mean_1s],
            fmt='o', color="tab:blue", capsize=4, markersize=6,
            label=f"1-sigma accepted ({len(data_1s)} candidates)")
ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")
ax.set_xticks(x)
ax.set_xticklabels(LABELS)
ax.set_ylabel("knob value [%]")
ax.set_title("FP3 family: per-knob mean +/- observed range, by acceptance sigma")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
outfile = os.path.join(OUTPUT_DIR, "fp3_family_knob_means_by_sigma.png")
fig.savefig(outfile, dpi=150)
print(f"\nwrote {outfile}")

print(f"\n{'knob':6s} {'2s mean':>10s} {'2s range':>20s} {'1s mean':>10s} {'1s range':>20s}")
for i, lbl in enumerate(LABELS):
    print(f"{lbl:6s} {mean_2s[i]:10.4f} [{min_2s[i]:8.4f},{max_2s[i]:8.4f}] "
          f"{mean_1s[i]:10.4f} [{min_1s[i]:8.4f},{max_1s[i]:8.4f}]")
