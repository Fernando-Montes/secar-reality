"""
Per-knob mean +/- range across the combined v1+v7 E2-accepted family,
split by acceptance sigma (1-sigma: chi2<=4.7196, 2-sigma: chi2<=9.716,
both df=4) -- per user request 2026-08-15, a summary companion to the
corner plot (plot_family_mc_corner_v1_v7_full.py) showing the same 12
knobs on one axis instead of pairwise scatter.

v1 (fp2_family_mc_results.csv, 7-knob search) is missing Q3/B1-B4 --
treated as exactly 0 for those columns, same convention used in the
corner plot (those knobs weren't part of v1's search space, not that
v1's candidates happened to find 0 for them).

"Range" = the actual observed [min, max] across the accepted population
at that sigma level (not a parametric std) -- consistent with how the
rest of this project has treated acceptance as an empirical chi-square
cut, not a Gaussian-std summary.

Run:
    cd AgentSearch
    python3 plot_family_knob_means_by_sigma.py
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

V1_FILE = "Results/fp2_family_mc_results.csv"
V7_FILE = "Results/fp2_family_mc_v7_results.csv"
OUTPUT_DIR = "Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHI2_1SIGMA_DF4 = 4.7196
CHI2_2SIGMA_DF4 = 9.715641142541578

KNOBS = ['Q1_B_SC', 'Q2_B_SC', 'Q3_B_SC', 'Q4_B_SC', 'Q5_B_SC', 'Q6_B_SC', 'Q7_B_SC',
         'WF1_Exit_dB', 'B1_Exit_dB', 'B2_Exit_dB', 'B3_Exit_dB', 'B4_Exit_dB']
KNOB_LABELS = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'WF1', 'B1', 'B2', 'B3', 'B4']


def load_candidates(path, chi2_threshold):
    rows = list(csv.DictReader(open(path)))
    out = []
    for r in rows:
        if r["loss"] in (None, "") or float(r["loss"]) > chi2_threshold:
            continue
        out.append({k: (float(r[k]) if k in r and r[k] not in (None, "") else 0.0) for k in KNOBS})
    return out


def stack(candidates):
    return np.array([[c[k] for k in KNOBS] for c in candidates]) * 100  # -> %


if __name__ == "__main__":
    v1_1s = stack(load_candidates(V1_FILE, CHI2_1SIGMA_DF4))
    v7_1s = stack(load_candidates(V7_FILE, CHI2_1SIGMA_DF4))
    v1_2s = stack(load_candidates(V1_FILE, CHI2_2SIGMA_DF4))
    v7_2s = stack(load_candidates(V7_FILE, CHI2_2SIGMA_DF4))

    combined_1s = np.vstack([v1_1s, v7_1s])
    combined_2s = np.vstack([v1_2s, v7_2s])
    print(f"1-sigma combined: {len(combined_1s)} candidates (v1={len(v1_1s)}, v7={len(v7_1s)})")
    print(f"2-sigma combined: {len(combined_2s)} candidates (v1={len(v1_2s)}, v7={len(v7_2s)})")

    mean_1s, min_1s, max_1s = combined_1s.mean(axis=0), combined_1s.min(axis=0), combined_1s.max(axis=0)
    mean_2s, min_2s, max_2s = combined_2s.mean(axis=0), combined_2s.min(axis=0), combined_2s.max(axis=0)

    x = np.arange(len(KNOBS))
    offset = 0.12

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.errorbar(x - offset, mean_2s, yerr=[mean_2s - min_2s, max_2s - mean_2s],
                fmt='o', color="tab:orange", capsize=4, markersize=6,
                label=f"2-sigma accepted ({len(combined_2s)} candidates)")
    ax.errorbar(x + offset, mean_1s, yerr=[mean_1s - min_1s, max_1s - mean_1s],
                fmt='o', color="tab:blue", capsize=4, markersize=6,
                label=f"1-sigma accepted ({len(combined_1s)} candidates)")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels(KNOB_LABELS)
    ax.set_ylabel("knob value [%]")
    ax.set_title("v1+v7 combined family: per-knob mean +/- observed range, by acceptance sigma\n"
                 "(Q3/B1-B4 = 0 for v1 candidates -- not part of v1's 7-knob search space)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    outfile = os.path.join(OUTPUT_DIR, "family_knob_means_by_sigma.png")
    fig.savefig(outfile, dpi=150)
    print(f"\nwrote {outfile}")

    print(f"\n{'knob':6s} {'2s mean':>10s} {'2s range':>20s} {'1s mean':>10s} {'1s range':>20s}")
    for i, lbl in enumerate(KNOB_LABELS):
        print(f"{lbl:6s} {mean_2s[i]:10.4f} [{min_2s[i]:8.4f},{max_2s[i]:8.4f}] "
              f"{mean_1s[i]:10.4f} [{min_1s[i]:8.4f},{max_1s[i]:8.4f}]")
