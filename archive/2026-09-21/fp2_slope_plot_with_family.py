"""
Same as fp2_slope_plot.py (Nominal / User's E2 best / BO best vs.
Experimental, M12 and M16 at VD1638), but additionally overlays all 45
E2-passing family MC candidates (fp2_family_mc_results.csv) as faint
background lines, per the user's request 2026-08-12 -- to see visually
how the whole family compares to nominal and the experimental lines.

No new COSY runs needed for the 45 family lines -- their fitted
slope+intercept (M12_sim/Offset12_sim, M16_sim/Offset16_sim) are already
in fp2_family_mc_results.csv, so each is just the line
y = slope*x + intercept drawn over the same x-range as the other cases.
Only the three CASES (Nominal, E2-best, BO-best) and the experimental
fit require actual computation/data, same as the original script.

Run:
    cd AgentSearch
    python3 fp2_slope_plot_with_family.py
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multiprocessing import Pool

from ViewerAnalysis import groupsInfo
from fp2_metrics import (compute_M12_M16, DE_GROUPS, DE_GROUPS_MEASURED_X,
                          DALPHAX_GROUPS, DALPHAX_GROUPS_MEASURED_X)

NRAYS = 3000
OUTPUT_DIR = "Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
FAMILY_FILE = "Results/fp2_family_mc_results.csv"
CHI2_THRESHOLD_2SIGMA_DF4 = 9.715641142541578

CASES = {
    "Nominal": {"tuneChanges": [], "ta_y_offset": 0.0, "color": "tab:orange"},
    "User's E2 best (Q6-2.7%, WF1+0.25%, TA_Y+0.1mm)": {
        "tuneChanges": [['Q6', 'B_SC', -0.027], ['WF1 Exit', 'dB', 0.0025]],
        "ta_y_offset": 0.1 / 1000, "color": "tab:green"},
    "BO best (Q6-0.63%, Q2-4.13%, Q5+8.65%)": {
        "tuneChanges": [['Q6', 'B_SC', -0.0063], ['Q2', 'B_SC', -0.0413], ['Q5', 'B_SC', 0.0865]],
        "ta_y_offset": 0.0, "color": "tab:purple"},
}


def run_case(item):
    name, cfg = item
    if not cfg["tuneChanges"]:
        cosyFile = "pg_June2025"
    else:
        cosyFile = f"pg_June2025_slopeplot_{name[:10]}".replace(" ", "_").replace("'", "")
    M12, M16, per_group, _ = compute_M12_M16(
        tuneChanges=cfg["tuneChanges"], nRays=NRAYS, cosyFile=cosyFile,
        ta_y_offset=cfg["ta_y_offset"], verbose=False)
    return name, M12, M16, per_group


def group_marker(group):
    label = groupsInfo[group]['pp_alignment']['label']
    return '^' if '20 mrad' in label else 'o'


def load_family_candidates():
    rows = list(csv.DictReader(open(FAMILY_FILE)))
    passing = [r for r in rows if r["loss"] not in (None, "") and float(r["loss"]) <= CHI2_THRESHOLD_2SIGMA_DF4]
    return [{"M12_sim": float(r["M12_sim"]), "M16_sim": float(r["M16_sim"]),
             "Offset12_sim": float(r["Offset12_sim"]), "Offset16_sim": float(r["Offset16_sim"])}
            for r in passing]


if __name__ == "__main__":
    with Pool(processes=3) as pool:
        results = dict((name, (M12, M16, pg)) for name, M12, M16, pg in pool.map(run_case, CASES.items()))

    for name, (M12, M16, _) in results.items():
        print(f"{name}: M12={M12:.4f}  M16={M16:.4f}")

    family = load_family_candidates()
    print(f"\n{len(family)} E2-passing family candidates loaded from {FAMILY_FILE}")

    fig, (ax_alpha, ax_de) = plt.subplots(2, 1, figsize=(9, 12))

    # ---------------- X_mean vs dalphaX (M12) ----------------
    groups = list(DALPHAX_GROUPS.keys())
    x_vals = np.array([DALPHAX_GROUPS[g] for g in groups])
    meas_y = np.array([DALPHAX_GROUPS_MEASURED_X[g] for g in groups])
    xs = np.linspace(x_vals.min(), x_vals.max(), 50)

    for i, cand in enumerate(family):
        ax_alpha.plot(xs, cand["M12_sim"] * xs + cand["Offset12_sim"], "-", color="gray",
                       alpha=0.25, linewidth=0.8, zorder=1,
                       label="45 E2-passing family candidates" if i == 0 else None)

    for g, xv, yv in zip(groups, x_vals, meas_y):
        ax_alpha.scatter(xv, yv, marker=group_marker(g), color="black", s=60, zorder=5)
    m, b = np.polyfit(x_vals, meas_y, 1)
    ax_alpha.plot(xs, m * xs + b, "k--", label=f"Exp. (y={m:.3f}x+{b:.2f})", zorder=4, linewidth=2)

    for name, cfg in CASES.items():
        _, _, per_group = results[name]
        sim_y = np.array([per_group[str(g)][0] for g in groups])
        m, b = np.polyfit(x_vals, sim_y, 1)
        ax_alpha.plot(xs, m * xs + b, "--", color=cfg["color"], label=f"{name} (y={m:.3f}x+{b:.2f})",
                       zorder=4, linewidth=2)
        ax_alpha.scatter(x_vals, sim_y, marker="s", color=cfg["color"], s=30, alpha=0.9, zorder=4)

    ax_alpha.set_xlabel("dalphaX [mrad]")
    ax_alpha.set_ylabel("X [mm]")
    ax_alpha.set_title("M12: X_mean vs dalphaX at VD1638 (FP2)")
    ax_alpha.legend(fontsize=8, loc="best")
    ax_alpha.grid(alpha=0.3)

    # ---------------- X_mean vs dE (M16) ----------------
    groups = list(DE_GROUPS.keys())
    x_vals = np.array([DE_GROUPS[g] for g in groups])
    meas_y = np.array([DE_GROUPS_MEASURED_X[g] for g in groups])
    xs = np.linspace(x_vals.min(), x_vals.max(), 50)

    for i, cand in enumerate(family):
        ax_de.plot(xs, cand["M16_sim"] * xs + cand["Offset16_sim"], "-", color="gray",
                    alpha=0.25, linewidth=0.8, zorder=1,
                    label="45 E2-passing family candidates" if i == 0 else None)

    for g, xv, yv in zip(groups, x_vals, meas_y):
        ax_de.scatter(xv, yv, marker=group_marker(g), color="black", s=60, zorder=5)
    m, b = np.polyfit(x_vals, meas_y, 1)
    ax_de.plot(xs, m * xs + b, "k--", label=f"Exp. (y={m:.3f}x+{b:.2f})", zorder=4, linewidth=2)

    for name, cfg in CASES.items():
        _, _, per_group = results[name]
        sim_y = np.array([per_group[str(g)][0] for g in groups])
        m, b = np.polyfit(x_vals, sim_y, 1)
        ax_de.plot(xs, m * xs + b, "--", color=cfg["color"], label=f"{name} (y={m:.3f}x+{b:.2f})",
                    zorder=4, linewidth=2)
        ax_de.scatter(x_vals, sim_y, marker="s", color=cfg["color"], s=30, alpha=0.9, zorder=4)

    ax_de.set_xlabel("dE [%]")
    ax_de.set_ylabel("X [mm]")
    ax_de.set_title("M16: X_mean vs dE at VD1638 (FP2)")
    ax_de.legend(fontsize=8, loc="best")
    ax_de.grid(alpha=0.3)

    fig.suptitle("M12/M16 at FP2 (VD1638): Nominal vs. E2-best vs. BO-best vs. 45-candidate family\n"
                  "circle/triangle = measured (1 MM / 20 mrad), square = simulated, faint gray = family", fontsize=11)
    fig.tight_layout()
    outfile = os.path.join(OUTPUT_DIR, "fp2_slope_comparison_with_family.png")
    fig.savefig(outfile, dpi=200)
    print(f"\nwrote {outfile}")
