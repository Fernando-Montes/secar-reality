"""
Reproduces the M12/M16 slope-fit plots from OpticsComparison-MatrixElements.pdf
(X_mean vs dalphaX, and X_mean vs dE, at VD1638), overlaying THREE cases:
  - Nominal        (tuneChanges=[])
  - User's E2 best (Q6 -2.7%, WF1 Exit dB +0.25%, TA_Y +0.1mm)
  - BO-found best  (Q6 -0.63%, Q2 -4.13%, Q5 +8.65%, TA_Y 0)   [2026-08-08]
alongside the measured data and its own fit line, same style/format as
the PDF's "M12 and M16 at FP2 (VD1638)" slides.

Run:
    cd AgentSearch
    python3 fp2_slope_plot.py
"""
import os
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
    # A case with empty tuneChanges (e.g. "Nominal") produces an empty
    # cosyChanges internally, and runECAT's run_cosy_flag=True fallback
    # path then runs COSY on whatever name is passed in as `foxFile`
    # rather than deriving one -- must be the real base name
    # ("pg_June2025") in that case, not a made-up one. Same fix as
    # fp2_oat_screen.py / check_q1_dy_diagnostic.py (2026-08-08 bug,
    # keeps recurring in new scripts -- worth remembering generally).
    if not cfg["tuneChanges"]:
        cosyFile = "pg_June2025"
    else:
        cosyFile = f"pg_June2025_slopeplot_{name[:10]}".replace(" ", "_").replace("'", "")
    M12, M16, per_group, _ = compute_M12_M16(
        tuneChanges=cfg["tuneChanges"], nRays=NRAYS, cosyFile=cosyFile,
        ta_y_offset=cfg["ta_y_offset"], verbose=False)
    return name, M12, M16, per_group


def group_marker(group):
    # "20 mrad" pepperpot position groups get a triangle, everything else
    # (1 MM / 3 hole) gets a circle -- matches the marker distinction used
    # in the original presentation's dX/dE plot.
    label = groupsInfo[group]['pp_alignment']['label']
    return '^' if '20 mrad' in label else 'o'


if __name__ == "__main__":
    with Pool(processes=3) as pool:
        results = dict((name, (M12, M16, pg)) for name, M12, M16, pg in pool.map(run_case, CASES.items()))

    for name, (M12, M16, _) in results.items():
        print(f"{name}: M12={M12:.4f}  M16={M16:.4f}")

    fig, (ax_alpha, ax_de) = plt.subplots(2, 1, figsize=(9, 12))

    # ---------------- X_mean vs dalphaX ----------------
    groups = list(DALPHAX_GROUPS.keys())
    x_vals = np.array([DALPHAX_GROUPS[g] for g in groups])
    meas_y = np.array([DALPHAX_GROUPS_MEASURED_X[g] for g in groups])
    for g, xv, yv in zip(groups, x_vals, meas_y):
        ax_alpha.scatter(xv, yv, marker=group_marker(g), color="black", s=60, zorder=5)
    m, b = np.polyfit(x_vals, meas_y, 1)
    xs = np.linspace(x_vals.min(), x_vals.max(), 50)
    ax_alpha.plot(xs, m * xs + b, "k--", label=f"Exp. (y={m:.3f}x+{b:.2f})")

    for name, cfg in CASES.items():
        _, _, per_group = results[name]
        sim_y = np.array([per_group[str(g)][0] for g in groups])
        m, b = np.polyfit(x_vals, sim_y, 1)
        ax_alpha.plot(xs, m * xs + b, "--", color=cfg["color"], label=f"{name} (y={m:.3f}x+{b:.2f})")
        ax_alpha.scatter(x_vals, sim_y, marker="s", color=cfg["color"], s=30, alpha=0.7)

    ax_alpha.set_xlabel("dalphaX [mrad]")
    ax_alpha.set_ylabel("X [mm]")
    ax_alpha.set_title("M12: X_mean vs dalphaX at VD1638 (FP2)")
    ax_alpha.legend(fontsize=8, loc="best")
    ax_alpha.grid(alpha=0.3)

    # ---------------- X_mean vs dE ----------------
    groups = list(DE_GROUPS.keys())
    x_vals = np.array([DE_GROUPS[g] for g in groups])
    meas_y = np.array([DE_GROUPS_MEASURED_X[g] for g in groups])
    for g, xv, yv in zip(groups, x_vals, meas_y):
        ax_de.scatter(xv, yv, marker=group_marker(g), color="black", s=60, zorder=5)
    m, b = np.polyfit(x_vals, meas_y, 1)
    xs = np.linspace(x_vals.min(), x_vals.max(), 50)
    ax_de.plot(xs, m * xs + b, "k--", label=f"Exp. (y={m:.3f}x+{b:.2f})")

    for name, cfg in CASES.items():
        _, _, per_group = results[name]
        sim_y = np.array([per_group[str(g)][0] for g in groups])
        m, b = np.polyfit(x_vals, sim_y, 1)
        ax_de.plot(xs, m * xs + b, "--", color=cfg["color"], label=f"{name} (y={m:.3f}x+{b:.2f})")
        ax_de.scatter(x_vals, sim_y, marker="s", color=cfg["color"], s=30, alpha=0.7)

    ax_de.set_xlabel("dE [%]")
    ax_de.set_ylabel("X [mm]")
    ax_de.set_title("M16: X_mean vs dE at VD1638 (FP2)")
    ax_de.legend(fontsize=8, loc="best")
    ax_de.grid(alpha=0.3)

    fig.suptitle("M12/M16 at FP2 (VD1638): Nominal vs. E2-best vs. BO-best\n"
                  "circle/triangle = measured (1 MM / 20 mrad), square = simulated", fontsize=11)
    fig.tight_layout()
    outfile = os.path.join(OUTPUT_DIR, "fp2_slope_comparison.png")
    fig.savefig(outfile, dpi=200)
    print(f"\nwrote {outfile}")
