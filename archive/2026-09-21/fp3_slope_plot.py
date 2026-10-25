"""
M12/M16 slope-fit plots at FP3 (VD1783), Experimental ONLY (per user
request 2026-08-14: drop Nominal from both panels -- no COSY run
needed here anymore).

M16 (dX/dE) is split into two separate experimental fits, per user
request, to check for a possible 1MM-vs-20mrad pepperpot-position
systematic:
  - "1 MM" groups:    3, 4, 10
  - "20 mrad" groups: 5, 7, 8, 9
Group 6 (20 mrad, wide-open FP1 slits) is plotted as a point but left
out of both fits, same as the original manual analysis.

Also overlays the 44 FP2 +-1-sigma E2-passing family candidates
(v1: 24, v7: 20), computed at FP3 by fp3_family_compute.py, as faint
gray lines -- same style as fp2_slope_plot_with_family.py -- if
Results/fp3_family_1sigma_results.csv exists.

Also, per user request 2026-08-16, computes and overlays the user's
manually-derived FP3 best-fit tune: TA_Y=+0.1mm, Q6-2.7%, WF1+0.25%,
Q11-0.67%, Q9+9.49%, WF2-0.077%. Unlike the family overlay (pre-computed
by fp3_family_compute.py), this needs ONE real COSY run done here --
mixes B_SC (Q6/Q11/Q9, lattice edits) and dB (WF1/WF2 Exit, per-ray)
tuneChanges, so a fresh compile is required.

Run:
    cd AgentSearch
    python3 fp3_slope_plot.py
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ViewerAnalysis import groupsInfo
from fp3_metrics import (compute_M12_M16, DE_GROUPS, DE_GROUPS_MEASURED_X,
                          DALPHAX_GROUPS, DALPHAX_GROUPS_MEASURED_X)

OUTPUT_DIR = "Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
FAMILY_FILE = "Results/fp3_family_1sigma_results.csv"

DE_1MM_GROUPS = ['3', '4', '10']
DE_20MRAD_GROUPS = ['5', '7', '8', '9']

MANUAL_BEST_TUNECHANGES = [
    ['Q6', 'B_SC', -0.027],
    ['WF1 Exit', 'dB', 0.0025],
    ['Q11', 'B_SC', -0.0067],
    ['Q9', 'B_SC', 0.0949],
    ['WF2 Exit', 'dB', -0.00077],
]
MANUAL_BEST_TA_Y_OFFSET = 0.1 / 1000
NRAYS_MANUAL = 3000


def load_family():
    if not os.path.exists(FAMILY_FILE):
        return []
    rows = list(csv.DictReader(open(FAMILY_FILE)))
    return [{"M12_sim": float(r["M12_sim"]), "M16_sim": float(r["M16_sim"]),
             "Offset12_sim": float(r["Offset12_sim"]), "Offset16_sim": float(r["Offset16_sim"])}
            for r in rows]


def group_marker(group):
    label = groupsInfo[group]['pp_alignment']['label']
    return '^' if '20 mrad' in label else 'o'


def fit_and_plot(ax, groups, color, label_prefix):
    x_vals = np.array([DE_GROUPS[g] for g in groups])
    y_vals = np.array([DE_GROUPS_MEASURED_X[g] for g in groups])
    m, b = np.polyfit(x_vals, y_vals, 1)
    xs = np.linspace(x_vals.min(), x_vals.max(), 50)
    ax.plot(xs, m * xs + b, "--", color=color, linewidth=2, zorder=4,
            label=f"{label_prefix} (y={m:.4f}x+{b:.2f})")


if __name__ == "__main__":
    family = load_family()
    print(f"{len(family)} 1-sigma family candidates loaded from {FAMILY_FILE}"
          if family else f"{FAMILY_FILE} not found -- skipping family overlay")

    print("\nComputing manual best-fit tune at FP3 "
          "(TA_Y=+0.1mm, Q6-2.7%, WF1+0.25%, Q11-0.67%, Q9+9.49%, WF2-0.077%)...")
    M12_manual, M16_manual, per_group_manual, _ = compute_M12_M16(
        tuneChanges=MANUAL_BEST_TUNECHANGES, nRays=NRAYS_MANUAL, cosyFile="pg_manual_fp3_best",
        ta_y_offset=MANUAL_BEST_TA_Y_OFFSET, verbose=True)

    def fit_offset(groups_dict):
        xs = np.array([groups_dict[g] for g in groups_dict])
        ys = np.array([per_group_manual[g][0] for g in groups_dict])
        return np.polyfit(xs, ys, 1)[1]

    Offset12_manual = fit_offset(DALPHAX_GROUPS)
    Offset16_manual = fit_offset(DE_GROUPS)
    print(f"\nManual best fit -- M12: slope={M12_manual:.4f}  intercept={Offset12_manual:.4f}")
    print(f"Manual best fit -- M16: slope={M16_manual:.4f}  intercept={Offset16_manual:.4f}")

    fig, (ax_alpha, ax_de) = plt.subplots(2, 1, figsize=(9, 12))

    # ---------------- X_mean vs dalphaX (M12) ----------------
    groups = list(DALPHAX_GROUPS.keys())
    x_vals = np.array([DALPHAX_GROUPS[g] for g in groups])
    meas_y = np.array([DALPHAX_GROUPS_MEASURED_X[g] for g in groups])
    xs = np.linspace(x_vals.min(), x_vals.max(), 50)

    for i, cand in enumerate(family):
        ax_alpha.plot(xs, cand["M12_sim"] * xs + cand["Offset12_sim"], "-", color="gray",
                       alpha=0.25, linewidth=0.8, zorder=1,
                       label=f"{len(family)} 1-sigma family candidates" if i == 0 else None)

    for g, xv, yv in zip(groups, x_vals, meas_y):
        ax_alpha.scatter(xv, yv, marker=group_marker(g), color="black", s=70, zorder=5)
        ax_alpha.annotate(g, (xv, yv), textcoords="offset points", xytext=(6, 6), fontsize=8, color="black")
    m, b = np.polyfit(x_vals, meas_y, 1)
    ax_alpha.plot(xs, m * xs + b, "k--", label=f"Exp. (y={m:.4f}x+{b:.2f})", zorder=4)
    ax_alpha.plot(xs, M12_manual * xs + Offset12_manual, "-", color="tab:green", linewidth=2, zorder=4,
                  label=f"Manual best fit (y={M12_manual:.4f}x+{Offset12_manual:.2f})")

    ax_alpha.set_xlabel("dalphaX [mrad]")
    ax_alpha.set_ylabel("X [mm]")
    ax_alpha.set_title("M12: X_mean vs dalphaX at VD1783 (FP3)")
    ax_alpha.legend(fontsize=9, loc="best")
    ax_alpha.grid(alpha=0.3)

    # ---------------- X_mean vs dE (M16) ----------------
    groups = list(DE_GROUPS.keys())
    x_vals = np.array([DE_GROUPS[g] for g in groups])
    meas_y = np.array([DE_GROUPS_MEASURED_X[g] for g in groups])
    xs = np.linspace(x_vals.min(), x_vals.max(), 50)

    for i, cand in enumerate(family):
        ax_de.plot(xs, cand["M16_sim"] * xs + cand["Offset16_sim"], "-", color="gray",
                    alpha=0.25, linewidth=0.8, zorder=1,
                    label=f"{len(family)} 1-sigma family candidates" if i == 0 else None)

    def point_color(g):
        if g in DE_1MM_GROUPS:
            return "tab:purple"
        if g in DE_20MRAD_GROUPS:
            return "tab:red"
        return "black"

    for g, xv, yv in zip(groups, x_vals, meas_y):
        c = point_color(g)
        ax_de.scatter(xv, yv, marker=group_marker(g), color=c, s=70, zorder=5)
        ax_de.annotate(g, (xv, yv), textcoords="offset points", xytext=(6, 6), fontsize=8, color=c)

    fit_and_plot(ax_de, DE_1MM_GROUPS, "tab:purple", "Exp. 1 MM (groups 3,4,10)")
    fit_and_plot(ax_de, DE_20MRAD_GROUPS, "tab:red", "Exp. 20 mrad (groups 5,7,8,9)")
    ax_de.plot(xs, M16_manual * xs + Offset16_manual, "-", color="tab:green", linewidth=2, zorder=4,
               label=f"Manual best fit (y={M16_manual:.4f}x+{Offset16_manual:.2f})")

    ax_de.set_xlabel("dE [%]")
    ax_de.set_ylabel("X [mm]")
    ax_de.set_title("M16: X_mean vs dE at VD1783 (FP3)")
    ax_de.legend(fontsize=9, loc="best")
    ax_de.grid(alpha=0.3)

    fig.suptitle("M12/M16 at FP3 (VD1783): Experimental + FP2 1-sigma family + manual best fit\n"
                 "circle = 1 MM groups, triangle = 20 mrad groups (group 6 shown, not fit); "
                 "faint gray = family; green = manual best fit", fontsize=11)
    fig.tight_layout()
    outfile = os.path.join(OUTPUT_DIR, "fp3_slope_comparison.png")
    fig.savefig(outfile, dpi=200)
    print(f"wrote {outfile}")
