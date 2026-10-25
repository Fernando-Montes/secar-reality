"""
FP3 M12/M16 slope-fit plot with the POST-correction FP3 family overlaid
(fp3_family_mc.py's 51 E2-passing candidates), showing whether the
solved FP3 knob deltas actually bring the simulated M12/M16 response
close to the experimental fit lines -- companion to fp3_slope_plot.py,
which only had the PRE-correction (FP2-tuned-only) family.

Run:
    cd AgentSearch
    python3 plot_fp3_family_mc_slope.py
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ViewerAnalysis import groupsInfo
from fp3_metrics import DE_GROUPS, DE_GROUPS_MEASURED_X, DALPHAX_GROUPS, DALPHAX_GROUPS_MEASURED_X

RESULTS_FILE = "Results/fp3_family_mc_results.csv"
OUTPUT_DIR = "Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CHI2_1SIGMA_DF4 = 4.7196

DE_1MM_GROUPS = ['3', '4', '10']
DE_20MRAD_GROUPS = ['5', '7', '8', '9']


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
    rows = list(csv.DictReader(open(RESULTS_FILE)))
    passing = [r for r in rows if r["status"] == "e2_passed"]
    losses = np.array([float(r["loss"]) for r in passing])
    within_1sigma = losses <= CHI2_1SIGMA_DF4
    print(f"{len(passing)} FP3-corrected candidates ({within_1sigma.sum()} also within 1-sigma)")

    family = [{"M12_sim": float(r["M12_sim"]), "M16_sim": float(r["M16_sim"]),
               "Offset12_sim": float(r["Offset12_sim"]), "Offset16_sim": float(r["Offset16_sim"])}
              for r in passing]

    fig, (ax_alpha, ax_de) = plt.subplots(2, 1, figsize=(9, 12))

    # ---------------- M12 ----------------
    groups = list(DALPHAX_GROUPS.keys())
    x_vals = np.array([DALPHAX_GROUPS[g] for g in groups])
    meas_y = np.array([DALPHAX_GROUPS_MEASURED_X[g] for g in groups])
    xs = np.linspace(x_vals.min(), x_vals.max(), 50)

    for i, cand in enumerate(family):
        ax_alpha.plot(xs, cand["M12_sim"] * xs + cand["Offset12_sim"], "-", color="tab:green",
                       alpha=0.3, linewidth=0.8, zorder=1,
                       label=f"{len(family)} FP3-corrected candidates" if i == 0 else None)

    for g, xv, yv in zip(groups, x_vals, meas_y):
        ax_alpha.scatter(xv, yv, marker=group_marker(g), color="black", s=70, zorder=5)
        ax_alpha.annotate(g, (xv, yv), textcoords="offset points", xytext=(6, 6), fontsize=8, color="black")
    m, b = np.polyfit(x_vals, meas_y, 1)
    ax_alpha.plot(xs, m * xs + b, "k--", linewidth=2, label=f"Exp. (y={m:.4f}x+{b:.2f})", zorder=4)

    ax_alpha.set_xlabel("dalphaX [mrad]")
    ax_alpha.set_ylabel("X [mm]")
    ax_alpha.set_title("M12: X_mean vs dalphaX at VD1783 (FP3), post FP3-correction")
    ax_alpha.legend(fontsize=9, loc="best")
    ax_alpha.grid(alpha=0.3)

    # ---------------- M16 ----------------
    groups = list(DE_GROUPS.keys())
    x_vals = np.array([DE_GROUPS[g] for g in groups])
    meas_y = np.array([DE_GROUPS_MEASURED_X[g] for g in groups])
    xs = np.linspace(x_vals.min(), x_vals.max(), 50)

    for i, cand in enumerate(family):
        ax_de.plot(xs, cand["M16_sim"] * xs + cand["Offset16_sim"], "-", color="tab:green",
                    alpha=0.3, linewidth=0.8, zorder=1,
                    label=f"{len(family)} FP3-corrected candidates" if i == 0 else None)

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
    m, b = np.polyfit(x_vals, meas_y, 1)
    ax_de.plot(xs, m * xs + b, "k--", linewidth=2, label=f"Exp. combined (y={m:.4f}x+{b:.2f})", zorder=4)

    ax_de.set_xlabel("dE [%]")
    ax_de.set_ylabel("X [mm]")
    ax_de.set_title("M16: X_mean vs dE at VD1783 (FP3), post FP3-correction")
    ax_de.legend(fontsize=9, loc="best")
    ax_de.grid(alpha=0.3)

    fig.suptitle(f"FP3 (VD1783): Experimental vs. {len(family)} FP3-corrected candidates\n"
                 "circle = 1 MM groups, triangle = 20 mrad groups; green = corrected family", fontsize=11)
    fig.tight_layout()
    outfile = os.path.join(OUTPUT_DIR, "fp3_family_mc_slope_comparison.png")
    fig.savefig(outfile, dpi=200)
    print(f"wrote {outfile}")
