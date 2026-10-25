"""
Same style as fp2_slope_plot_with_family.py, but per the user's request
2026-08-14: drops "BO best" (kept Nominal + User's E2 best only), and
overlays BOTH v1 (45 candidates, 7-knob search) and v7 (46 candidates,
12-knob search) family lines, in different colors, so their spread can
be compared directly against each other and against nominal/experiment.

No new COSY runs needed for the family lines -- both files already have
fitted slope+intercept (M12_sim/Offset12_sim, M16_sim/Offset16_sim) per
candidate; each is just the line y = slope*x + intercept. Only Nominal
and User's E2 best need actual COSY computation.

Run:
    cd AgentSearch
    python3 fp2_slope_plot_v1_v7_family.py
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
V1_FAMILY_FILE = "Results/fp2_family_mc_results.csv"
V7_FAMILY_FILE = "Results/fp2_family_mc_v7_results.csv"
CHI2_THRESHOLD_2SIGMA_DF4 = 9.715641142541578

CASES = {
    "Nominal": {"tuneChanges": [], "ta_y_offset": 0.0, "color": "tab:orange"},
    "User's E2 best (Q6-2.7%, WF1+0.25%, TA_Y+0.1mm)": {
        "tuneChanges": [['Q6', 'B_SC', -0.027], ['WF1 Exit', 'dB', 0.0025]],
        "ta_y_offset": 0.1 / 1000, "color": "tab:green"},
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


def load_family_candidates(path):
    rows = list(csv.DictReader(open(path)))
    passing = [r for r in rows if r["loss"] not in (None, "") and float(r["loss"]) <= CHI2_THRESHOLD_2SIGMA_DF4]
    return [{"M12_sim": float(r["M12_sim"]), "M16_sim": float(r["M16_sim"]),
             "Offset12_sim": float(r["Offset12_sim"]), "Offset16_sim": float(r["Offset16_sim"])}
            for r in passing]


if __name__ == "__main__":
    with Pool(processes=2) as pool:
        results = dict((name, (M12, M16, pg)) for name, M12, M16, pg in pool.map(run_case, CASES.items()))

    for name, (M12, M16, _) in results.items():
        print(f"{name}: M12={M12:.4f}  M16={M16:.4f}")

    v1_family = load_family_candidates(V1_FAMILY_FILE)
    v7_family = load_family_candidates(V7_FAMILY_FILE)
    print(f"\nv1: {len(v1_family)} E2-passing candidates.  v7: {len(v7_family)} E2-passing candidates.")

    fig, (ax_alpha, ax_de) = plt.subplots(2, 1, figsize=(9, 12))

    # ---------------- X_mean vs dalphaX (M12) ----------------
    groups = list(DALPHAX_GROUPS.keys())
    x_vals = np.array([DALPHAX_GROUPS[g] for g in groups])
    meas_y = np.array([DALPHAX_GROUPS_MEASURED_X[g] for g in groups])
    xs = np.linspace(x_vals.min(), x_vals.max(), 50)

    for i, cand in enumerate(v1_family):
        ax_alpha.plot(xs, cand["M12_sim"] * xs + cand["Offset12_sim"], "-", color="tab:blue",
                       alpha=0.25, linewidth=0.8, zorder=1,
                       label=f"v1 family ({len(v1_family)}, 7-knob)" if i == 0 else None)
    for i, cand in enumerate(v7_family):
        ax_alpha.plot(xs, cand["M12_sim"] * xs + cand["Offset12_sim"], "-", color="tab:red",
                       alpha=0.25, linewidth=0.8, zorder=2,
                       label=f"v7 family ({len(v7_family)}, 12-knob)" if i == 0 else None)

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

    for i, cand in enumerate(v1_family):
        ax_de.plot(xs, cand["M16_sim"] * xs + cand["Offset16_sim"], "-", color="tab:blue",
                    alpha=0.25, linewidth=0.8, zorder=1,
                    label=f"v1 family ({len(v1_family)}, 7-knob)" if i == 0 else None)
    for i, cand in enumerate(v7_family):
        ax_de.plot(xs, cand["M16_sim"] * xs + cand["Offset16_sim"], "-", color="tab:red",
                    alpha=0.25, linewidth=0.8, zorder=2,
                    label=f"v7 family ({len(v7_family)}, 12-knob)" if i == 0 else None)

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

    fig.suptitle("M12/M16 at FP2 (VD1638): Nominal vs. E2-best vs. v1 family vs. v7 family\n"
                  "circle/triangle = measured (1 MM / 20 mrad), square = simulated", fontsize=11)
    fig.tight_layout()
    outfile = os.path.join(OUTPUT_DIR, "fp2_slope_comparison_v1_v7_family.png")
    fig.savefig(outfile, dpi=200)
    print(f"\nwrote {outfile}")
