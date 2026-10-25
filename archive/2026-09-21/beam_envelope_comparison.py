"""
Beam envelope (X vs Z) comparison, per user request 2026-08-15: 5 rows x
3 columns:
  - column 1: whole SECAR, all 5 rays
  - column 2: zoomed near FP2 (z=15.5-19.5m), central + the two angle rays only
  - column 3: zoomed near FP2 (z=15.5-19.5m), central + the two energy rays only
(columns 2/3 split out of a single combined inset so each one's y-scale
is sized to its own rays -- the angle rays' much larger excursion had
been swamping the energy rays' visible range when they shared a panel)

5 rays per full panel (column 1); columns 2/3 show a 3-ray subset:
  - central (dE/E=0, dalphaX=0): black solid
  - dalphaX=+25mrad (dE/E=0): viridis-max color, solid
  - dalphaX=-25mrad (dE/E=0): viridis-min color, solid
  - dE/E=+3.1% (dalphaX=0): viridis-max color, dashed
  - dE/E=-3.1% (dalphaX=0): viridis-min color, dashed
Colors match z_evolution_test.py's colored plots (viridis, colored by
the swept parameter) -- since both that script's M12 and M16 tests used
their own independent 0-1 viridis normalization, the EXTREME rays in
either test map to the same colormap endpoints (cmap(0.0) / cmap(1.0)),
so both angle and dE extremes reuse the same two colors here.

Rows:
  1. Nominal
  2. Q3_B_SC shifted to move the M12(z)=0 crossing +0.28m
  3. Q3_B_SC shifted to move the M16(z)=0 crossing +0.28m
  4. Q6_B_SC shifted to move the M12(z)=0 crossing +0.28m
  5. Q6_B_SC shifted to move the M16(z)=0 crossing +0.28m
Required knob values taken from the CORRECTED (continuation-tracked)
Results/z_crossing_sensitivity_summary_v2.csv (2026-08-15 max-jump-cap
fix -- see z_crossing_sensitivity_continuation.py), not the original
buggy first-sign-change method.

Run:
    cd AgentSearch
    python3 beam_envelope_comparison.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import z_coordinates, element_names_stripped
from helper_functions import runECAT

OUTPUT_DIR = "Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FINAL_INDEX = element_names_stripped.index('DSSD')
FP2_ZOOM_MIN, FP2_ZOOM_MAX = 15.5, 19.5

FOCAL_PLANE_ELEMENTS = [
    ("FP1", "FP1 Slits"),
    ("FP2", "FP2 Slits"),
    ("FP3", "FP3 Slits & VD1783"),
    ("FP4", "FP4 Slits"),
]

CMAP = matplotlib.colormaps.get_cmap('viridis')
COLOR_MIN = CMAP(0.0)
COLOR_MAX = CMAP(1.0)

# Required knob values for a +0.28m crossing shift, from the corrected
# continuation-tracked summary (Results/z_crossing_sensitivity_summary_v2.csv,
# after the 2026-08-15 max-jump-cap fix -- see z_crossing_sensitivity_continuation.py):
#   Q3_B_SC: m12_global_fit_required_pct=-1.4054%, m16_global_fit_required_pct=+0.5088%
#   Q6_B_SC: m12_global_fit_required_pct=-1.5048%, m16_global_fit_required_pct=-0.4671%
ROWS = [
    ("Nominal", []),
    ("Q3_B_SC = -1.405% (shift M12 crossing +0.28m)", [['Q3', 'B_SC', -0.014053865868095643]]),
    ("Q3_B_SC = +0.509% (shift M16 crossing +0.28m)", [['Q3', 'B_SC', 0.005088151888602663]]),
    ("Q6_B_SC = -1.505% (shift M12 crossing +0.28m)", [['Q6', 'B_SC', -0.0150480883268255]]),
    ("Q6_B_SC = -0.467% (shift M16 crossing +0.28m)", [['Q6', 'B_SC', -0.004670543070399196]]),
]


def is_db_only(tuneChanges):
    return not tuneChanges or all(c[1] == 'dB' for c in tuneChanges)


def get_rays(tuneChanges, run_cosy_flag, cosyFile):
    dist_central = {
        'type': 'stepped_uniform',
        'x': {'min': 0, 'max': 0, 'steps': 1}, 'ax': {'min': 0, 'max': 0, 'steps': 1},
        'y': {'min': 0, 'max': 0, 'steps': 1}, 'ay': {'min': 0, 'max': 0, 'steps': 1},
        'dE': {'min': 0, 'max': 0, 'steps': 1}, 'dZ': {'min': 0, 'max': 0, 'steps': 1},
    }
    _, _, _, _, _, tx_c, _, _, _, _, usedFoxFile = runECAT(
        initialDistribution=dist_central, run_cosy_flag=run_cosy_flag, tuneChanges=tuneChanges,
        save_rays_flag=False, final_index=FINAL_INDEX, foxFile=cosyFile)

    dist_angle = {
        'type': 'stepped_uniform',
        'x': {'min': 0, 'max': 0, 'steps': 1}, 'ax': {'min': -25 / 1000, 'max': 25 / 1000, 'steps': 2},
        'y': {'min': 0, 'max': 0, 'steps': 1}, 'ay': {'min': 0, 'max': 0, 'steps': 1},
        'dE': {'min': 0, 'max': 0, 'steps': 1}, 'dZ': {'min': 0, 'max': 0, 'steps': 1},
    }
    _, _, _, _, _, tx_a, _, _, _, _, _ = runECAT(
        initialDistribution=dist_angle, run_cosy_flag=False, tuneChanges=tuneChanges,
        save_rays_flag=False, final_index=FINAL_INDEX, foxFile=usedFoxFile)

    dist_de = {
        'type': 'stepped_uniform',
        'x': {'min': 0, 'max': 0, 'steps': 1}, 'ax': {'min': 0, 'max': 0, 'steps': 1},
        'y': {'min': 0, 'max': 0, 'steps': 1}, 'ay': {'min': 0, 'max': 0, 'steps': 1},
        'dE': {'min': -3.1 / 100, 'max': 3.1 / 100, 'steps': 2}, 'dZ': {'min': 0, 'max': 0, 'steps': 1},
    }
    _, _, _, _, _, tx_d, _, _, _, _, _ = runECAT(
        initialDistribution=dist_de, run_cosy_flag=False, tuneChanges=tuneChanges,
        save_rays_flag=False, final_index=FINAL_INDEX, foxFile=usedFoxFile)

    return {
        "central": tx_c[0],
        "angle_neg": tx_a[0], "angle_pos": tx_a[1],  # min=-25mrad, max=+25mrad
        "dE_neg": tx_d[0], "dE_pos": tx_d[1],         # min=-3.1%, max=+3.1%
    }, usedFoxFile


def plot_ray(ax, x_traj, z_min, z_max, color, linestyle, linewidth, label):
    z = np.array(z_coordinates[:len(x_traj)])
    x = np.array(x_traj) * 1000
    mask = (z >= z_min) & (z <= z_max)
    ax.plot(z[mask], x[mask], color=color, linestyle=linestyle, linewidth=linewidth, label=label)


ALL_SPECS = [
    ("central", "black", "-", "central (dE=0, dalphaX=0)", 2.8),
    ("angle_pos", COLOR_MAX, "-", "dalphaX=+25mrad", 1.5),
    ("angle_neg", COLOR_MIN, "-", "dalphaX=-25mrad", 1.5),
    ("dE_pos", COLOR_MAX, "--", "dE/E=+3.1%", 1.5),
    ("dE_neg", COLOR_MIN, "--", "dE/E=-3.1%", 1.5),
]
CENTRAL_SPEC = ALL_SPECS[0]
ANGLE_SPECS = [CENTRAL_SPEC] + ALL_SPECS[1:3]
DE_SPECS = [CENTRAL_SPEC] + ALL_SPECS[3:5]


def draw_panel(ax, rays, specs, z_min, z_max):
    for key, color, ls, label, lw in specs:
        plot_ray(ax, rays[key], z_min, z_max, color, ls, lw, label)
    ax.axhline(0, color="grey", linestyle=":", linewidth=1)
    y_top = ax.get_ylim()[1]
    for elem_label, elem in FOCAL_PLANE_ELEMENTS + [("VD1638", "VD1638")]:
        idx = element_names_stripped.index(elem)
        z_elem = z_coordinates[idx]
        if z_min <= z_elem <= z_max:
            ax.axvline(z_elem, color="tab:purple", linestyle=":", alpha=0.6)
            ax.text(z_elem, y_top, elem_label, rotation=90, fontsize=7,
                    ha="right", va="top", color="tab:purple")
    ax.set_xlim(z_min, z_max)
    ax.set_ylabel("X [mm]")


def plot_row(axFull, axAngle, axDE, rays, title):
    z_full_max = z_coordinates[FINAL_INDEX]
    draw_panel(axFull, rays, ALL_SPECS, 0, z_full_max)
    draw_panel(axAngle, rays, ANGLE_SPECS, FP2_ZOOM_MIN, FP2_ZOOM_MAX)
    draw_panel(axDE, rays, DE_SPECS, FP2_ZOOM_MIN, FP2_ZOOM_MAX)
    axFull.set_title(f"{title} -- whole SECAR")
    axAngle.set_title(f"{title} -- zoom near FP2 (angle rays)")
    axDE.set_title(f"{title} -- zoom near FP2 (energy rays)")


if __name__ == "__main__":
    fig, axes = plt.subplots(5, 3, figsize=(22, 22))

    for i, (title, tuneChanges) in enumerate(ROWS):
        db_only = is_db_only(tuneChanges)
        if i == 0:
            cosyFile = "pg_June2025"
            run_cosy_flag = True
        elif db_only:
            cosyFile = "pg_June2025"
            run_cosy_flag = False
        else:
            tag = f"envelope_row{i}"
            cosyFile = f"pg_June2025_{tag}"[:60]
            run_cosy_flag = True
        print(f"Row {i+1}: {title}  (run_cosy_flag={run_cosy_flag}, cosyFile={cosyFile})")
        rays, usedFoxFile = get_rays(tuneChanges, run_cosy_flag, cosyFile)
        plot_row(axes[i, 0], axes[i, 1], axes[i, 2], rays, title)

    axes[0, 0].legend(fontsize=8, loc="upper right")
    for j in range(3):
        axes[-1, j].set_xlabel("Z [m]")
    fig.tight_layout()
    outfile = os.path.join(OUTPUT_DIR, "beam_envelope_comparison.png")
    fig.savefig(outfile, dpi=150)
    print(f"\nwrote {outfile}")
