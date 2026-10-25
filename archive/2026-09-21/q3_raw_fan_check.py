"""
Diagnostic (2026-08-14): checks whether M12/M16 vs Q3's sign at VD1638
depends on HOW the transfer-matrix element is measured.

Two different methods have been used in this project:
  1. fp2_metrics.compute_M12_M16 -- group-averaged X means from REAL
     aperture-type initial distributions (groupsInfo pepperpot
     positions), with FP1-slit filtering, fit vs each group's nominal
     dalphaX/dE value. This is what the family searches and the
     Q3 OAT/offset-Jacobian scan (Results/fp2_offset_jacobian_scan_results.csv)
     use, and it gives dM12/dQ3 > 0, dM16/dQ3 < 0 (opposite signs).
  2. z_crossing_sensitivity_scan.py's method -- a clean, APERTURE-FREE
     stepped ray fan (11 evenly spaced angles or dE values, no FP1-slit
     cut, no pepperpot/target offsets), slope-fit at a fixed z. This is
     what produced z_crossing_sensitivity_bar.png, which the user reads
     as implying dM12/dQ3 and dM16/dQ3 should have the SAME sign.

This script uses method (2)'s exact ray-fan generation, but evaluates
the slope AT A FIXED z (VD1638) instead of searching for a zero-crossing
along z -- so it directly answers: does the raw, unfiltered transfer
matrix element M12(VD1638) vs Q3 have a different sign than the
group/FP1-slit-filtered method found? If yes, the group-based method
(and everything built on it -- the whole family search) has an aperture-
selection artifact worth understanding. If no, the two ARE consistent
and the disagreement is in how the z-crossing plot's "required %" is
being interpreted, not in the underlying M12/M16 sign.

Run:
    cd AgentSearch
    python3 q3_raw_fan_check.py
"""
import numpy as np

from config import element_names_stripped
from helper_functions import runECAT

VD1638_IDX = element_names_stripped.index('VD1638')
N_STEPS = 11


def m12_m16_at_vd1638(tuneChanges, run_cosy_flag, cosyFile):
    dist_m16 = {
        'type': 'stepped_uniform',
        'x': {'min': 0, 'max': 0, 'steps': 1}, 'ax': {'min': 0, 'max': 0, 'steps': 1},
        'y': {'min': 0, 'max': 0, 'steps': 1}, 'ay': {'min': 0, 'max': 0, 'steps': 1},
        'dE': {'min': -3.1 / 100, 'max': 3.1 / 100, 'steps': N_STEPS},
        'dZ': {'min': 0, 'max': 0, 'steps': 1},
    }
    all_x, all_ax, all_y, all_ay, all_dE, tx, ty, ti, cn, bp, usedFoxFile = runECAT(
        initialDistribution=dist_m16, run_cosy_flag=run_cosy_flag, tuneChanges=tuneChanges,
        save_rays_flag=False, final_index=VD1638_IDX, foxFile=cosyFile)
    dE_vals = [dE[0] for dE in all_dE]
    x_at_vd = [x[VD1638_IDX] for x in tx]
    M16 = np.polyfit(dE_vals, x_at_vd, 1)[0]

    dist_m12 = {
        'type': 'stepped_uniform',
        'x': {'min': 0, 'max': 0, 'steps': 1}, 'ax': {'min': -25 / 1000, 'max': 25 / 1000, 'steps': N_STEPS},
        'y': {'min': 0, 'max': 0, 'steps': 1}, 'ay': {'min': 0, 'max': 0, 'steps': 1},
        'dE': {'min': 0, 'max': 0, 'steps': 1}, 'dZ': {'min': 0, 'max': 0, 'steps': 1},
    }
    all_x2, all_ax2, all_y2, all_ay2, all_dE2, tx2, ty2, ti2, cn2, bp2, usedFoxFile2 = runECAT(
        initialDistribution=dist_m12, run_cosy_flag=False, tuneChanges=tuneChanges,
        save_rays_flag=False, final_index=VD1638_IDX, foxFile=usedFoxFile)
    ax_vals = [ax[0] for ax in all_ax2]
    x_at_vd2 = [x[VD1638_IDX] for x in tx2]
    M12 = np.polyfit(ax_vals, x_at_vd2, 1)[0]
    return M12, M16, usedFoxFile


if __name__ == "__main__":
    print("NOMINAL:")
    M12_0, M16_0, fox0 = m12_m16_at_vd1638([], True, "pg_June2025")
    print(f"  M12={M12_0:.5f}  M16={M16_0:.5f}\n")

    results = [(0.0, M12_0, M16_0)]
    for q3 in [-0.02, -0.01, 0.01, 0.02]:
        tc = [['Q3', 'B_SC', q3]]
        tag = f"pg_June2025_rawfan_Q3{q3:+.4f}".replace(".", "p")
        M12, M16, _ = m12_m16_at_vd1638(tc, True, tag)
        print(f"Q3={q3*100:+.1f}%:  M12={M12:.5f}  M16={M16:.5f}")
        results.append((q3, M12, M16))

    results.sort()
    qs = np.array([r[0] for r in results])
    M12s = np.array([r[1] for r in results])
    M16s = np.array([r[2] for r in results])
    print(f"\nRaw-fan (no aperture) local slope, all 5 points:")
    print(f"  dM12/dQ3 = {np.polyfit(qs, M12s, 1)[0]:.3f}")
    print(f"  dM16/dQ3 = {np.polyfit(qs, M16s, 1)[0]:.3f}")
    print(f"\nCompare to group/FP1-slit-filtered method (fp2_offset_jacobian_scan_results.csv):")
    print(f"  dM12/dQ3 = +15.576  dM16/dQ3 = -41.496")
