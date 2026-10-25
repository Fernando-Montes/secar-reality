"""
FP3 (VD1783) metric computation, mirroring fp2_metrics.py exactly but
reading out at 'FP3 Slits & VD1783' instead of VD1638. Same per-group
aperture-type initial distributions (groupsInfo), same FP1-slits ray
filtering convention -- only the readout element and the group/target
sets differ (a smaller subset, matching what the user's manual E2
spreadsheet fit actually used for FP3).

Group sets and MEASURED X values extracted directly from
Results/OpticsOptimization.xlsx, sheet 'E2', rows 68-121 ("Focal plane
3: 1783"), cross-checked against the sheet's own recorded slope cells
(dX/dE slope=-2.0956, dX/dalphaX slope=-0.2382 -- matching the KB's
recorded M16(FP3)=-2.1, M12(FP3)=-0.24) and against
MCoutput_analysis.py's independently-defined group/x-value sets for the
misalignment-MC robustness check (same groups, same dalphaX/dE values).
2026-08-14.

This module is a new file (not a modified copy of anything in the
parent ECAT directory), but it imports groupsInfo from a local copy of
ViewerAnalysis.py, and runECAT from a local copy of helper_functions.py,
both living alongside this file in AgentSearch/.
"""
import numpy as np
from config import element_names_stripped, fox_file as BASE_FOX_FILE
from helper_functions import runECAT
from ViewerAnalysis import groupsInfo
from fp2_metrics import (_needs_base_fox_fallback, build_initial_distribution,
                          SEPARATION_DISTANCE, ANGLES, DE_PARAM)

VD1783_INDEX = element_names_stripped.index('FP3 Slits & VD1783')
FP1_SLITS_INDEX = element_names_stripped.index('FP1 Slits')

# From Results/OpticsOptimization.xlsx, sheet 'E2': dX/dE section
# ("Focal plane 3: 1783"), rows 71-80 (groups 5,7,8,9,6,3,4,10), column G
# = dE, column K = measured X_mean [mm] at VD1783. Confirmed exact match
# (2026-08-14, per user correction) against the sheet's own recorded
# slope cell (K81 = -2.095627815006042): refitting these exact 8 groups
# gives slope = -2.095627815006044, agreeing to 13 decimal places.
DE_GROUPS = {
    '3':  1.91,
    '4': -1.87,
    '5':  0.14,
    '6':  0.70,
    '7':  0.14,
    '8':  2.02,
    '9': -1.74,
    '10': 0.02,
}
DE_GROUPS_MEASURED_X = {
    '3': -3.86,
    '4':  4.40,
    '5': -0.72,
    '6':  2.04,
    '7':  7.00,
    '8':  0.74,
    '9':  8.74,
    '10': 3.38,
}

# dX/dalphaX section, rows 85-87 (groups 10,14,15), column G = dalphaX,
# column K = measured X_mean [mm] at VD1783. Note: group 14/15's dalphaX
# values here (-16.36/-8.18) differ slightly from FP2's own DALPHAX_GROUPS
# (-16.38/-8.10) -- preserved exactly as recorded in the FP3 section of
# the spreadsheet, not assumed identical to FP2's.
DALPHAX_GROUPS = {
    '10':  0.33,
    '14': -16.36,
    '15': -8.18,
}
DALPHAX_GROUPS_MEASURED_X = {
    '10': 3.38,
    '14': 7.32,
    '15': 8.08,
}

ALL_GROUPS = sorted(set(DE_GROUPS) | set(DALPHAX_GROUPS), key=int)


def mean_x_at_vd1783(group, tuneChanges, nRays, run_cosy_flag, cosyFile, ta_y_offset=0.0):
    """Same FP1-slits-then-readout filtering convention as
    fp2_metrics.py's mean_x_at_vd1638, just reading out at VD1783
    instead of VD1638."""
    dist = build_initial_distribution(group, nRays, ta_y_offset=ta_y_offset)
    if run_cosy_flag and _needs_base_fox_fallback(tuneChanges):
        cosyFile = BASE_FOX_FILE
    all_x, all_ax, all_y, all_ay, all_dE, transmitted_x, transmitted_y, transmission_indices, \
        chamber_names, beampipes, usedFoxFile = runECAT(
            initialDistribution=dist, run_cosy_flag=run_cosy_flag, foxFile=cosyFile,
            tuneChanges=tuneChanges, save_rays_flag=False, final_index=VD1783_INDEX)

    slit_lo, slit_hi = groupsInfo[group]['fp1_slits']['slits_mm']
    x_mm = []
    for i, xx in enumerate(all_x):
        if transmission_indices[i] < VD1783_INDEX:
            continue
        x_at_slits = -xx[FP1_SLITS_INDEX] * 1000
        if slit_lo <= x_at_slits <= slit_hi:
            x_mm.append(-xx[VD1783_INDEX] * 1000)
    if len(x_mm) == 0:
        raise RuntimeError(
            f"No rays survived the FP1-slit+transmission cut for group {group} "
            f"with tuneChanges={tuneChanges} (nRays requested={nRays})")
    x_mm = np.array(x_mm)
    return float(np.mean(x_mm)), float(np.std(x_mm)), len(x_mm), usedFoxFile


def compute_M12_M16(tuneChanges, nRays, cosyFile, ta_y_offset=0.0, verbose=True,
                     run_cosy_flag=None):
    """Returns (M12, M16, per_group_results, usedFoxFile), same
    semantics as fp2_metrics.py's compute_M12_M16 but for FP3/VD1783."""
    per_group = {}
    first = True
    usedFoxFile = cosyFile
    for group in ALL_GROUPS:
        this_run_cosy_flag = (first if run_cosy_flag is None else run_cosy_flag)
        mean_x, std_x, n, usedFoxFile = mean_x_at_vd1783(
            group, tuneChanges, nRays, this_run_cosy_flag, cosyFile, ta_y_offset=ta_y_offset)
        per_group[group] = (mean_x, std_x, n)
        if verbose:
            print(f"  group {group}: sim X_mean={mean_x:.3f} mm, std={std_x:.3f} mm, n={n}")
        if first:
            cosyFile = usedFoxFile
            first = False

    dalphaX_vals = np.array([DALPHAX_GROUPS[g] for g in DALPHAX_GROUPS])
    dalphaX_sim_x = np.array([per_group[g][0] for g in DALPHAX_GROUPS])
    M12 = np.polyfit(dalphaX_vals, dalphaX_sim_x, 1)[0]

    dE_vals = np.array([DE_GROUPS[g] for g in DE_GROUPS])
    dE_sim_x = np.array([per_group[g][0] for g in DE_GROUPS])
    M16 = np.polyfit(dE_vals, dE_sim_x, 1)[0]

    return M12, M16, per_group, usedFoxFile
