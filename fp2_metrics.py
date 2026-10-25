"""
Reusable FP2 (VD1638) metric computation: builds per-group aperture-type
initial distributions from groupsInfo, runs runECAT, and computes the
M12 (dX/dalphaX) and M16 (dX/dE) slopes via linear regression across the
group sets confirmed with the user (2026-08-07).

This module is a new file (not a modified copy of anything in the
parent ECAT directory), but it imports groupsInfo from a local copy of
ViewerAnalysis.py, and runECAT from a local copy of helper_functions.py,
both living alongside this file in AgentSearch/.
"""
import numpy as np
from config import element_names_stripped, fox_file as BASE_FOX_FILE
from helper_functions import runECAT
from ViewerAnalysis import groupsInfo


def _needs_base_fox_fallback(tuneChanges):
    """
    True when runECAT's internal prepareCOSYchanges will produce an empty
    cosyChanges list -- i.e. tuneChanges is empty, OR every entry is a
    'dB' change (dB is applied as a per-ray field perturbation during
    propagation, not a .fox file edit -- see prepareTuneChanges in
    helper_functions.py -- so it never contributes to cosyChanges).
    When that happens, runECAT's run_cosy_flag=True path falls back to
    running COSY on whatever `foxFile` the caller passed in verbatim
    (helper_functions.py:547-551), which must be the real base file, not
    a made-up name that doesn't exist as an actual .fox file. Found
    2026-08-10: an earlier fix here only checked `not tuneChanges`, which
    missed the dB-only case (e.g. tuneChanges=[['WF1 Exit','dB',0.0025]]),
    reproducing the same failure mode.
    """
    return not tuneChanges or all(c[1] == 'dB' for c in tuneChanges)

VD1638_INDEX = element_names_stripped.index('VD1638')
FP1_SLITS_INDEX = element_names_stripped.index('FP1 Slits')
FP2_SLITS_INDEX = element_names_stripped.index('FP2 Slits')

# Generic/default distribution, matching main_new_MC2.py's E3 Q6 scan
# exactly (same numeric values) — used for the true-FP2-vs-VD1638 width
# diagnostic, not for the E2 slope fit (which uses per-group distributions).
GENERIC_DISTRIBUTION_TEMPLATE = {
    'type': 'mixed_rays',
    'target': {'option': 'gaussian_2d', 'param': {'mean': (0.1304/1000, -0.24/1000), 'sigma': (0.6652/1000, 0.5291/1000), 'rho': 0.0}},
    'angles': {'option': 'gaussian_2d', 'param': {'mean': (-0.223/1000, 0.0/1000), 'sigma': (4.8/1000, 5/1000), 'rho': 0.0}},
    'dE': {'option': 'normal', 'param': {'mu': -0.13/100, 'sigma': 1.05/100}},
    'dZ': {'option': 'fixed', 'param': 0},
}

SEPARATION_DISTANCE = 0.48619
ANGLES = {'mu_x': -0.12/1000, 'mu_y': 0.0043393, 'sigma': 10.24/1000}
DE_PARAM = {'mu': 0.020249, 'sigma': 0.077522}

# Confirmed with user, 2026-08-07:
DE_GROUPS = {
    '3':  1.91,
    '4': -1.87,
    '5':  0.16,
    '7':  0.16,
    '8':  2.04,
    '9': -1.75,
    '10': 0.02,
}
DE_GROUPS_MEASURED_X = {
    '3': -1.23, '4': 0.47, '5': -0.57, '7': 1.83, '8': 0.70, '9': 2.05, '10': 0.25,
}
DALPHAX_GROUPS = {
    '10':  0.33,
    '13':  8.74,
    '14': -16.38,
    '15': -8.10,
    '16': 21.66,
}
DALPHAX_GROUPS_MEASURED_X = {
    '10': 0.25, '13': -0.49, '14': 6.02, '15': 3.81, '16': -3.42,
}

ALL_GROUPS = sorted(set(DE_GROUPS) | set(DALPHAX_GROUPS), key=int)


def build_initial_distribution(group, nRays, ta_y_offset=0.0):
    g = groupsInfo[group]
    return {
        'type': 'aperture',
        'nRays': nRays,
        'target_alignment': {
            'X': g['target_alignment']['X'],
            'Y': g['target_alignment']['Y'] + ta_y_offset,
            'R': g['target_alignment']['R'],
        },
        'aperture_alignment': {
            'X': g['pp_alignment']['X'],
            'Y': g['pp_alignment']['Y'],
            'R': g['pp_alignment']['R'],
            'separation_distance': SEPARATION_DISTANCE,
        },
        'angles': ANGLES,
        'dE': {'option': 'normal', 'param': DE_PARAM},
        'dZ': {'option': 'fixed', 'param': 0},
    }


def mean_x_at_vd1638(group, tuneChanges, nRays, run_cosy_flag, cosyFile, ta_y_offset=0.0):
    """
    Reproduces the exact FP1-slits-then-VD1638 filtering pattern found in
    the original ViewerAnalysis.py (lines ~484-499): a ray only counts if
    (a) it survives transmission past VD1638, AND (b) its X position at
    'FP1 Slits' (note the sign flip: -x*1000, matching that file's
    convention throughout) falls inside the group's slits_mm window.
    Without step (b), groups sharing the same target/pepperpot alignment
    but different FP1 slit windows are indistinguishable (2026-08-07 bug
    caught by the user).
    """
    dist = build_initial_distribution(group, nRays, ta_y_offset=ta_y_offset)
    if run_cosy_flag and _needs_base_fox_fallback(tuneChanges):
        cosyFile = BASE_FOX_FILE
    all_x, all_ax, all_y, all_ay, all_dE, transmitted_x, transmitted_y, transmission_indices, \
        chamber_names, beampipes, usedFoxFile = runECAT(
            initialDistribution=dist, run_cosy_flag=run_cosy_flag, foxFile=cosyFile,
            tuneChanges=tuneChanges, save_rays_flag=False, final_index=VD1638_INDEX)

    slit_lo, slit_hi = groupsInfo[group]['fp1_slits']['slits_mm']
    x_mm = []
    for i, xx in enumerate(all_x):
        if transmission_indices[i] < VD1638_INDEX:
            continue
        x_at_slits = -xx[FP1_SLITS_INDEX] * 1000
        if slit_lo <= x_at_slits <= slit_hi:
            x_mm.append(-xx[VD1638_INDEX] * 1000)
    if len(x_mm) == 0:
        # np.mean/np.std of an empty array silently return NaN (with only
        # a RuntimeWarning, not an exception) -- that NaN would otherwise
        # propagate silently into M12/M16/loss and crash things much
        # later (e.g. skopt's GP fit refuses NaN targets). Raise here so
        # callers' existing exception handling catches it as a real
        # failure instead. Found 2026-08-08: some (Q6,Q2,Q5) combinations
        # within the +-25% BO search bounds push transmission to zero for
        # this group's tight FP1-slit window.
        raise RuntimeError(
            f"No rays survived the FP1-slit+transmission cut for group {group} "
            f"with tuneChanges={tuneChanges} (nRays requested={nRays})")
    x_mm = np.array(x_mm)
    return float(np.mean(x_mm)), float(np.std(x_mm)), len(x_mm), usedFoxFile


def compute_M12_M16(tuneChanges, nRays, cosyFile, ta_y_offset=0.0, verbose=True,
                     run_cosy_flag=None):
    """
    Returns (M12, M16, per_group_results, usedFoxFile).
    If run_cosy_flag is None (default): first group triggers the COSY
    compute (run_cosy_flag=True), remaining groups reuse it — standalone
    use. If run_cosy_flag is explicitly True/False: that value is used
    for every group unchanged, letting a caller that already computed
    matrices for this cosyFile (e.g. the outer Q6 loop in the search
    script) force run_cosy_flag=False throughout and avoid re-running
    COSY for every WF1 dB value at the same Q6.
    """
    per_group = {}
    first = True
    usedFoxFile = cosyFile
    for group in ALL_GROUPS:
        this_run_cosy_flag = (first if run_cosy_flag is None else run_cosy_flag)
        mean_x, std_x, n, usedFoxFile = mean_x_at_vd1638(
            group, tuneChanges, nRays, this_run_cosy_flag, cosyFile, ta_y_offset=ta_y_offset)
        per_group[group] = (mean_x, std_x, n)
        if verbose:
            print(f"  group {group}: sim X_mean={mean_x:.3f} mm, std={std_x:.3f} mm, n={n}")
        if first:
            cosyFile = usedFoxFile  # lock in the actual fox file name for subsequent calls
            first = False

    dalphaX_vals = np.array([DALPHAX_GROUPS[g] for g in DALPHAX_GROUPS])
    dalphaX_sim_x = np.array([per_group[g][0] for g in DALPHAX_GROUPS])
    M12 = np.polyfit(dalphaX_vals, dalphaX_sim_x, 1)[0]

    dE_vals = np.array([DE_GROUPS[g] for g in DE_GROUPS])
    dE_sim_x = np.array([per_group[g][0] for g in DE_GROUPS])
    M16 = np.polyfit(dE_vals, dE_sim_x, 1)[0]

    return M12, M16, per_group, usedFoxFile


def compute_fp2_width(tuneChanges, nRays, cosyFile, run_cosy_flag=False, verbose=True):
    """
    E3-style diagnostic: mean/std(X) at 'FP2 Slits' (true focal plane) and
    at 'VD1638' (viewer, 0.28m downstream), using the same generic
    distribution as the user's own main_new_MC2.py Q6 scan. Not part of
    the E2 slope fit — purely diagnostic / for the E3 prior sanity check.
    Reuses the COSY matrices from a prior compute_M12_M16 call if
    run_cosy_flag=False and cosyFile matches (no repeated COSY invocation).
    """
    dist = dict(GENERIC_DISTRIBUTION_TEMPLATE, nRays=nRays)
    if run_cosy_flag and _needs_base_fox_fallback(tuneChanges):
        # Same fallback condition as mean_x_at_vd1638 above.
        cosyFile = BASE_FOX_FILE
    all_x, all_ax, all_y, all_ay, all_dE, transmitted_x, transmitted_y, transmission_indices, \
        chamber_names, beampipes, usedFoxFile = runECAT(
            initialDistribution=dist, run_cosy_flag=run_cosy_flag, foxFile=cosyFile,
            tuneChanges=tuneChanges, save_rays_flag=False, final_index=VD1638_INDEX)

    def stats_at(idx):
        x_mm = np.array([-xx[idx] * 1000 for i, xx in enumerate(all_x) if transmission_indices[i] >= idx])
        if len(x_mm) == 0:
            # Same empty-slice-NaN hazard as mean_x_at_vd1638 -- raise
            # instead of silently returning NaN. See that function's
            # comment for the full explanation (2026-08-08).
            raise RuntimeError(
                f"No rays survived transmission to index {idx} with "
                f"tuneChanges={tuneChanges} (nRays requested={nRays})")
        return float(np.mean(x_mm)), float(np.std(x_mm)), len(x_mm)

    fp2_mean, fp2_std, fp2_n = stats_at(FP2_SLITS_INDEX)
    vd_mean, vd_std, vd_n = stats_at(VD1638_INDEX)
    if verbose:
        print(f"  FP2 Slits: mean={fp2_mean:.3f} std={fp2_std:.3f} n={fp2_n}   "
              f"VD1638: mean={vd_mean:.3f} std={vd_std:.3f} n={vd_n}")
    return {'fp2_mean': fp2_mean, 'fp2_std': fp2_std, 'fp2_n': fp2_n,
            'vd1638_mean': vd_mean, 'vd1638_std': vd_std, 'vd1638_n': vd_n,
            'usedFoxFile': usedFoxFile}
