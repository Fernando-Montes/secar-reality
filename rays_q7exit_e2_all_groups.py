"""
E2 beam at the Q7 Exit for all FP2 and FP3 groups (2026-10-19), for efl-d7's
WF1 EFL scenarios: the rays that actually enter the E2 observables.

Groups: FP2 {3,4,5,7,8,9,10,13,14,15,16} and FP3 {3,4,5,6,7,8,9,10,14,15}
(fp2_metrics / fp3_metrics), union of 12 groups. Per group, exactly the beam
the FP2/FP3 evaluations use (fp3_eval): build_initial_distribution(group,
3000), seed 42, TA_Y offset 0. Nominal pg_June2025_corrected (cached maps).

Kept: rays inside every aperture up to and including the Q7 Exit AND inside
their group's FP1 slit window (-x*1000 at 'FP1 Slits'). Every other ray is
excluded from M12/M16/offsets at both FP2 and FP3 (both readouts are
downstream of the Q7 Exit), so nothing is lost by dropping them.

Output: Results/rays_q7exit_e2_all_groups.csv
  ray_uid (unique), group, ray_index (index within the group's 3000),
  in_fp2, in_fp3 (group membership), x_m, a_rad, y_m, b_rad, l_m, dE, dm, dz
  (COSY coordinates at the Q7 Exit, all 8 incl. l).

Run:
    cd AgentSearch
    python3 rays_q7exit_e2_all_groups.py
"""
import csv
import pickle
import numpy as np

from config import element_names_stripped as NAMES, fox_file, pkl_directory, beamline_geometry, z_coordinates
from helper_functions import load_optics_data, find_polygons_at_z, generateInitialDistribution
from fast_tracer import compile_map, apply_map, rays_to_array
import fp2_metrics
import fp3_metrics
from ViewerAnalysis import groupsInfo
from shapely import contains_xy

Q7 = NAMES.index('Q7 Exit')
FP1 = NAMES.index('FP1 Slits')
OUT = "Results/rays_q7exit_e2_all_groups.csv"
GROUPS = sorted(set(fp2_metrics.ALL_GROUPS) | set(fp3_metrics.ALL_GROUPS), key=int)

_, groups = load_optics_data(fox_file, pkl_directory)
with open(beamline_geometry, 'rb') as f:
    _, polys = find_polygons_at_z(pickle.load(f), z_coordinates)
maps = [compile_map(groups[k]) for k in range(Q7)]

rows = []
uid = 0
for g in GROUPS:
    X = rays_to_array(generateInitialDistribution(fp2_metrics.build_initial_distribution(g, 3000)))
    S = [X]
    for k in range(Q7):
        S.append(apply_map(S[-1], *maps[k]))
    S = np.stack(S, axis=1)                          # (n, Q7+1, 8)
    ok = np.ones(len(X), bool)
    for loc in range(Q7 + 1):
        if polys[loc] is not None:
            ok &= contains_xy(polys[loc], S[:, loc, 0], S[:, loc, 2])
    lo, hi = groupsInfo[g]['fp1_slits']['slits_mm']
    xs = -S[:, FP1, 0] * 1000
    keep = ok & (xs >= lo) & (xs <= hi)
    for i in np.where(keep)[0]:
        rows.append([uid, g, int(i), int(g in fp2_metrics.ALL_GROUPS), int(g in fp3_metrics.ALL_GROUPS),
                     *S[i, Q7]])
        uid += 1
    print(f"group {g:>2}: {keep.sum():5d} of {len(X)} kept  (FP2 {g in fp2_metrics.ALL_GROUPS}, FP3 {g in fp3_metrics.ALL_GROUPS})")

with open(OUT, "w", newline="") as f:
    f.write(f"# E2 beam at Q7 Exit (element {Q7}), lattice {fox_file}, MISALIGN=1, 3000 rays/group, seed 42; "
            f"only rays inside all apertures to Q7 Exit and inside their group's FP1 slit window\n")
    f.write("# COSY units: x,y [m]; a,b [rad]; l [m]; dE = dE/E\n")
    w = csv.writer(f)
    w.writerow(["ray_uid", "group", "ray_index", "in_fp2", "in_fp3", "x_m", "a_rad", "y_m", "b_rad", "l_m", "dE", "dm", "dz"])
    w.writerows(rows)
print(f"{len(rows)} rays -> {OUT}")
