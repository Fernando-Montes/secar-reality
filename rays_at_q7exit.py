"""
Ray coordinates at the Q7 Exit element (2026-10-17), for another session.

Beam: the E2 per-group aperture distribution used in the FP2 family search
(fp2_metrics.build_initial_distribution: target + pepperpot from groupsInfo,
angles mu_x -0.12 mrad, mu_y +4.34 mrad, sigma 10.24 mrad; dE/E normal
mu 2.02%, sigma 7.75%; dZ = 0; seed 42; TA_Y offset 0).
Optics: nominal, pg_June2025_corrected (config.base). Q7 Exit is upstream of
WF1, so this is identical to pg_June2025 here. Cached maps; no COSY compile.
Tracing: fast_tracer (ray-by-ray identical to the Cython tracer).

Output CSV, one row per ray that passed the pepperpot (aperture rays are
generated already filtered by it):
  ray_id, group
  x0_m, a0_rad, y0_m, b0_rad, dE0     initial coordinates at the target
  x_m, a_rad, y_m, b_rad, dE          coordinates at Q7 Exit (COSY units;
                                      a, b are COSY's angle coordinates)
  transmission_index                  number of beamline locations passed
                                      before the first aperture failure
  reaches_q7exit                      transmission_index > Q7 Exit index
                                      (inside every aperture up to and
                                      including Q7 Exit)
  passes_fp1_slits                    reaches FP1 Slits and -x*1000 at FP1
                                      Slits is inside the group's window
                                      (the E2 analysis cut)
Coordinates are written for every ray, including lost ones (the tracer keeps
propagating them; values past the loss point are not physical) -- filter on
reaches_q7exit. Sign convention: the E2 analysis plots X [mm] = -x * 1000.

Run:
    cd AgentSearch
    python3 rays_at_q7exit.py [--group 10] [--nrays 1000]
"""
import argparse
import csv
import pickle
import numpy as np

from config import element_names_stripped, z_coordinates, beamline_geometry, pkl_directory, fox_file
from helper_functions import load_optics_data, find_polygons_at_z, generateInitialDistribution
from fast_tracer import trace_rays, rays_to_array
from fp2_metrics import build_initial_distribution
from ViewerAnalysis import groupsInfo

Q7_EXIT = element_names_stripped.index('Q7 Exit')
FP1_SLITS = element_names_stripped.index('FP1 Slits')

ap = argparse.ArgumentParser()
ap.add_argument("--group", default="10")
ap.add_argument("--nrays", type=int, default=1000)
args = ap.parse_args()

_, groups = load_optics_data(fox_file, pkl_directory)
with open(beamline_geometry, 'rb') as f:
    _, polys = find_polygons_at_z(pickle.load(f), z_coordinates)
R0 = rays_to_array(generateInitialDistribution(build_initial_distribution(args.group, args.nrays)))
all_x, all_ax, all_y, all_ay, all_dE, _, _, ti = trace_rays(R0, groups[:Q7_EXIT], polys[:Q7_EXIT + 1])
ti = np.array(ti)
X = np.array(all_x)
lo, hi = groupsInfo[args.group]['fp1_slits']['slits_mm']
reach = ti > Q7_EXIT
slits = (ti > FP1_SLITS) & (-X[:, FP1_SLITS] * 1000 >= lo) & (-X[:, FP1_SLITS] * 1000 <= hi)

out = f"Results/rays_q7exit_group{args.group}_{args.nrays}.csv"
with open(out, "w", newline="") as f:
    f.write(f"# rays at Q7 Exit (element index {Q7_EXIT}, z = {z_coordinates[Q7_EXIT]:.4f} m), "
            f"lattice {fox_file}, E2 group {args.group}, {args.nrays} rays requested, seed 42\n")
    f.write("# COSY units: x,y [m]; a,b [rad]; dE = dE/E. E2 plots use X[mm] = -x*1000. "
            "Coordinates of lost rays (reaches_q7exit=0) are not physical.\n")
    w = csv.writer(f)
    w.writerow(["ray_id", "group", "x0_m", "a0_rad", "y0_m", "b0_rad", "dE0",
                "x_m", "a_rad", "y_m", "b_rad", "dE", "transmission_index", "reaches_q7exit", "passes_fp1_slits"])
    for i in range(len(R0)):
        w.writerow([i, args.group, *R0[i, [0, 1, 2, 3, 5]],
                    all_x[i][Q7_EXIT], all_ax[i][Q7_EXIT], all_y[i][Q7_EXIT], all_ay[i][Q7_EXIT], all_dE[i][Q7_EXIT],
                    int(ti[i]), int(reach[i]), int(slits[i])])

print(f"group {args.group}: {len(R0)} rays passed the pepperpot (of {args.nrays} requested); "
      f"{reach.sum()} reach Q7 Exit; {(reach & slits).sum()} of those pass the FP1 slit window {lo, hi} mm")
sel = reach
print("at Q7 Exit (reaching rays):  X = -x*1000 mean {:+.3f} mm, std {:.3f} mm;  a mean {:+.3f} mrad, std {:.3f} mrad".format(
    np.mean(-X[sel, Q7_EXIT] * 1000), np.std(X[sel, Q7_EXIT] * 1000),
    np.mean(np.array(all_ax)[sel, Q7_EXIT] * 1000), np.std(np.array(all_ax)[sel, Q7_EXIT] * 1000)))
lost = ti[~reach]
if lost.size:
    where = {element_names_stripped[k]: int((lost == k).sum()) for k in np.unique(lost)}
    print("lost before Q7 Exit, at:", where)
print(f"wrote {out}")
