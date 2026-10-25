"""
Nominal-optics baseline check (2026-09-18), for the new quad-cycling /
field-error investigation: traces the generic E3-style beam
(GENERIC_DISTRIBUTION_TEMPLATE from fp2_metrics.py) through the WHOLE
beamline at nominal tune (tuneChanges=[]), plots the rays, and reports
the average beam spot position at DSSD (the final element) plus the
transmission fraction reaching it. This is the reference point that
+-1% quad field-error runs will be compared against.

Run:
    cd AgentSearch
    python3 nominal_dssd_check.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from config import element_names_stripped, z_coordinates
from helper_functions import runECAT, plot_rays_colored, compute_transmission, transmission_histogram
from fp2_metrics import GENERIC_DISTRIBUTION_TEMPLATE

FOCAL_PLANES = [("FP1 Slits", "VD1542", "FP 1"), ("FP2 Slits", "VD1638", "FP 2"),
                ("FP3 Slits", "FP3 Slits", "FP 3"), ("FP4 Slits", "DSSD", "FP 4")]

OUTPUT_DIR = "Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NRAYS = 5000
DSSD_INDEX = element_names_stripped.index('DSSD')
FP2_ZOOM_MIN, FP2_ZOOM_MAX = 15.5, 19.5


def add_legend(fig, legend_info):
    if not legend_info:
        return
    handles = [Line2D([0], [0], color=info["color"], lw=2, label=info["label"]) for info in legend_info]
    fig.axes[0].legend(handles=handles, loc='upper right', fontsize=7, title="Ray color groups")


if __name__ == "__main__":
    dist = dict(GENERIC_DISTRIBUTION_TEMPLATE, nRays=NRAYS)

    all_x, all_ax, all_y, all_ay, all_dE, transmitted_x, transmitted_y, transmission_indices, \
        chamber_names, beampipes, usedFoxFile = runECAT(
            initialDistribution=dist, run_cosy_flag=True, foxFile="pg_June2025",
            tuneChanges=[], save_rays_flag=False, final_index=DSSD_INDEX)

    # ---------------- transmission to DSSD ----------------
    # transmission_index is a COUNT of successfully-transmitted positions
    # (process_single_ray's "+= 1" adjustment): a ray blocked exactly AT
    # position V ends up with transmission_index == V, not V-1. So "did
    # this ray reach DSSD" requires transmission_index > DSSD_INDEX (it
    # was NOT blocked at DSSD's own check), not >=. Found 2026-09-18: the
    # >= version silently counted rays blocked exactly at DSSD as
    # "transmitted", giving a false 100% instead of the true 91.32%.
    n_total = len(transmission_indices)
    n_to_dssd = sum(1 for t in transmission_indices if t > DSSD_INDEX)
    transmission_pct = 100 * n_to_dssd / n_total
    print(f"\nLaunched {n_total} rays.")
    print(f"Transmission to DSSD (index {DSSD_INDEX}): {n_to_dssd}/{n_total} = {transmission_pct:.2f}%")

    # full per-position stopping histogram, for context. percentages[k] =
    # % of rays with transmission_index == k+1, i.e. blocked AT position
    # (k+1) -- so the corresponding element index is k+1, not k. The LAST
    # entry (k = DSSD_INDEX) is transmission_index == DSSD_INDEX+1, i.e.
    # the fully-transmitted bucket, not a stopping point -- excluded here.
    percentages = compute_transmission(transmitted_x, DSSD_INDEX + 1)
    worst_k = int(np.argmax(percentages[:-1]))
    print(f"(of the {100-transmission_pct:.2f}% lost, the single worst stopping point was "
          f"index {worst_k + 1} [{element_names_stripped[worst_k + 1]}] "
          f"at {percentages[worst_k]:.2f}% of all rays)")

    # ---------------- average spot position at DSSD ----------------
    # Same > vs >= fix as above: a ray blocked exactly at DSSD has a
    # "phantom" all_x[DSSD_INDEX] (the matrix-transformed position is
    # still computed even though it fell outside the aperture there) --
    # must be excluded, not averaged in.
    x_dssd = np.array([-x[DSSD_INDEX] * 1000 for i, x in enumerate(all_x) if transmission_indices[i] > DSSD_INDEX])
    y_dssd = np.array([-y[DSSD_INDEX] * 1000 for i, y in enumerate(all_y) if transmission_indices[i] > DSSD_INDEX])
    print(f"\nDSSD spot (rays that reach it, n={len(x_dssd)}):")
    print(f"  X: mean={x_dssd.mean():+.3f} mm, std={x_dssd.std():.3f} mm")
    print(f"  Y: mean={y_dssd.mean():+.3f} mm, std={y_dssd.std():.3f} mm")

    # ---------------- plots ----------------
    fig0 = transmission_histogram(transmitted_x, element_names_stripped, transmission_indices,
                                    title=f"Transmission analysis, nominal optics ({n_total} rays)",
                                    focal_planes=FOCAL_PLANES, return_plt=False)
    outfile0 = os.path.join(OUTPUT_DIR, "nominal_dssd_transmission_histogram.png")
    fig0.savefig(outfile0, dpi=150)
    print(f"wrote {outfile0}")

    color_dict = {"color_list": [dE[0] for dE in all_dE], "n_bins": 11, "color_key": "Initial dE/E"}
    fig, legend_info = plot_rays_colored(transmitted_x, transmitted_y, z_coordinates, color_dict,
                                           element_names_stripped, return_plt=False)
    add_legend(fig, legend_info)
    fig.suptitle(f"Nominal optics: {n_total} rays, transmission to DSSD={transmission_pct:.1f}%")
    outfile1 = os.path.join(OUTPUT_DIR, "nominal_dssd_check_full.png")
    fig.savefig(outfile1, dpi=150)
    print(f"\nwrote {outfile1}")

    # zoomed spot plot at DSSD itself
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.scatter(x_dssd, y_dssd, s=4, alpha=0.4, color="tab:blue")
    ax2.scatter([x_dssd.mean()], [y_dssd.mean()], marker="x", s=120, color="red", zorder=5,
                label=f"mean ({x_dssd.mean():+.2f}, {y_dssd.mean():+.2f}) mm")
    ax2.set_xlabel("X [mm]")
    ax2.set_ylabel("Y [mm]")
    ax2.set_title(f"DSSD spot, nominal optics (n={len(x_dssd)}, transmission={transmission_pct:.1f}%)")
    ax2.set_aspect("equal")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    outfile2 = os.path.join(OUTPUT_DIR, "nominal_dssd_spot.png")
    fig2.savefig(outfile2, dpi=150)
    print(f"wrote {outfile2}")
