"""
Transmission histogram + DSSD X-Y spot plot + X/Y-vs-Z ray tracing for the
Q6=0% (nominal) point of q6_field_error_scan.py, per user request
2026-09-21 -- to see WHERE along the beamline rays are being lost
(default stepped_uniform beam, 189 launched, 125 reach DSSD = 66.14%).

Reuses the already-cached "pg_June2025" matrix -- no COSY needed.

Run:
    cd AgentSearch
    python3 q6_zero_transmission_detail.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from config import element_names_stripped, z_coordinates
from helper_functions import runECAT, transmission_histogram, plot_rays_colored


def add_legend(fig, legend_info):
    if not legend_info:
        return
    handles = [Line2D([0], [0], color=info["color"], lw=2, label=info["label"]) for info in legend_info]
    fig.axes[0].legend(handles=handles, loc='upper right', fontsize=7, title="Ray color groups")

OUTPUT_DIR = "Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
DSSD_INDEX = element_names_stripped.index('DSSD')
FOCAL_PLANES = [("FP1 Slits", "VD1542", "FP 1"), ("FP2 Slits", "VD1638", "FP 2"),
                ("FP3 Slits", "FP3 Slits", "FP 3"), ("FP4 Slits", "DSSD", "FP 4")]

if __name__ == "__main__":
    all_x, all_ax, all_y, all_ay, all_dE, transmitted_x, transmitted_y, transmission_indices, \
        chamber_names, beampipes, usedFoxFile = runECAT(
            initialDistribution=None, run_cosy_flag=False, foxFile="pg_June2025",
            tuneChanges=[], save_rays_flag=False, final_index=DSSD_INDEX)

    n_total = len(transmission_indices)
    n_to_dssd = sum(1 for t in transmission_indices if t > DSSD_INDEX)
    print(f"Launched {n_total} rays (default stepped_uniform), {n_to_dssd} reach DSSD "
          f"({100*n_to_dssd/n_total:.2f}%)")

    fig0 = transmission_histogram(transmitted_x, element_names_stripped, transmission_indices,
                                    title=f"Transmission analysis, Q6=0% nominal ({n_total} rays)",
                                    focal_planes=FOCAL_PLANES, return_plt=False)
    outfile0 = os.path.join(OUTPUT_DIR, "q6_zero_transmission_histogram.png")
    fig0.savefig(outfile0, dpi=150)
    print(f"wrote {outfile0}")

    x_dssd = np.array([-x[DSSD_INDEX] * 1000 for i, x in enumerate(all_x) if transmission_indices[i] > DSSD_INDEX])
    y_dssd = np.array([-y[DSSD_INDEX] * 1000 for i, y in enumerate(all_y) if transmission_indices[i] > DSSD_INDEX])

    fig1, ax1 = plt.subplots(figsize=(6, 6))
    ax1.scatter(x_dssd, y_dssd, s=20, alpha=0.7, color="tab:blue")
    ax1.scatter([x_dssd.mean()], [y_dssd.mean()], marker="x", s=120, color="red", zorder=5,
                label=f"mean ({x_dssd.mean():+.2f}, {y_dssd.mean():+.2f}) mm")
    ax1.set_xlabel("X [mm]")
    ax1.set_ylabel("Y [mm]")
    ax1.set_title(f"DSSD spot, Q6=0% nominal (n={len(x_dssd)}/{n_total} survive, "
                  f"{100*n_to_dssd/n_total:.1f}%)")
    ax1.set_aspect("equal")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    fig1.tight_layout()
    outfile1 = os.path.join(OUTPUT_DIR, "q6_zero_dssd_spot.png")
    fig1.savefig(outfile1, dpi=150)
    print(f"wrote {outfile1}")

    # X and Y vs Z ray tracing (rays truncated at their own stopping point,
    # so lost rays visibly stop mid-plot rather than continuing past where
    # they were actually blocked)
    color_dict = {"color_list": [dE[0] for dE in all_dE], "n_bins": 11, "color_key": "Initial dE/E"}
    fig2, legend_info = plot_rays_colored(transmitted_x, transmitted_y, z_coordinates, color_dict,
                                            element_names_stripped, return_plt=False)
    add_legend(fig2, legend_info)
    fig2.suptitle(f"Q6=0% nominal: X/Y vs Z ray tracing ({n_total} rays, "
                 f"{n_to_dssd} reach DSSD = {100*n_to_dssd/n_total:.1f}%)")
    outfile2 = os.path.join(OUTPUT_DIR, "q6_zero_ray_trace.png")
    fig2.savefig(outfile2, dpi=150)
    print(f"wrote {outfile2}")
