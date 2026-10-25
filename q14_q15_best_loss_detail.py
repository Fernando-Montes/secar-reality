"""
Loss-location diagnostic (2026-09-21) at a representative point on the
Q14/Q15 transmission plateau found by q14_q15_corner_extension_scan.py:
transmission caps at 96.83% (183/189 rays) across a whole ridge of
different (Q14,Q15) combinations, suggesting the remaining 6 lost rays
are blocked somewhere that has nothing to do with Q14/Q15 tuning. Uses
Q14=-20%, Q15=-20% (mid-ridge, 96.83%) and traces where the lost rays
actually stop, the same way q6_zero_transmission_detail.py did for the
nominal tune.

Single COSY compile (not a sweep) -- still runs the .fox compiler, so per
project convention this is handed to the user to run rather than run in
the background.

Run:
    cd AgentSearch
    /Users/fernandomontes/opt/anaconda3/envs/SECAR/bin/python3 q14_q15_best_loss_detail.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from config import element_names_stripped, z_coordinates
from helper_functions import runECAT, transmission_histogram, plot_rays_colored

Q14_VALUE = -0.20
Q15_VALUE = -0.20


def add_legend(fig, legend_info):
    if not legend_info:
        return
    handles = [Line2D([0], [0], color=info["color"], lw=2, label=info["label"]) for info in legend_info]
    # placed outside the axes (to the right) so it doesn't sit on top of
    # the ray traces themselves
    fig.axes[0].legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02, 1),
                        borderaxespad=0., fontsize=7, title="Ray color groups")


OUTPUT_DIR = "Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
DSSD_INDEX = element_names_stripped.index('DSSD')
FOCAL_PLANES = [("FP1 Slits", "VD1542", "FP 1"), ("FP2 Slits", "VD1638", "FP 2"),
                ("FP3 Slits", "FP3 Slits", "FP 3"), ("FP4 Slits", "DSSD", "FP 4")]

if __name__ == "__main__":
    tuneChanges = [['Q14', 'B_SC', Q14_VALUE], ['Q15', 'B_SC', Q15_VALUE]]
    cosyFile = "pg_June2025_q1415_best_m20_m20"

    all_x, all_ax, all_y, all_ay, all_dE, transmitted_x, transmitted_y, transmission_indices, \
        chamber_names, beampipes, usedFoxFile = runECAT(
            initialDistribution=None, run_cosy_flag=True, foxFile=cosyFile,
            tuneChanges=tuneChanges, save_rays_flag=False, final_index=DSSD_INDEX)

    n_total = len(transmission_indices)
    n_to_dssd = sum(1 for t in transmission_indices if t > DSSD_INDEX)
    print(f"Q14={Q14_VALUE*100:+.0f}%  Q15={Q15_VALUE*100:+.0f}%: "
          f"{n_to_dssd}/{n_total} reach DSSD ({100*n_to_dssd/n_total:.2f}%)")

    lost = [(i, idx) for i, idx in enumerate(transmission_indices) if idx <= DSSD_INDEX]
    print(f"\n{len(lost)} rays lost before DSSD. Lost-at element breakdown:")
    from collections import Counter
    counts = Counter(element_names_stripped[idx] for _, idx in lost)
    for elem, n in counts.most_common():
        print(f"  {elem}: {n} ray(s)")

    fig0 = transmission_histogram(transmitted_x, element_names_stripped, transmission_indices,
                                    title=f"Transmission analysis, Q14={Q14_VALUE*100:+.0f}%, "
                                          f"Q15={Q15_VALUE*100:+.0f}% ({n_total} rays)",
                                    focal_planes=FOCAL_PLANES, return_plt=False)
    outfile0 = os.path.join(OUTPUT_DIR, "q14_q15_best_transmission_histogram.png")
    fig0.savefig(outfile0, dpi=150)
    print(f"\nwrote {outfile0}")

    x_dssd = np.array([-x[DSSD_INDEX] * 1000 for i, x in enumerate(all_x) if transmission_indices[i] > DSSD_INDEX])
    y_dssd = np.array([-y[DSSD_INDEX] * 1000 for i, y in enumerate(all_y) if transmission_indices[i] > DSSD_INDEX])

    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.scatter(x_dssd, y_dssd, s=20, alpha=0.7, color="tab:blue")
    ax2.scatter([x_dssd.mean()], [y_dssd.mean()], marker="x", s=120, color="red", zorder=5,
                label=f"mean ({x_dssd.mean():+.2f}, {y_dssd.mean():+.2f}) mm")
    ax2.set_xlabel("X [mm]")
    ax2.set_ylabel("Y [mm]")
    ax2.set_title(f"DSSD spot, Q14={Q14_VALUE*100:+.0f}%, Q15={Q15_VALUE*100:+.0f}% "
                  f"(n={len(x_dssd)}/{n_total} survive, {100*n_to_dssd/n_total:.1f}%)")
    ax2.set_aspect("equal")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    outfile2 = os.path.join(OUTPUT_DIR, "q14_q15_best_dssd_spot.png")
    fig2.savefig(outfile2, dpi=150)
    print(f"wrote {outfile2}")

    color_dict = {"color_list": [dE[0] for dE in all_dE], "n_bins": 11, "color_key": "Initial dE/E"}
    fig1, legend_info = plot_rays_colored(transmitted_x, transmitted_y, z_coordinates, color_dict,
                                            element_names_stripped, return_plt=False)
    add_legend(fig1, legend_info)
    fig1.suptitle(f"Q14={Q14_VALUE*100:+.0f}%, Q15={Q15_VALUE*100:+.0f}%: X/Y vs Z ray tracing "
                 f"({n_total} rays, {n_to_dssd} reach DSSD = {100*n_to_dssd/n_total:.1f}%)")
    outfile1 = os.path.join(OUTPUT_DIR, "q14_q15_best_ray_trace.png")
    fig1.savefig(outfile1, dpi=150, bbox_inches="tight")
    print(f"wrote {outfile1}")
