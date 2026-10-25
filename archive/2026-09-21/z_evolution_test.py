"""
Educational visualization (2026-08-12, per user's suggestion) of how M12
and M16 evolve as a function of Z along the beamline, using nominal
tuneChanges=[] -- not tied to any specific candidate, just understanding
the SECAR lattice's general optics behavior.

Test 1 (M16 vs Z): fixed angle_X=angle_Y=0 (point-like, parallel beam),
sweep dE/E over +-3.1% (11 evenly-stepped rays). Colored by initial dE ->
X(z) for each ray is ~ M16(z)*dE, so this directly visualizes M16(z).

Test 2 (M12 vs Z): fixed dE=0, sweep angle_X over +-25mrad (11
evenly-stepped rays, angle_Y~0). Colored by initial angle_X -> X(z) for
each ray is ~ M12(z)*angle_X, directly visualizing M12(z).

Uses the 'stepped_uniform' initial-distribution type for exact, evenly
spaced deterministic values (cleaner than random sampling for this kind
of illustrative plot). The "fixed" dimensions use exactly min=max=0 --
NOT a tiny +-epsilon range as originally tried, which turned out to be a
real bug: stepped_uniform_rays' internal COSY-acceptance-ellipse check
normalizes each angle by ITS OWN max (ax_max = max(|min|,|max|)), so an
epsilon range makes that dimension's contribution to the ellipse metric
exactly (+-eps/eps)^2 = 1.0 regardless of how small eps is -- with both
ax and ay "fixed" that way the metric already exceeds the 1.01 cutoff
before dE is even considered, excluding every ray (the empty-array error
seen 2026-08-12). Exactly 0/0 instead evaluates to NaN, and
`NaN > 1.01` is always False, so the ray is correctly kept -- just with
a harmless RuntimeWarning, suppressed below.

Both figures run the full beamline (through DSSD) so all four focal
planes are visible, using plot_rays_colored's built-in FP1-FP4
highlighting.

Run:
    cd AgentSearch
    python3 z_evolution_test.py
"""
import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from config import z_coordinates, element_names_stripped
from helper_functions import runECAT, plot_rays_colored

warnings.filterwarnings("ignore", category=RuntimeWarning)

OUTPUT_DIR = "Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_STEPS = 11
FINAL_INDEX = element_names_stripped.index('DSSD')

# FP2 Slits z=17.30m, VD1638 z=17.58m -- zoom window with margin on both sides
FP2_ZOOM_MIN, FP2_ZOOM_MAX = 15.5, 19.5


def add_legend(fig, legend_info):
    if not legend_info:
        return
    handles = [Line2D([0], [0], color=info["color"], lw=2, label=info["label"]) for info in legend_info]
    fig.axes[0].legend(handles=handles, loc='upper right', fontsize=7, title="Ray color groups")


def plot_zoomed_x_vs_z(x_list, color_list, n_bins, z_min, z_max, title, outfile):
    """Standalone zoomed X-vs-Z plot, own figure (not a small inset) so the
    FP2 region is actually readable -- reconstructs the same viridis
    binned coloring plot_rays_colored uses internally, so colors match."""
    coords = np.asarray(color_list)
    min_val, max_val = coords.min(), coords.max()
    bounds = np.linspace(min_val, max_val, n_bins + 1)
    groups = np.digitize(coords, bounds[1:-1], right=False)
    cmap = matplotlib.colormaps.get_cmap('viridis').resampled(n_bins)
    ray_colors = [cmap(g) for g in groups]

    fig, ax = plt.subplots(figsize=(9, 6))
    y_all = []
    for i, x in enumerate(x_list):
        z = np.array(z_coordinates[:len(x)])
        x = np.array(x)
        mask = (z >= z_min) & (z <= z_max)
        if not np.any(mask):
            continue
        ax.plot(z[mask], x[mask] * 1000, color=ray_colors[i], linewidth=1.5)
        y_all.extend(x[mask] * 1000)

    for label, elem in [("FP2 Slits", "FP2 Slits"), ("VD1638", "VD1638")]:
        idx = element_names_stripped.index(elem)
        z_elem = z_coordinates[idx]
        if z_min <= z_elem <= z_max:
            ax.axvline(z_elem, color="tab:purple", linestyle=":", alpha=0.7)
            ax.text(z_elem, max(y_all) if y_all else 0, label, rotation=90, fontsize=8,
                    ha="right", va="top", color="tab:purple")

    ax.axhline(0, color='grey', linestyle='--', linewidth=1)
    ax.set_xlabel("z-Position Along Beamline (m)")
    ax.set_ylabel("x-Position (mm)")
    ax.set_xlim(z_min, z_max)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    print(f"wrote {outfile}")


# ---------------------------------------------------------------- Test 1: M16(z)
dist_m16 = {
    'type': 'stepped_uniform',
    'x':  {'min': 0, 'max': 0, 'steps': 1},
    'ax': {'min': 0, 'max': 0, 'steps': 1},
    'y':  {'min': 0, 'max': 0, 'steps': 1},
    'ay': {'min': 0, 'max': 0, 'steps': 1},
    'dE': {'min': -3.1 / 100, 'max': 3.1 / 100, 'steps': N_STEPS},
    'dZ': {'min': 0, 'max': 0, 'steps': 1},
}

all_x, all_ax, all_y, all_ay, all_dE, transmitted_x, transmitted_y, transmission_indices, \
    chamber_names, beampipes, usedFoxFile = runECAT(
        initialDistribution=dist_m16, run_cosy_flag=True, tuneChanges=[],
        save_rays_flag=False, final_index=FINAL_INDEX)

color_dict = {"color_list": [dE[0] for dE in all_dE], "n_bins": N_STEPS, "color_key": "Initial dE/E"}
fig, legend_info = plot_rays_colored(transmitted_x, transmitted_y, z_coordinates, color_dict,
                                       element_names_stripped, return_plt=False)
add_legend(fig, legend_info)
fig.suptitle("M16(z): X trajectories vs Z, colored by initial dE/E (angle=0, nominal tune)")
outfile1 = os.path.join(OUTPUT_DIR, "m16_vs_z_test.png")
fig.savefig(outfile1, dpi=150)
print(f"wrote {outfile1}")

plot_zoomed_x_vs_z(
    transmitted_x, [dE[0] for dE in all_dE], N_STEPS, FP2_ZOOM_MIN, FP2_ZOOM_MAX,
    "M16(z) zoomed near FP2: X trajectories vs Z, colored by initial dE/E",
    os.path.join(OUTPUT_DIR, "m16_vs_z_test_zoomFP2.png"))

# ---------------------------------------------------------------- Test 2: M12(z)
dist_m12 = {
    'type': 'stepped_uniform',
    'x':  {'min': 0, 'max': 0, 'steps': 1},
    'ax': {'min': -25 / 1000, 'max': 25 / 1000, 'steps': N_STEPS},
    'y':  {'min': 0, 'max': 0, 'steps': 1},
    'ay': {'min': 0, 'max': 0, 'steps': 1},
    'dE': {'min': 0, 'max': 0, 'steps': 1},
    'dZ': {'min': 0, 'max': 0, 'steps': 1},
}

all_x2, all_ax2, all_y2, all_ay2, all_dE2, transmitted_x2, transmitted_y2, transmission_indices2, \
    chamber_names2, beampipes2, usedFoxFile2 = runECAT(
        initialDistribution=dist_m12, run_cosy_flag=True, tuneChanges=[],
        save_rays_flag=False, final_index=FINAL_INDEX)

color_dict2 = {"color_list": [ax[0] for ax in all_ax2], "n_bins": N_STEPS, "color_key": "Initial angle_X [rad]"}
fig2, legend_info2 = plot_rays_colored(transmitted_x2, transmitted_y2, z_coordinates, color_dict2,
                                         element_names_stripped, return_plt=False)
add_legend(fig2, legend_info2)
fig2.suptitle("M12(z): X trajectories vs Z, colored by initial angle_X (dE=0, nominal tune)")
outfile2 = os.path.join(OUTPUT_DIR, "m12_vs_z_test.png")
fig2.savefig(outfile2, dpi=150)
print(f"wrote {outfile2}")

plot_zoomed_x_vs_z(
    transmitted_x2, [ax[0] for ax in all_ax2], N_STEPS, FP2_ZOOM_MIN, FP2_ZOOM_MAX,
    "M12(z) zoomed near FP2: X trajectories vs Z, colored by initial angle_X",
    os.path.join(OUTPUT_DIR, "m12_vs_z_test_zoomFP2.png"))
