"""
Histogram/visual acceptance test for the BO-found best point (2026-08-08):
    Q6_B_SC = -0.63%, Q2_B_SC = -4.13%, Q5_B_SC = +8.65%   (loss=17.18)

Adapted from ViewerAnalysis.py's load_projection (same chi-squared/plot
logic), but made group-aware: that function's save path is hardcoded
per-viewer only (f"{results_directory}/imageCompared_VD{viewer}"), which
would overwrite between our 11 E2 groups since we're always looking at
VD1638. This copy saves one file per group instead.

Run:
    cd AgentSearch
    python3 fp2_histogram_check.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import element_names_stripped, viewers
from helper_functions import runECAT
from ViewerAnalysis import groupsInfo
from fp2_metrics import build_initial_distribution, ALL_GROUPS

BEST_TUNECHANGES = [['Q6', 'B_SC', -0.0063], ['Q2', 'B_SC', -0.0413], ['Q5', 'B_SC', 0.0865]]
TA_Y_OFFSET = 0.0  # matches what the BO search assumed throughout
NRAYS = 5000        # higher than the search's 1000 -- this is a one-time visual check, not a batch search
VD1638_INDEX = element_names_stripped.index('VD1638')
OUTPUT_DIR = "Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# PartD_ViewerAnalysis lives in the parent ECAT directory (read-only
# measurement data), not inside AgentSearch/ -- absolute path to avoid
# relying on cwd.
PARTD_DIR = "/Users/fernandomontes/Research/COSY/MC/ECAT/PartD_ViewerAnalysis"


def plot_group_comparison(group, all_x, all_y, all_ax, all_dE, transmission_indices, cosyFile):
    posV = viewers['1638']['index']
    posS = element_names_stripped.index('FP1 Slits')
    slit_lo, slit_hi = groupsInfo[group]['fp1_slits']['slits_mm']
    slit_label = groupsInfo[group]['fp1_slits']['label']

    data_path = f"{PARTD_DIR}/Group{group}"
    x_proj_path = os.path.join(data_path, f"VD1638_Group{group}_xf_cosy.csv")
    y_proj_path = os.path.join(data_path, f"VD1638_Group{group}_yf_cosy.csv")
    image_2d_path = os.path.join(data_path, f"Group{group}_VD1638_2D_cosy.npz")

    df_x = pd.read_csv(x_proj_path, comment="#")
    df_y = pd.read_csv(y_proj_path, comment="#")
    x_pos = -df_x.iloc[:, 0].to_numpy()
    x_intensity = df_x.iloc[:, 1].to_numpy()
    y_pos = df_y.iloc[:, 0].to_numpy()
    y_intensity = df_y.iloc[:, 1].to_numpy()

    x_bins = 50
    x_scale = x_bins / len(x_pos)
    x_intensityC = [i * x_scale for i in x_intensity]
    x_intensityC_sum = np.sum(x_intensityC)
    x_intensityC = [max(0, round(100 * x_bins * i / x_intensityC_sum)) for i in x_intensityC]
    x_intensityC[0] = max(1, x_intensityC[0]); x_intensityC[-1] = max(1, x_intensityC[-1])
    x_intensity_scaled = [100 * x_bins * i / x_intensityC_sum for i in x_intensity]
    x_vd = [p for p, n in zip(x_pos, x_intensityC) for _ in range(n)]

    y_bins = 50
    y_scale = y_bins / len(y_pos)
    y_intensityC = [i * y_scale for i in y_intensity]
    y_intensityC_sum = np.sum(y_intensityC)
    y_intensityC = [max(0, round(100 * y_bins * i / y_intensityC_sum)) for i in y_intensityC]
    y_intensityC[0] = max(1, y_intensityC[0]); y_intensityC[-1] = max(1, y_intensityC[-1])
    y_intensity_scaled = [100 * y_bins * i / y_intensityC_sum for i in y_intensity]
    y_vd = [p for p, n in zip(y_pos, y_intensityC) for _ in range(n)]

    # FP1-slits-then-VD1638 filter, same convention as fp2_metrics.py / ViewerAnalysis.py
    x_list, y_list = [], []
    for i in range(len(all_x)):
        if transmission_indices[i] >= posV and slit_lo <= -all_x[i][posS] * 1000 <= slit_hi:
            x_list.append(-all_x[i][posV] * 1000)
            y_list.append(all_y[i][posV] * 1000)
    x = np.array(x_list)
    y = np.array(y_list)

    x_edges = np.histogram_bin_edges(x_vd, bins=x_bins)
    vd_counts, _ = np.histogram(x_vd, bins=x_edges)
    ecat_counts, _ = np.histogram(x, bins=x_edges)
    widths = np.diff(x_edges)
    area_vd = np.sum(vd_counts * widths)
    area_ecat = np.sum(ecat_counts * widths)
    x_ecat_scale = area_vd / area_ecat if area_ecat > 0 else 0.0
    viewer_x_counts, _ = np.histogram(x_vd, bins=x_bins)
    ecat_x_counts, _ = np.histogram(x, bins=x_edges, weights=np.full(len(x), x_ecat_scale))
    if len(x) == 0 or np.abs(np.mean(x) - np.mean(x_vd)) > np.max(x_pos):
        chisq_x = np.inf
    else:
        chisq_x = np.sum((viewer_x_counts - ecat_x_counts) ** 2) / np.sum(viewer_x_counts)

    y_edges = np.histogram_bin_edges(y_vd, bins=y_bins)
    vd_y_counts, _ = np.histogram(y_vd, bins=y_edges)
    ecat_y_counts_raw, _ = np.histogram(y, bins=y_edges)
    y_widths = np.diff(y_edges)
    area_vd_y = np.sum(vd_y_counts * y_widths)
    area_ecat_y = np.sum(ecat_y_counts_raw * y_widths)
    y_ecat_scale = area_vd_y / area_ecat_y if area_ecat_y > 0 else 0.0
    viewer_y_counts, _ = np.histogram(y_vd, bins=y_bins)
    ecat_y_counts = np.histogram(y, bins=y_edges, weights=np.full(len(y), y_ecat_scale))[0]
    chisq_y = np.sum((viewer_y_counts - ecat_y_counts) ** 2) / np.sum(viewer_y_counts) if len(y) else np.inf

    data = np.load(image_2d_path)
    corrected_full_image = data["corrected_full_image"][:, ::-1]
    x2d_range = data["xf_range"]
    y2d_range = data["yf_range"]

    fig, axs = plt.subplots(2, 2, figsize=(17, 9))
    plt.suptitle(f"Group {group} - VD1638 - {slit_label}\n"
                 f"BO best: Q6={BEST_TUNECHANGES[0][2]*100:+.2f}% Q2={BEST_TUNECHANGES[1][2]*100:+.2f}% Q5={BEST_TUNECHANGES[2][2]*100:+.2f}%",
                 fontsize=14)

    axs[0, 0].plot(x_pos, x_intensity_scaled, label="Viewer original")
    axs[0, 0].hist(x_vd, bins=x_bins, histtype="step", color="blue", linewidth=4, label="Viewer binned")
    axs[0, 0].hist(x, bins=x_edges, histtype="step", linewidth=4,
                    weights=np.full(len(x), x_ecat_scale), label=f"ecat x {x_ecat_scale:.3g}", color="orange")
    axs[0, 0].set_xlabel("X [mm]"); axs[0, 0].set_ylabel("Intensity")
    axs[0, 0].set_title("X projection", fontweight="bold")
    axs[0, 0].set_xlim(min(x_pos), max(x_pos))
    axs[0, 0].legend(loc="best")
    axs[0, 0].text(0.01, 0.98, f"chi-sq = {chisq_x:.1f}", transform=axs[0, 0].transAxes,
                    ha="left", va="top", fontsize=12, color="orange", fontweight="bold")

    axs[0, 1].plot(y_pos, y_intensity_scaled, label="Viewer original")
    axs[0, 1].hist(y_vd, bins=y_bins, histtype="step", color="blue", linewidth=4, label="Viewer binned")
    axs[0, 1].hist(y, bins=y_edges, histtype="step", linewidth=4,
                    weights=np.full(len(y), y_ecat_scale), label=f"ecat x {y_ecat_scale:.3g}", color="orange")
    axs[0, 1].set_xlabel("Y [mm]"); axs[0, 1].set_ylabel("Intensity")
    axs[0, 1].set_title("Y projection", fontweight="bold")
    axs[0, 1].set_xlim(min(y_pos), max(y_pos))
    axs[0, 1].legend(loc="best")
    axs[0, 1].text(0.01, 0.98, f"chi-sq = {chisq_y:.1f}", transform=axs[0, 1].transAxes,
                    ha="left", va="top", fontsize=12, color="orange", fontweight="bold")

    im = axs[1, 0].imshow(corrected_full_image, extent=(-x2d_range[-1], -x2d_range[0], y2d_range[-1], y2d_range[0]),
                           cmap="viridis", aspect="auto")
    fig.colorbar(im, ax=axs[1, 0])
    axs[1, 0].set_xlim(-x2d_range[-1], -x2d_range[0]); axs[1, 0].set_ylim(y2d_range[0], y2d_range[-1])
    axs[1, 0].set_xlabel("X (mm)"); axs[1, 0].set_ylabel("Y (mm)")
    axs[1, 0].set_title("Viewer", fontweight="bold")

    axs[1, 1].hist2d(x, y, bins=(50, 50), cmap="viridis",
                      range=[[-x2d_range[-1], -x2d_range[0]], [y2d_range[0], y2d_range[-1]]])
    axs[1, 1].set_xlabel("X (mm)"); axs[1, 1].set_ylabel("Y (mm)")
    axs[1, 1].set_title("ECAT", fontweight="bold")

    plt.tight_layout()
    outfile = os.path.join(OUTPUT_DIR, f"histogram_check_Group{group}_VD1638.png")
    plt.savefig(outfile, dpi=200)
    plt.close(fig)
    print(f"Group {group}: chi-sq_x={chisq_x:.1f}  chi-sq_y={chisq_y:.1f}  n_rays={len(x)}  -> {outfile}")
    return chisq_x, chisq_y


if __name__ == "__main__":
    total_chisq = 0.0
    cosyFile = "pg_June2025_histcheck"
    first = True
    for group in ALL_GROUPS:
        dist = build_initial_distribution(group, NRAYS, ta_y_offset=TA_Y_OFFSET)
        all_x, all_ax, all_y, all_ay, all_dE, transmitted_x, transmitted_y, transmission_indices, \
            chamber_names, beampipes, usedFoxFile = runECAT(
                initialDistribution=dist, run_cosy_flag=first, foxFile=cosyFile,
                tuneChanges=BEST_TUNECHANGES, save_rays_flag=False, final_index=VD1638_INDEX)
        if first:
            cosyFile = usedFoxFile
            first = False
        chisq_x, chisq_y = plot_group_comparison(group, all_x, all_y, all_ax, all_dE, transmission_indices, cosyFile)
        total_chisq += chisq_x + chisq_y

    print(f"\nTotal chi-sq (sum over {len(ALL_GROUPS)} groups, X+Y): {total_chisq:.1f}")
