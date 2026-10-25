"""
Reads fp2_joint_search_results.csv (can be a partial/in-progress run) and
produces two heatmaps over the (Q6, WF1_dB) grid:
  1. fp2_loss_heatmap.png   -- combined loss (color)
  2. fp2_width_heatmap.png  -- true-FP2-Slits X std (color), the E3-style
                                diagnostic, shown for comparison against
                                where the loss minimum actually lands.
Safe to run on a partial results file (e.g. while the search script is
still running, or after stopping it early) -- just plots whatever rows
exist so far. Run with:
    python3 plot_fp2_search.py
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_FILE = "Results/fp2_joint_search_results.csv"  # moved here 2026-08-08, run complete
OUTPUT_DIR = "Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_results():
    rows = []
    with open(RESULTS_FILE, "r", newline="") as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def make_grid(rows, value_key):
    q6_vals = sorted(set(round(r["Q6"], 6) for r in rows))
    wf1_vals = sorted(set(round(r["WF1_dB"], 6) for r in rows))
    grid = np.full((len(wf1_vals), len(q6_vals)), np.nan)
    q6_idx = {v: i for i, v in enumerate(q6_vals)}
    wf1_idx = {v: i for i, v in enumerate(wf1_vals)}
    for r in rows:
        i = wf1_idx[round(r["WF1_dB"], 6)]
        j = q6_idx[round(r["Q6"], 6)]
        grid[i, j] = r[value_key]
    return np.array(q6_vals), np.array(wf1_vals), grid


def plot_heatmap(q6_vals, wf1_vals, grid, title, cbar_label, outfile, mark_min=True, cmap="viridis"):
    fig, ax = plt.subplots(figsize=(9, 6))
    extent = [q6_vals.min() * 100, q6_vals.max() * 100, wf1_vals.min() * 100, wf1_vals.max() * 100]
    im = ax.imshow(grid, origin="lower", aspect="auto", extent=extent, cmap=cmap)
    fig.colorbar(im, ax=ax, label=cbar_label)
    ax.set_xlabel("Q6 B_SC change [%]")
    ax.set_ylabel("WF1 Exit dB [%]")
    ax.set_title(title)

    if mark_min and np.any(~np.isnan(grid)):
        i, j = np.unravel_index(np.nanargmin(grid), grid.shape)
        ax.plot(q6_vals[j] * 100, wf1_vals[i] * 100, "r*", markersize=18,
                markeredgecolor="white", markeredgewidth=0.8,
                label=f"min: Q6={q6_vals[j]*100:+.2f}%, WF1 dB={wf1_vals[i]*100:+.2f}%")
        ax.legend(loc="upper right", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    print(f"wrote {outfile}")


if __name__ == "__main__":
    rows = load_results()
    print(f"Loaded {len(rows)} completed grid points from {RESULTS_FILE}")
    if not rows:
        raise SystemExit("No results yet -- run fp2_joint_search.py first (or wait for it to write some rows).")

    q6_vals, wf1_vals, loss_grid = make_grid(rows, "loss")
    plot_heatmap(q6_vals, wf1_vals, loss_grid,
                 title="FP2 joint search: combined loss",
                 cbar_label="loss (lower is better)",
                 outfile=os.path.join(OUTPUT_DIR, "fp2_loss_heatmap.png"), cmap="viridis_r")

    _, _, width_grid = make_grid(rows, "fp2_slits_std")
    plot_heatmap(q6_vals, wf1_vals, width_grid,
                 title="True FP2 Slits: X spot width (std)",
                 cbar_label="std(X) at FP2 Slits [mm]",
                 outfile=os.path.join(OUTPUT_DIR, "fp2_width_heatmap.png"), mark_min=False, cmap="magma")

    # Also overlay the loss-minimum location on the width plot, for direct comparison.
    fig, ax = plt.subplots(figsize=(9, 6))
    extent = [q6_vals.min() * 100, q6_vals.max() * 100, wf1_vals.min() * 100, wf1_vals.max() * 100]
    im = ax.imshow(width_grid, origin="lower", aspect="auto", extent=extent, cmap="magma")
    fig.colorbar(im, ax=ax, label="std(X) at FP2 Slits [mm]")
    if np.any(~np.isnan(loss_grid)):
        i, j = np.unravel_index(np.nanargmin(loss_grid), loss_grid.shape)
        ax.plot(q6_vals[j] * 100, wf1_vals[i] * 100, "c*", markersize=18,
                markeredgecolor="white", markeredgewidth=0.8,
                label=f"loss min: Q6={q6_vals[j]*100:+.2f}%, WF1 dB={wf1_vals[i]*100:+.2f}%")
        ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xlabel("Q6 B_SC change [%]")
    ax.set_ylabel("WF1 Exit dB [%]")
    ax.set_title("True FP2 Slits X width, with loss-minimum overlaid")
    fig.tight_layout()
    outfile = os.path.join(OUTPUT_DIR, "fp2_width_with_loss_min.png")
    fig.savefig(outfile, dpi=150)
    print(f"wrote {outfile}")
