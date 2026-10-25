"""
Reports the size and shape of the "acceptable solution family" from the
corrected (2-term, no Q6 prior) BO search in fp2_bo_search_v2_results.csv.

loss = t12 + t16 is a sum of two independent normalized-squared-residuals,
i.e. approximately chi-square distributed with 2 dof. Standard delta-chi-square
joint confidence regions apply directly on top of the achieved minimum:
    68% CL (1 sigma-equivalent, 2 dof): loss - loss_min < 2.30
    95% CL              (2 dof): loss - loss_min < 6.18
Every evaluated point (BO-proposed AND OAT warm-start rows) is included --
this characterizes the whole explored loss landscape, not just the BO's
own proposals.

Run:
    cd AgentSearch
    python3 fp2_bo_v2_family_analysis.py
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_FILE = "Results/fp2_bo_search_v2_results.csv"
OUTPUT_DIR = "Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DELTA_68 = 2.30
DELTA_95 = 6.18

KNOBS = ["Q6_B_SC", "Q2_B_SC", "Q5_B_SC"]


def load_rows():
    rows = []
    with open(RESULTS_FILE, "r", newline="") as f:
        for row in csv.DictReader(f):
            rows.append({k: (float(v) if k != "source" else v) for k, v in row.items()})
    return rows


def summarize(name, family):
    print(f"\n{name}: {len(family)} / {len(rows)} points")
    if not family:
        return
    for k in KNOBS:
        vals = np.array([r[k] for r in family]) * 100
        print(f"  {k:10s}: [{vals.min():+.2f}%, {vals.max():+.2f}%]  (range {vals.max()-vals.min():.2f} pts)")


if __name__ == "__main__":
    rows = load_rows()
    best = min(rows, key=lambda r: r["loss"])
    loss_min = best["loss"]
    print(f"Loaded {len(rows)} evaluated points.")
    print(f"Global best: loss={loss_min:.3f} at Q6={best['Q6_B_SC']*100:+.2f}% "
          f"Q2={best['Q2_B_SC']*100:+.2f}% Q5={best['Q5_B_SC']*100:+.2f}%  "
          f"(M12={best['M12_sim']:.4f}, M16={best['M16_sim']:.4f})")

    family_68 = [r for r in rows if r["loss"] - loss_min < DELTA_68]
    family_95 = [r for r in rows if r["loss"] - loss_min < DELTA_95]
    summarize("68% CL family (Dloss < 2.30)", family_68)
    summarize("95% CL family (Dloss < 6.18)", family_95)

    # ---------------- pairwise plots highlighting the families ----------------
    all_pts = {k: np.array([r[k] for r in rows]) * 100 for k in KNOBS}
    pts_68 = {k: np.array([r[k] for r in family_68]) * 100 for k in KNOBS}
    pts_95 = {k: np.array([r[k] for r in family_95]) * 100 for k in KNOBS}

    pairs = [("Q6_B_SC", "Q2_B_SC"), ("Q6_B_SC", "Q5_B_SC"), ("Q2_B_SC", "Q5_B_SC")]
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, (kx, ky) in zip(axs, pairs):
        ax.scatter(all_pts[kx], all_pts[ky], c="lightgray", s=25, label="all evaluated", zorder=1)
        ax.scatter(pts_95[kx], pts_95[ky], c="tab:orange", s=35, label="95% CL family", zorder=2)
        ax.scatter(pts_68[kx], pts_68[ky], c="tab:red", s=45, label="68% CL family", zorder=3)
        ax.scatter([best[kx]*100], [best[ky]*100], c="black", marker="*", s=250,
                   label="global best", zorder=4, edgecolor="white", linewidth=0.8)
        ax.set_xlabel(f"{kx} [%]")
        ax.set_ylabel(f"{ky} [%]")
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=8, loc="best")
    fig.suptitle("BO v2 (M12+M16-only loss): acceptable-solution family across knob space", fontsize=12)
    fig.tight_layout()
    outfile = os.path.join(OUTPUT_DIR, "bo_v2_family_pairwise.png")
    fig.savefig(outfile, dpi=200)
    print(f"\nwrote {outfile}")
