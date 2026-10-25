"""
Joint (slope, intercept) confidence regions for the FP3 M12 and M16
experimental fits, side by side in one figure -- per user request
2026-08-18, extending plot_m12_slope_intercept_contour.py (M12 only) to
also cover M16 (combined 8-group fit: groups 3,4,5,6,7,8,9,10 -- the fit
used as the actual FP3 M16 target, per the user's 2026-08-15 choice).

Same method for both: the OLS parameter covariance matrix
Cov(beta) = sigma_hat^2 * (X^T X)^-1 captures slope/intercept variance
AND their correlation (unlike treating them as independent), plotted as
confidence ellipses. Each panel also overlays the 46 FP2-tuned
candidates' own simulated (slope, intercept) at FP3, to show directly
how far their natural response sits from the experimental fit's own
uncertainty region.

Run:
    cd AgentSearch
    python3 plot_m12_m16_slope_intercept_contour.py
"""
import os
import csv
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from fp3_metrics import DALPHAX_GROUPS, DALPHAX_GROUPS_MEASURED_X, DE_GROUPS, DE_GROUPS_MEASURED_X

FP3_FAMILY_FILE = "Results/fp3_family_1sigma_results.csv"
FP3_CORRECTED_FILE = "Results/fp3_family_mc_results.csv"
OUTPUT_DIR = "Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def fit_covariance(x, y):
    n = len(x)
    X = np.column_stack([np.ones(n), x])  # beta = [intercept, slope]
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    intercept, slope = beta
    rss = float(np.sum((y - X @ beta) ** 2))
    dof = n - 2
    sigma_hat2 = rss / dof
    cov = sigma_hat2 * np.linalg.inv(X.T @ X)  # order [intercept, slope]
    return slope, intercept, cov, dof


def plot_panel(ax, x, y, sim_slopes, sim_intercepts, title, xlabel,
                corrected_slopes=None, corrected_intercepts=None, cov_scale=1.0):
    slope, intercept, cov, dof = fit_covariance(x, y)
    order = [1, 0]  # -> [slope, intercept]
    mean_pt = np.array([slope, intercept])
    cov_si = cov[np.ix_(order, order)] * cov_scale
    corr = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])  # correlation is scale-invariant, unaffected by cov_scale

    res = stats.linregress(x, y)
    print(f"{title}: slope={slope:.4f}+/-{np.sqrt(cov[1,1]):.4f}  intercept={intercept:.4f}+/-{np.sqrt(cov[0,0]):.4f}  "
          f"corr={corr:.4f}  (n={len(x)}, dof={dof})")
    print(f"  scipy cross-check: slope={res.slope:.4f}+/-{res.stderr:.4f}  intercept={res.intercept:.4f}+/-{res.intercept_stderr:.4f}")

    eigvals, eigvecs = np.linalg.eigh(cov_si)
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    for n_sigma, chi2_val, color in [(1, 2.2957, "tab:blue"), (2, 6.1801, "tab:orange")]:
        width, height = 2 * np.sqrt(eigvals * chi2_val)
        ell = Ellipse(xy=mean_pt, width=width, height=height, angle=angle,
                      facecolor="none", edgecolor=color, linewidth=2,
                      label=f"{n_sigma}-sigma joint region (2 dof)")
        ax.add_patch(ell)

    ax.scatter(*mean_pt, color="black", marker="x", s=80, zorder=5, label="experimental best fit")
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.axvline(0, color="grey", linewidth=0.5)

    if len(sim_slopes):
        ax.scatter(sim_slopes, sim_intercepts, color="tab:green", marker="o", s=20, alpha=0.6,
                   zorder=4, label="46 FP2-tuned candidates' own FP3 fit (pre-correction)")

    if corrected_slopes is not None and len(corrected_slopes):
        ax.scatter(corrected_slopes, corrected_intercepts, color="tab:red", marker="^", s=30, alpha=0.7,
                   zorder=4, label=f"{len(corrected_slopes)} FP3-corrected candidates")

    margin_x = max(3 * np.sqrt(eigvals[0]), 3 * np.sqrt(cov[1, 1]))
    margin_y = max(3 * np.sqrt(eigvals[1]), 3 * np.sqrt(cov[0, 0]))
    xlo, xhi = mean_pt[0] - margin_x, mean_pt[0] + margin_x
    ylo, yhi = mean_pt[1] - margin_y, mean_pt[1] + margin_y
    if len(sim_slopes):
        xlo, xhi = min(xlo, sim_slopes.min() - 0.05), max(xhi, sim_slopes.max() + 0.05)
        ylo, yhi = min(ylo, sim_intercepts.min() - 0.2), max(yhi, sim_intercepts.max() + 0.2)
    if corrected_slopes is not None and len(corrected_slopes):
        xlo, xhi = min(xlo, corrected_slopes.min() - 0.05), max(xhi, corrected_slopes.max() + 0.05)
        ylo, yhi = min(ylo, corrected_intercepts.min() - 0.2), max(yhi, corrected_intercepts.max() + 0.2)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("intercept")
    ax.set_title(f"{title}\n(n={len(x)}, dof={dof}); correlation={corr:.3f}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


if __name__ == "__main__":
    sim_M12, sim_Off12, sim_M16, sim_Off16 = [], [], [], []
    if os.path.exists(FP3_FAMILY_FILE):
        rows = list(csv.DictReader(open(FP3_FAMILY_FILE)))
        sim_M12 = np.array([float(r["M12_sim"]) for r in rows])
        sim_Off12 = np.array([float(r["Offset12_sim"]) for r in rows])
        sim_M16 = np.array([float(r["M16_sim"]) for r in rows])
        sim_Off16 = np.array([float(r["Offset16_sim"]) for r in rows])

    corr_M12, corr_Off12, corr_M16, corr_Off16 = None, None, None, None
    if os.path.exists(FP3_CORRECTED_FILE):
        rows = [r for r in csv.DictReader(open(FP3_CORRECTED_FILE)) if r["status"] == "e2_passed"]
        corr_M12 = np.array([float(r["M12_sim"]) for r in rows])
        corr_Off12 = np.array([float(r["Offset12_sim"]) for r in rows])
        corr_M16 = np.array([float(r["M16_sim"]) for r in rows])
        corr_Off16 = np.array([float(r["Offset16_sim"]) for r in rows])
        print(f"{len(rows)} FP3-corrected (e2_passed) candidates loaded from {FP3_CORRECTED_FILE}\n")

    fig, (ax_m12, ax_m16) = plt.subplots(1, 2, figsize=(16, 7))

    m12_groups = list(DALPHAX_GROUPS.keys())
    x12 = np.array([DALPHAX_GROUPS[g] for g in m12_groups])
    y12 = np.array([DALPHAX_GROUPS_MEASURED_X[g] for g in m12_groups])
    plot_panel(ax_m12, x12, y12, sim_M12, sim_Off12,
               "FP3 M12 experimental fit (groups 10,14,15)\ncovariance/4 (tightened, matches fp3_family_mc.py)", "M12 slope",
               corrected_slopes=corr_M12, corrected_intercepts=corr_Off12, cov_scale=0.25)

    m16_groups = list(DE_GROUPS.keys())
    x16 = np.array([DE_GROUPS[g] for g in m16_groups])
    y16 = np.array([DE_GROUPS_MEASURED_X[g] for g in m16_groups])
    plot_panel(ax_m16, x16, y16, sim_M16, sim_Off16,
               "FP3 M16 experimental fit (groups 3,4,5,6,7,8,9,10, combined)", "M16 slope",
               corrected_slopes=corr_M16, corrected_intercepts=corr_Off16)

    fig.tight_layout()
    outfile = os.path.join(OUTPUT_DIR, "m12_m16_slope_intercept_contour.png")
    fig.savefig(outfile, dpi=150)
    print(f"\nwrote {outfile}")
