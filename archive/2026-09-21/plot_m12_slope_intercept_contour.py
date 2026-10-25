"""
Joint (slope, intercept) confidence region for the FP3 M12 experimental
fit (groups 10, 14, 15; dX/dalphaX at VD1783), per user request
2026-08-18 -- prompted by finding the M12 acceptance sigma (0.04,
chosen by eye against how well the manual best-fit matched the
experimental mean, not against the fit's own stderr) implies the 46
FP2-tuned candidates' natural FP3 M12 sits ~13 sigma from target, using
independent slope/intercept sigmas that ignore their correlation.

With only 3 data points (dof=1), slope and intercept from an OLS fit are
NOT independent -- the standard OLS parameter covariance matrix
Cov(beta) = sigma_hat^2 * (X^T X)^-1 (X = [1, x_i], sigma_hat^2 =
residual_sum_of_squares/(n-2)) captures both individual variances AND
their covariance. This plots that as confidence ellipses in
(slope, intercept) space -- the actual "good fit" region, rather than
treating the two as independently uncertain.

Run:
    cd AgentSearch
    python3 plot_m12_slope_intercept_contour.py
"""
import os
import csv
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from fp3_metrics import DALPHAX_GROUPS, DALPHAX_GROUPS_MEASURED_X

FP3_FAMILY_FILE = "Results/fp3_family_1sigma_results.csv"

OUTPUT_DIR = "Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

groups = list(DALPHAX_GROUPS.keys())
x = np.array([DALPHAX_GROUPS[g] for g in groups])
y = np.array([DALPHAX_GROUPS_MEASURED_X[g] for g in groups])
n = len(x)

X = np.column_stack([np.ones(n), x])  # design matrix: [1, x_i], beta = [intercept, slope]
beta, residuals_ss, rank, sv = np.linalg.lstsq(X, y, rcond=None)
intercept, slope = beta
yhat = X @ beta
rss = float(np.sum((y - yhat) ** 2))
dof = n - 2
sigma_hat2 = rss / dof
cov = sigma_hat2 * np.linalg.inv(X.T @ X)  # 2x2, order [intercept, slope]

# reorder to [slope, intercept] for plotting (slope on x-axis)
order = [1, 0]
mean_pt = np.array([slope, intercept])
cov_si = cov[np.ix_(order, order)]

print(f"Best fit: slope={slope:.4f} +/- {np.sqrt(cov[1,1]):.4f}   intercept={intercept:.4f} +/- {np.sqrt(cov[0,0]):.4f}")
corr = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
print(f"Correlation(slope, intercept) = {corr:.4f}")

# scipy cross-check (independent marginal stderr, for comparison)
res = stats.linregress(x, y)
print(f"scipy linregress cross-check: slope={res.slope:.4f}+/-{res.stderr:.4f}  "
      f"intercept={res.intercept:.4f}+/-{res.intercept_stderr:.4f}")

fig, ax = plt.subplots(figsize=(8, 7))

eigvals, eigvecs = np.linalg.eigh(cov_si)
angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
# chi2 quantiles for 2 dof: 1-sigma-equivalent (39.4%) and 2-sigma-equivalent (86.5%)
for n_sigma, chi2_val, color in [(1, 2.2957, "tab:blue"), (2, 6.1801, "tab:orange")]:
    width, height = 2 * np.sqrt(eigvals * chi2_val)
    ell = Ellipse(xy=mean_pt, width=width, height=height, angle=angle,
                  facecolor="none", edgecolor=color, linewidth=2,
                  label=f"{n_sigma}-sigma joint region (2 dof)")
    ax.add_patch(ell)

ax.scatter(*mean_pt, color="black", marker="x", s=80, zorder=5, label="experimental best fit")
ax.axhline(0, color="grey", linewidth=0.5)
ax.axvline(0, color="grey", linewidth=0.5)

# Overlay the 46 FP2-tuned candidates' own simulated (M12, Offset12) at
# FP3 -- shows directly how far their natural FP3 response sits from the
# experimental fit's own uncertainty region.
cand_slopes, cand_intercepts = [], []
if os.path.exists(FP3_FAMILY_FILE):
    rows = list(csv.DictReader(open(FP3_FAMILY_FILE)))
    cand_slopes = np.array([float(r["M12_sim"]) for r in rows])
    cand_intercepts = np.array([float(r["Offset12_sim"]) for r in rows])
    ax.scatter(cand_slopes, cand_intercepts, color="tab:green", marker="o", s=20, alpha=0.6,
               zorder=4, label=f"46 FP2-tuned candidates' own FP3 (M12,Offset12)")

margin_x = max(3 * np.sqrt(eigvals[0]), 3 * np.sqrt(cov[1, 1]))
margin_y = max(3 * np.sqrt(eigvals[1]), 3 * np.sqrt(cov[0, 0]))
xlo, xhi = mean_pt[0] - margin_x, mean_pt[0] + margin_x
ylo, yhi = mean_pt[1] - margin_y, mean_pt[1] + margin_y
if len(cand_slopes):
    xlo, xhi = min(xlo, cand_slopes.min() - 0.05), max(xhi, cand_slopes.max() + 0.05)
    ylo, yhi = min(ylo, cand_intercepts.min() - 0.2), max(yhi, cand_intercepts.max() + 0.2)
ax.set_xlim(xlo, xhi)
ax.set_ylim(ylo, yhi)
ax.set_xlabel("M12 slope")
ax.set_ylabel("M12 intercept")
ax.set_title(f"FP3 M12 experimental fit: joint (slope, intercept) confidence region\n"
             f"groups 10,14,15 (n=3, dof=1); correlation={corr:.3f}")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
outfile = os.path.join(OUTPUT_DIR, "m12_slope_intercept_contour.png")
fig.savefig(outfile, dpi=150)
print(f"\nwrote {outfile}")
