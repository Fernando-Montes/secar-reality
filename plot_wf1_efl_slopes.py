"""
FP2 E2 slope plots for the WF1 EFL scenarios (2026-10-19), styled after the
earlier "M12/M16 at FP2 (VD1638)" figure: M12 (X_mean vs dalphaX) on top,
M16 (X_mean vs dE) below, at VD1638.

Measured E2 data: black markers (circle = 1 MM pepperpot, triangle = 20 mrad
pepperpot, from groupsInfo labels) with the group number next to each, and
their linear fit (black dashed). Simulated: linear fits only (no markers), for
ECAT at nominal (orange dashed) and efl-d7's six WF1 scenarios (solid). All
simulations use nominal pg_June2025_corrected optics, no quad corrections.

Input: Results/wf1_efl_scenarios_group_means.csv (analyse_wf1_efl_scenarios.py).

Run:
    cd AgentSearch
    python3 plot_wf1_efl_slopes.py
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fp2_metrics as m
from ViewerAnalysis import groupsInfo

IN = "Results/wf1_efl_scenarios_group_means.csv"
OUT = "Results/wf1_efl_scenarios_slopes_FP2.png"
LABELS = {'1': 'Scenario 1: COSY E and B', '2': 'Scenario 2: Danfysik E and B',
          '3': 'Scenario 3: B EFL 2337 mm, COSY tails', '4': 'Scenario 4: B EFL 2382 mm, COSY tails',
          '5': 'Scenario 5: B EFL 2337 mm, Danfysik tails', '6': 'Scenario 6: B EFL 2382 mm, Danfysik tails'}

means = {}
for r in csv.DictReader(open(IN)):
    if r['plane'] == 'FP2':
        means.setdefault(r['case'], {})[r['group']] = float(r['mean_x_mm'])
scen = sorted((c for c in means if c != 'ECAT'), key=lambda s: (len(s), s))
cmap = plt.get_cmap('tab10')
colors = {c: cmap([0, 2, 3, 4, 5, 6][i % 6]) for i, c in enumerate(scen)}

fig, axes = plt.subplots(2, 1, figsize=(10, 13))
for ax, (gdict, meas, xlab, name) in zip(axes, (
        (m.DALPHAX_GROUPS, m.DALPHAX_GROUPS_MEASURED_X, "dalphaX [mrad]", "M12: X_mean vs dalphaX at VD1638 (FP2)"),
        (m.DE_GROUPS, m.DE_GROUPS_MEASURED_X, "dE [%]", "M16: X_mean vs dE at VD1638 (FP2)"))):
    gs = list(gdict)
    xv = np.array([gdict[g] for g in gs])
    xx = np.array([xv.min(), xv.max()])

    for c in scen:
        q = np.polyfit(xv, [means[c][g] for g in gs], 1)
        ax.plot(xx, np.polyval(q, xx), color=colors[c], lw=1.4, label=LABELS.get(c, f"Scenario {c}"))
    q = np.polyfit(xv, [means['ECAT'][g] for g in gs], 1)
    ax.plot(xx, np.polyval(q, xx), color='tab:orange', lw=2.2, ls='--', label="ECAT nominal")

    ym = np.array([meas[g] for g in gs])
    p = np.polyfit(xv, ym, 1)
    ax.plot(xx, np.polyval(p, xx), color='black', lw=2.2, ls='--', label="Exp.")
    for g, x0, y0 in zip(gs, xv, ym):
        mk = '^' if '20 mrad' in groupsInfo[g]['pp_alignment']['label'] else 'o'
        ax.plot(x0, y0, mk, color='black', ms=9, zorder=5)
        ax.annotate(g, (x0, y0), textcoords="offset points", xytext=(7, 6), fontsize=10)

    ax.set_title(name, fontsize=13)
    ax.set_xlabel(xlab)
    ax.set_ylabel("X [mm]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc='best')
fig.suptitle("M12/M16 at FP2 (VD1638): ECAT nominal vs. WF1 EFL scenarios (efl-d7 model for WF1, nominal quads)\n"
             "circle/triangle = measured (1 MM / 20 mrad), number = group", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT, dpi=150)
print(f"wrote {OUT}")
