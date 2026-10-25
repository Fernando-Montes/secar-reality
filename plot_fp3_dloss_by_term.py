"""
dLoss/dknob split by observable (2026-10-05), from the 46-candidate
derivative run (fp3_knob_ranking_results.csv), which stores M12, Offset12,
M16, Offset16 at +-step for every knob at every candidate.

The FP3 loss is r^T H r with r = obs - target (fp3_loss): one Mahalanobis term
for the correlated (M12, Offset12) pair, plus independent M16 and Offset16
terms. Its derivative splits EXACTLY into one contribution per observable:
    dLoss/dk = sum_i  c_i,   c_i = 2 (H r)_i * dobs_i/dk
For M16 and Offset16, c_i is exactly the derivative of that term; for M12 and
Offset12 the shared Mahalanobis term is split by which observable the knob
moves. The four panels add up to the total dLoss/dk.

Units equated as in the equal-unit ranking: 1% (B_SC, dB) = 1 mm (X, Y)
= 0.5 deg (roll) = 0.1 deg (yaw, pitch). All 42 knobs are considered; each
panel shows the knobs with the largest |median contribution| (TOP_N).
Signed: negative = increasing the knob lowers the loss through that
observable. Median over candidates, whiskers 16-84th percentile.

Output: Results/fp3_dloss_by_term.png, Results/fp3_dloss_by_term.csv

Run:
    cd AgentSearch
    python3 plot_fp3_dloss_by_term.py
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fp3_loss
from fp3_knob_ranking import KNOBS, RESULTS_FILE

OBS = ['M12', 'Offset12', 'M16', 'Offset16']
TOP_N = 15
OUT_PNG = "Results/fp3_dloss_by_term.png"
OUT_CSV = "Results/fp3_dloss_by_term.csv"


def unit_of(k):
    if k.endswith('_B_SC') or k.endswith('_Exit_dB'):
        return 0.01, '1%'
    if k.endswith('_X') or k.endswith('_Y'):
        return 0.001, '1 mm'
    if k.endswith('_Roll'):
        return 0.5, '0.5 deg'
    return 0.1, '0.1 deg'


S = {}
for r in csv.DictReader(open(RESULTS_FILE)):
    S[((r["source"], int(r["iteration"])), r["knob"], int(r["sign"]))] = np.array([float(r[o]) for o in OBS])
cands = sorted({c for c, k, _ in S if k == 'base'})

C = {k: [] for k in KNOBS}                            # knob -> list over candidates of 4 contributions
for c in cands:
    o0 = S[(c, 'base', 0)]
    g = 2 * fp3_loss.H @ (o0 - fp3_loss.TARGETS)     # dLoss/dobs
    for k, (_, step, _, _) in KNOBS.items():
        dobs = (S[(c, k, 1)] - S[(c, k, -1)]) / (2 * step) * unit_of(k)[0]
        C[k].append(g * dobs)
C = {k: np.array(v) for k, v in C.items()}            # (n_cands, 4)

# check: the four contributions add up to the total chain-rule derivative
tot = max(abs(C[k].sum(axis=1) - np.array([fp3_loss.loss_gradient(S[(c, 'base', 0)],
          (S[(c, k, 1)] - S[(c, k, -1)]) / (2 * KNOBS[k][1]) * unit_of(k)[0]) for c in cands])).max()
          for k in KNOBS)
print(f"{len(cands)} candidates; max |sum of 4 contributions - total dLoss/dk| = {tot:.1e}")

with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["knob", "unit"] + [f"{o}_{s}" for o in OBS for s in ("median", "p16", "p84")] + ["total_median"])
    for k in KNOBS:
        row = [k, unit_of(k)[1]]
        for i in range(4):
            row += [np.median(C[k][:, i]), np.percentile(C[k][:, i], 16), np.percentile(C[k][:, i], 84)]
        w.writerow(row + [np.median(C[k].sum(axis=1))])

colors = {'1%': 'tab:green', '1 mm': 'tab:orange', '0.5 deg': 'tab:purple', '0.1 deg': 'tab:pink'}
labels = {'M12': "M12 (shared M12/Offset12 term)", 'Offset12': "Offset12 (shared M12/Offset12 term)",
          'M16': "M16 term", 'Offset16': "Offset16 term"}
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
for ax, i, o in zip(axes.flat, range(4), OBS):
    med = {k: np.median(C[k][:, i]) for k in KNOBS}
    top = sorted(KNOBS, key=lambda k: -abs(med[k]))[:TOP_N]
    y = np.arange(len(top))
    m = np.array([med[k] for k in top])
    lo = m - np.array([np.percentile(C[k][:, i], 16) for k in top])
    hi = np.array([np.percentile(C[k][:, i], 84) for k in top]) - m
    ax.barh(y, m, color=[colors[unit_of(k)[1]] for k in top])
    ax.errorbar(m, y, xerr=[lo, hi], fmt="none", ecolor="black", capsize=2, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{k} ({unit_of(k)[1]})" for k in top], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("contribution to dLoss per unit change  (median over 46 candidates)", fontsize=8)
    r_med = np.median([S[(c, 'base', 0)][i] - fp3_loss.TARGETS[i] for c in cands])
    ax.set_title(f"{labels[o]}\n(current miss: {o} - target = {r_med:+.3f})", fontsize=10)
    ax.grid(alpha=0.3, axis="x")
    print(f"\n{o}: top {TOP_N}")
    for k in top:
        print(f"  {k:<12} {unit_of(k)[1]:>8} {med[k]:+10.2f}")
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
fig.legend(handles, [f"1 unit = {u}" for u in colors], loc="lower center", ncol=4, fontsize=9)
fig.suptitle("FP3: dLoss per unit change, split by observable (the four panels add up to the total dLoss)\n"
             "units equated: 1% = 1 mm = 0.5 deg roll = 0.1 deg yaw/pitch; negative = increasing the knob lowers the loss",
             fontsize=11)
fig.tight_layout(rect=(0, 0.03, 1, 0.95))
fig.savefig(OUT_PNG, dpi=150)
print(f"\nwrote {OUT_PNG} and {OUT_CSV}")
