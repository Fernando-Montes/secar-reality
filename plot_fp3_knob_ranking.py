"""
Plot the FP3 knob ranking from Results/fp3_knob_ranking_results.csv
(fp3_knob_ranking.py). Same layout as the FP2 oat_derivative_ranking.png:
one panel per native unit (%, mm, deg), knobs sorted by |dLoss/dknob|.

dLoss/dknob at each candidate's base point, by the chain rule on the four
observables (as in the FP2 ranking): dLoss/dk = 2 r^T H dobs/dk, with
dobs/dk the central difference (obs(+h) - obs(-h)) / 2h and r = obs(base)
- targets. The direct central difference of the loss itself is also written
to the summary CSV as a cross-check.

With one candidate: bars are that candidate's signed derivatives. With
several: bars are the median of |dLoss/dk| across candidates (the ranking
quantity), whiskers its 16th-84th percentile range, colored by how
consistently the sign agrees across candidates. Only candidates whose
base point and both signs of a knob are present are used for that knob,
so this can be run while fp3_knob_ranking.py is still going.

Run:
    cd AgentSearch
    python3 plot_fp3_knob_ranking.py
"""
import os
import csv
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fp3_loss
from fp3_knob_ranking import KNOBS

RESULTS_FILE = "Results/fp3_knob_ranking_results.csv"
OUT_PNG = "Results/fp3_knob_ranking.png"
OUT_CSV = "Results/fp3_knob_ranking_summary.csv"

pts = {}
for r in csv.DictReader(open(RESULTS_FILE)):
    pts[(r["source"], int(r["iteration"]), r["knob"], int(r["sign"]))] = (
        np.array([float(r[k]) for k in ("M12", "Offset12", "M16", "Offset16")]), float(r["loss"]))
cands = sorted({(s, i) for s, i, k, _ in pts if k == 'base'})

deriv = defaultdict(dict)          # knob -> {cand: dLoss per display unit (chain rule)}
deriv_direct = defaultdict(dict)   # knob -> {cand: direct central difference of the loss}
for c in cands:
    obs0, _ = pts[(c[0], c[1], 'base', 0)]
    for k, (_, step, unit, factor) in KNOBS.items():
        p, m = pts.get((c[0], c[1], k, 1)), pts.get((c[0], c[1], k, -1))
        if p is None or m is None:
            continue
        dobs = (p[0] - m[0]) / (2 * step)                      # per native unit
        deriv[k][c] = fp3_loss.loss_gradient(obs0, dobs) / factor
        deriv_direct[k][c] = (p[1] - m[1]) / (2 * step) / factor

if not deriv:
    raise SystemExit("No complete knob points yet.")
n_c = max(len(v) for v in deriv.values())
print(f"{len(cands)} candidate(s) with a base point; up to {n_c} per knob.")

rows = []
for k, d in deriv.items():
    v = np.array(list(d.values()))
    vd = np.array([deriv_direct[k][c] for c in d])
    rows.append({"knob": k, "unit": KNOBS[k][2], "n_candidates": len(v),
                 "median": float(np.median(v)), "median_abs": float(np.median(np.abs(v))),
                 "abs_p16": float(np.percentile(np.abs(v), 16)),
                 "abs_p84": float(np.percentile(np.abs(v), 84)),
                 "frac_positive": float(np.mean(v > 0)),
                 "median_direct_fd": float(np.median(vd))})
rows.sort(key=lambda r: -r["median_abs"])
with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

units = ['%', 'mm', 'deg']
heights = [max(1, sum(r["unit"] == u for r in rows)) for u in units]
fig, axes = plt.subplots(3, 1, figsize=(10, 3 + 0.32 * sum(heights)),
                         gridspec_kw={'height_ratios': heights})
for ax, u in zip(axes, units):
    sel = sorted([r for r in rows if r["unit"] == u], key=lambda r: r["median_abs"])
    if not sel:
        ax.set_visible(False)
        continue
    y = np.arange(len(sel))
    if n_c == 1:
        # single candidate: signed derivative, as in the FP2 chart
        med = np.array([r["median"] for r in sel])
        ax.barh(y, med, color=np.where(med > 0, "tab:red", "tab:blue"))
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel(f"dLoss / d({u})")
    else:
        # several: magnitude (what the ranking uses); color = sign agreement
        mag = np.array([r["median_abs"] for r in sel])
        fp = np.array([r["frac_positive"] for r in sel])
        ax.barh(y, mag, color=np.where(fp >= 0.8, "tab:red", np.where(fp <= 0.2, "tab:blue", "tab:gray")))
        ax.errorbar(mag, y, xerr=[mag - [r["abs_p16"] for r in sel], [r["abs_p84"] for r in sel] - mag],
                    fmt="none", ecolor="black", capsize=2, lw=0.8)
        ax.set_xlabel(f"|dLoss / d({u})|   (red: >=80% of candidates positive, "
                      f"blue: >=80% negative, grey: sign varies)")
    ax.set_yticks(y)
    ax.set_yticklabels([r["knob"] for r in sel], fontsize=8)
    ax.set_title(f"Knobs with native unit '{u}' -- sorted by |derivative|")
    ax.grid(alpha=0.3, axis="x")
what = ("single candidate" if n_c == 1
        else f"median |dLoss/dk| over {n_c} FP2 candidates, whiskers 16-84th percentile")
fig.suptitle(f"FP3 loss sensitivity (E2 Mahalanobis loss), local derivative at the FP2 solutions\n"
             f"corrected lattice (WF2 aligned) -- {what}", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(OUT_PNG, dpi=150)
print(f"wrote {OUT_PNG} and {OUT_CSV}")
print("top 10 by median |dLoss/dk|:")
for r in rows[:10]:
    print(f"  {r['knob']:<16} median |d| {r['median_abs']:10.3f} per {r['unit']:<3}  "
          f"signed median {r['median']:+10.3f}  positive in {100*r['frac_positive']:.0f}%  "
          f"(direct FD {r['median_direct_fd']:+10.3f})")
