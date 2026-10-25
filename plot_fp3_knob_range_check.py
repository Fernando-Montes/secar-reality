"""
Plot fp3_knob_range_check.py results (2026-10-04).

Page 1 -- one ranking for all 42 knobs, independent of unit: the largest
|loss change| from the candidate's base point anywhere in -R..+R (median over
the checked candidates; whiskers = min/max over candidates). Black dots: what
the local derivative predicts for the same range, |dLoss/dk| * R, using
fp3_knob_ranking_results.csv for the same candidates where available. A bar
far above its dot = a knob the small-step ranking underrates (the response
curves); far below = the response saturates or turns over. Knobs with a
'no_rays' point (some group loses all rays somewhere in the range) are
marked with '*'.

Page 2 -- each knob's loss vs setting (-R..+R), one line per candidate,
dashed: the local tangent from the derivative ranking.

Also writes Results/fp3_knob_range_check_summary.csv.

Run:
    cd AgentSearch
    python3 plot_fp3_knob_range_check.py
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from fp3_knob_ranking import KNOBS, RESULTS_FILE as SLOPE_FILE
from fp3_knob_range_check import RESULTS_FILE, RANGES, FRACS

OUT_PDF = "Results/fp3_knob_range_check.pdf"
OUT_CSV = "Results/fp3_knob_range_check_summary.csv"

R = {}                                  # (cand, knob, frac) -> loss or None
for r in csv.DictReader(open(RESULTS_FILE)):
    R[((r["source"], int(r["iteration"])), r["knob"], float(r["frac"]))] = \
        float(r["loss"]) if r["status"] == "ok" else None
cands = sorted({c for c, k, _ in R if k == 'base'})
if not cands:
    raise SystemExit("No base points yet.")

slope = {}                              # (cand, knob) -> dLoss per native unit (direct central difference)
if os.path.exists(SLOPE_FILE):
    S = {}
    for r in csv.DictReader(open(SLOPE_FILE)):
        S[((r["source"], int(r["iteration"])), r["knob"], int(r["sign"]))] = float(r["loss"])
    for c in cands:
        for k, (_, step, _, _) in KNOBS.items():
            if (c, k, 1) in S and (c, k, -1) in S:
                slope[(c, k)] = (S[(c, k, 1)] - S[(c, k, -1)]) / (2 * step)

rows = []
for k in KNOBS:
    per_c, pred, broken = [], [], False
    for c in cands:
        base = R.get((c, 'base', 0.0))
        pts = [R.get((c, k, f), 'missing') for f in FRACS]
        if base is None or 'missing' in pts:
            continue
        if any(p is None for p in pts):
            broken = True
        d = [p - base for p in pts if p is not None]
        if d:
            per_c.append((max(abs(x) for x in d), min(d), d))
        if (c, k) in slope:
            pred.append(abs(slope[(c, k)]) * RANGES[k])
    if not per_c:
        continue
    m = np.array([x[0] for x in per_c])
    rows.append({"knob": k, "unit": KNOBS[k][2], "range_R": RANGES[k] * KNOBS[k][3],
                 "n_candidates": len(per_c), "median_max_abs_dloss": float(np.median(m)),
                 "min_over_cands": float(m.min()), "max_over_cands": float(m.max()),
                 "best_dloss_median": float(np.median([x[1] for x in per_c])),
                 "linear_prediction_median": float(np.median(pred)) if pred else "",
                 "no_rays_somewhere": broken})
rows.sort(key=lambda r: -r["median_max_abs_dloss"])
with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print(f"{len(cands)} candidate(s): {[f'{s}:{i}' for s, i in cands]}")
print(f"{'knob':<14} {'R':>8} {'max|dLoss|':>11} {'(min-max)':>19} {'slope*R':>9} {'best dLoss':>11}")
for r in rows:
    lp = f"{r['linear_prediction_median']:9.2f}" if r['linear_prediction_median'] != "" else "        -"
    print(f"{r['knob']:<14} {r['range_R']:6.2g}{r['unit']:<2} {r['median_max_abs_dloss']:11.2f} "
          f"({r['min_over_cands']:8.2f}-{r['max_over_cands']:8.2f}) {lp} {r['best_dloss_median']:+11.2f}"
          f"{'  * no rays somewhere' if r['no_rays_somewhere'] else ''}")

with PdfPages(OUT_PDF) as pdf:
    fig, ax = plt.subplots(figsize=(10, 0.28 * len(rows) + 2.5))
    y = np.arange(len(rows))
    med = np.array([r["median_max_abs_dloss"] for r in rows])
    colors = {'%': 'tab:green', 'mm': 'tab:orange', 'deg': 'tab:purple'}
    ax.barh(y, med, color=[colors[r["unit"]] for r in rows])
    if len(cands) > 1:
        ax.errorbar(med, y, xerr=[med - [r["min_over_cands"] for r in rows],
                                  [r["max_over_cands"] for r in rows] - med],
                    fmt="none", ecolor="black", capsize=2, lw=0.8)
    lp = [(i, r["linear_prediction_median"]) for i, r in enumerate(rows) if r["linear_prediction_median"] != ""]
    if lp:
        ax.plot([v for _, v in lp], [i for i, _ in lp], "ko", ms=4, label="local slope x R")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['knob']} (+-{r['range_R']:g} {r['unit']}){' *' if r['no_rays_somewhere'] else ''}"
                        for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("max |loss change| from the base point within +-R  (log scale)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    labels = [f"native unit {u}" for u in colors]
    if lp:
        handles.append(plt.Line2D([], [], color="k", marker="o", ls=""))
        labels.append("local slope x R")
    ax.legend(handles, labels, fontsize=8, loc="lower right")
    ax.grid(alpha=0.3, axis="x")
    ax.set_title(f"FP3 knobs over a realistic range, all units on one scale\n"
                 f"median over {len(cands)} FP2 candidate(s); * = some group loses all rays in the range",
                 fontsize=10)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    ncol = 6
    nrow = int(np.ceil(len(rows) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.4 * nrow))
    for ax, r in zip(axes.flat, rows):
        k = r["knob"]
        f = KNOBS[k][3]
        for c in cands:
            base = R.get((c, 'base', 0.0))
            xs, ys = [0.0], [0.0]
            for fr in FRACS:
                v = R.get((c, k, fr))
                if v not in (None, 'missing') and base is not None:
                    xs.append(fr * RANGES[k] * f)
                    ys.append(v - base)
            o = np.argsort(xs)
            line, = ax.plot(np.array(xs)[o], np.array(ys)[o], "o-", ms=3, lw=1)
            if (c, k) in slope:
                xx = np.array([-RANGES[k], RANGES[k]])
                ax.plot(xx * f, slope[(c, k)] * xx, "--", color=line.get_color(), lw=0.8)
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_title(k, fontsize=9)
        ax.set_xlabel(KNOBS[k][2], fontsize=7)
        ax.tick_params(labelsize=7)
    for ax in list(axes.flat)[len(rows):]:
        ax.set_visible(False)
    fig.suptitle("Loss change vs knob setting (one line per candidate; dashed: local slope)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    pdf.savefig(fig)
    plt.close(fig)
print(f"\nwrote {OUT_PDF} and {OUT_CSV}")
