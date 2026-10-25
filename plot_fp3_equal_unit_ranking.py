"""
FP3 knob ranking with all units equated (2026-10-04, user definition):
    1 unit = 1% (B_SC, Exit dB) = 1 mm (X, Y) = 0.5 deg (Roll) = 0.1 deg (Yaw, Pitch)
Purpose: choose the knobs for the FP3 Jacobian.

For a change of one unit, per knob:
  measured_dL  loss change actually observed at +-1 unit (larger |dL| of the two
               signs), median over the range-check candidates
               (fp3_knob_range_check_results.csv). For dB/Roll/Yaw/Pitch one unit
               equals the range check's R, so it is read directly; for B_SC and X/Y
               (unit inside +-R/2) it comes from the parabola through -R/2, 0, +R/2.
  linear_dL    |dLoss/dknob| x unit from the 46-candidate derivative run
               (fp3_knob_ranking_results.csv), median and 16-84th pct.
  obs_shift    |W J_k| x unit (W = H^1/2), median over the 46 -- CSV only. It is
               just the square root of the quadratic part of the same loss for a
               one-unit step, so it ranks the knobs almost identically and is not
               plotted (user, 2026-10-04).
Ranked by measured_dL. The PNG shows measured_dL (bars) and linear_dL (dots).

Run:
    cd AgentSearch
    python3 plot_fp3_equal_unit_ranking.py
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fp3_loss
from fp3_knob_ranking import KNOBS, RESULTS_FILE as SLOPE_FILE
from fp3_knob_range_check import RESULTS_FILE as RANGE_FILE, RANGES

OUT_PNG = "Results/fp3_equal_unit_ranking.png"
OUT_CSV = "Results/fp3_equal_unit_ranking.csv"


def unit_of(k):
    """One equivalent unit, in native tuneChanges units, plus a label."""
    if k.endswith('_B_SC') or k.endswith('_Exit_dB'):
        return 0.01, '1%'
    if k.endswith('_X') or k.endswith('_Y'):
        return 0.001, '1 mm'
    if k.endswith('_Roll'):
        return 0.5, '0.5 deg'
    if k.endswith('_Yaw') or k.endswith('_Pitch'):
        return 0.1, '0.1 deg'
    raise KeyError(k)


ev, evec = np.linalg.eigh(fp3_loss.H)
W = evec @ np.diag(np.sqrt(ev)) @ evec.T          # H^1/2

# ---- 46-candidate derivative run: linear dL and observable shift
S = {}
for r in csv.DictReader(open(SLOPE_FILE)):
    S[((r["source"], int(r["iteration"])), r["knob"], int(r["sign"]))] = (
        np.array([float(r[k]) for k in ("M12", "Offset12", "M16", "Offset16")]), float(r["loss"]))
cands = sorted({c for c, k, _ in S if k == 'base'})
lin, shift = {k: [] for k in KNOBS}, {k: [] for k in KNOBS}
for c in cands:
    o0 = S[(c, 'base', 0)][0]
    for k, (_, step, _, _) in KNOBS.items():
        if (c, k, 1) not in S or (c, k, -1) not in S:
            continue
        J = (S[(c, k, 1)][0] - S[(c, k, -1)][0]) / (2 * step)
        u = unit_of(k)[0]
        lin[k].append(abs(fp3_loss.loss_gradient(o0, J)) * u)
        shift[k].append(np.linalg.norm(W @ J) * u)

# ---- range check: measured dL at +-1 unit
Rg = {}
for r in csv.DictReader(open(RANGE_FILE)):
    Rg[((r["source"], int(r["iteration"])), r["knob"], float(r["frac"]))] = \
        float(r["loss"]) if r["status"] == "ok" else None
rcands = sorted({c for c, k, _ in Rg if k == 'base'})
meas = {k: [] for k in KNOBS}
for c in rcands:
    L0 = Rg.get((c, 'base', 0.0))
    for k in KNOBS:
        u, R = unit_of(k)[0], RANGES[k]
        if np.isclose(u, R):
            Lp, Lm = Rg.get((c, k, 1.0)), Rg.get((c, k, -1.0))
        else:
            a, b = Rg.get((c, k, -0.5)), Rg.get((c, k, 0.5))
            if None in (a, b, L0):
                continue
            x = np.array([-R / 2, 0, R / 2])
            p = np.polyfit(x, [a, L0, b], 2)
            Lp, Lm = np.polyval(p, u), np.polyval(p, -u)
        if None in (Lp, Lm, L0):
            continue
        meas[k].append(max(abs(Lp - L0), abs(Lm - L0)))

rows = []
for k in KNOBS:
    q = lambda v, p: float(np.percentile(v, p)) if v else float('nan')
    rows.append({"knob": k, "unit": unit_of(k)[1],
                 "measured_dL": q(meas[k], 50), "measured_n": len(meas[k]),
                 "measured_from": "direct" if np.isclose(unit_of(k)[0], RANGES[k]) else "parabola(+-R/2)",
                 "linear_dL": q(lin[k], 50), "linear_dL_p16": q(lin[k], 16), "linear_dL_p84": q(lin[k], 84),
                 "obs_shift_sigma": q(shift[k], 50)})
rows.sort(key=lambda r: -r["measured_dL"])
for i, r in enumerate(rows, 1):
    r["rank_measured"] = i
for i, r in enumerate(sorted(rows, key=lambda r: -r["linear_dL"]), 1):
    r["rank_linear"] = i
for i, r in enumerate(sorted(rows, key=lambda r: -r["obs_shift_sigma"]), 1):
    r["rank_obs_shift"] = i
with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print(f"derivative run: {len(cands)} candidates; range check: {len(rcands)} candidates")
print(f"{'#':>2} {'knob':<12} {'1 unit':>8} {'measured dL':>12} {'linear dL':>10} {'obs shift':>10}   ranks meas/lin/shift")
for r in rows:
    print(f"{r['rank_measured']:>2} {r['knob']:<12} {r['unit']:>8} {r['measured_dL']:12.2f} {r['linear_dL']:10.2f} "
          f"{r['obs_shift_sigma']:9.3f}σ   {r['rank_measured']:>2}/{r['rank_linear']:>2}/{r['rank_obs_shift']:>2}")

fig, ax = plt.subplots(figsize=(10, 0.27 * len(rows) + 2.5))
y = np.arange(len(rows))
colors = {'1%': 'tab:green', '1 mm': 'tab:orange', '0.5 deg': 'tab:purple', '0.1 deg': 'tab:pink'}
ax.barh(y, [r["measured_dL"] for r in rows], color=[colors[r["unit"]] for r in rows])
lv = np.array([r["linear_dL"] for r in rows])
ax.errorbar(lv, y, xerr=[lv - [r["linear_dL_p16"] for r in rows], [r["linear_dL_p84"] for r in rows] - lv],
            fmt="ko", ms=3, lw=0.8, capsize=2)
ax.set_yticks(y)
ax.set_yticklabels([f"{r['knob']}  ({r['unit']})" for r in rows], fontsize=8)
ax.invert_yaxis()
ax.set_xscale("log")
ax.set_xlabel("|loss change| for a change of 1 unit  (log)")
ax.set_title("bars: measured at +-1 unit (median, 3 candidates)\n"
             "dots: linear |dLoss/dk| x unit (median, whiskers 16-84%, 46 candidates)", fontsize=9)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
ax.legend(handles, [f"1 unit = {u}" for u in colors], fontsize=8, loc="lower right")
ax.grid(alpha=0.3, axis="x")
fig.suptitle("FP3 knob ranking, units equated:\n1% = 1 mm (X, Y) = 0.5 deg (roll) = 0.1 deg (yaw, pitch)",
             fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(OUT_PNG, dpi=150)
print(f"\nwrote {OUT_PNG} and {OUT_CSV}")
