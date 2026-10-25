"""
FP2 knob ranking with all units equated (2026-10-04), same definition as
plot_fp3_equal_unit_ranking.py:
    1 unit = 1% (B_SC, Exit dB) = 1 mm (dX, dY; WF1 length treated the same)
           = 0.5 deg (Roll)
(the FP2 screen has no yaw/pitch knobs).

Context = the original FP2 ranking's: at nominal, 2-term loss
((M12+0.2472)/0.02)^2 + ((M16+0.3842)/0.03)^2, E2 FP2 groups at VD1638,
1000 rays/group. No new COSY runs:
  measured_dL  loss change at +-1 unit (larger |dL| of the two signs) from the
               archived OAT screen grid (archive/2026-09-21/fp2_oat_screen_results.csv).
               Read directly where the grid has the point (dB +-1%, dX/dY +-1 mm,
               Roll +-0.5 deg); from the parabola through nominal and the two
               innermost grid points otherwise (B_SC: 1% vs grid +-1.11%;
               WF1 length: 1 mm vs grid +-2.22 mm).
  linear_dL    |dLoss/dknob| x unit, central difference from
               fp2_ranking_validation_results.csv (new pipeline; equals the
               archived innermost points to 1e-10).

Run:
    cd AgentSearch
    python3 plot_fp2_equal_unit_ranking.py
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fp2_ranking_validation import (ARCHIVED_OAT, RESULTS_FILE as VAL_FILE, M12_TARGET, M16_TARGET,
                                    SIGMA_M12, SIGMA_M16, load_steps)

OUT_PNG = "Results/fp2_equal_unit_ranking.png"
OUT_CSV = "Results/fp2_equal_unit_ranking.csv"


def loss2(m12, m16):
    return ((m12 - M12_TARGET) / SIGMA_M12) ** 2 + ((m16 - M16_TARGET) / SIGMA_M16) ** 2


def unit_of(k):
    if k.endswith('_B_SC') or k.endswith('_Exit_dB'):
        return 0.01, '1%'
    if k.endswith('_dX') or k.endswith('_dY') or k == 'WF1_length':
        return 0.001, '1 mm'
    if k.endswith('_Roll'):
        return 0.5, '0.5 deg'
    raise KeyError(k)


arch = {}
for r in csv.DictReader(open(ARCHIVED_OAT)):
    arch.setdefault(r["knob_id"], {})[round(float(r["value"]), 8)] = loss2(float(r["M12_sim"]), float(r["M16_sim"]))
L0 = arch["NOMINAL"][0.0]

new = {}
for r in csv.DictReader(open(VAL_FILE)):
    new[(r["knob"], round(float(r["value"]), 8))] = loss2(float(r["M12"]), float(r["M16"]))
steps = load_steps()

rows = []
for k, (vm, vp) in steps.items():
    u, label = unit_of(k)
    pts = arch[k]
    if round(u, 8) in pts and round(-u, 8) in pts:
        Lp, Lm, how = pts[round(u, 8)], pts[round(-u, 8)], "direct"
    else:
        x = np.array([vm, 0.0, vp])
        p = np.polyfit(x, [pts[round(vm, 8)], L0, pts[round(vp, 8)]], 2)
        Lp, Lm, how = np.polyval(p, u), np.polyval(p, -u), "parabola(innermost)"
    lin = abs(new[(k, round(vp, 8))] - new[(k, round(vm, 8))]) / (vp - vm) * u
    rows.append({"knob": k, "unit": label, "measured_dL": max(abs(Lp - L0), abs(Lm - L0)),
                 "measured_from": how, "dL_at_plus": Lp - L0, "dL_at_minus": Lm - L0, "linear_dL": lin})
rows.sort(key=lambda r: -r["measured_dL"])
for i, r in enumerate(rows, 1):
    r["rank_measured"] = i
for i, r in enumerate(sorted(rows, key=lambda r: -r["linear_dL"]), 1):
    r["rank_linear"] = i
with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print(f"nominal 2-term loss {L0:.2f}")
print(f"{'#':>2} {'knob':<12} {'1 unit':>8} {'measured dL':>12} {'(+1u':>9} {'-1u)':>9} {'linear dL':>10}  rank meas/lin")
for r in rows:
    print(f"{r['rank_measured']:>2} {r['knob']:<12} {r['unit']:>8} {r['measured_dL']:12.2f} {r['dL_at_plus']:+9.2f} "
          f"{r['dL_at_minus']:+9.2f} {r['linear_dL']:10.2f}  {r['rank_measured']:>2}/{r['rank_linear']:<2}"
          f"{'' if r['measured_from'] == 'direct' else '  (parabola)'}")

fig, ax = plt.subplots(figsize=(10, 0.25 * len(rows) + 2.5))
y = np.arange(len(rows))
colors = {'1%': 'tab:green', '1 mm': 'tab:orange', '0.5 deg': 'tab:purple'}
ax.barh(y, [r["measured_dL"] for r in rows], color=[colors[r["unit"]] for r in rows])
ax.plot([r["linear_dL"] for r in rows], y, "ko", ms=3)
ax.set_yticks(y)
ax.set_yticklabels([f"{r['knob']}  ({r['unit']})" for r in rows], fontsize=8)
ax.invert_yaxis()
ax.set_xscale("log")
ax.set_xlabel("|loss change| for a change of 1 unit  (log)")
ax.set_title("bars: measured at +-1 unit (archived OAT grid);  dots: linear |dLoss/dk| x unit\n"
             "at nominal, 2-term loss (M12, M16) as in the original FP2 ranking", fontsize=9)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
handles.append(plt.Line2D([], [], color="k", marker="o", ls=""))
ax.legend(handles, [f"1 unit = {u}" for u in colors] + ["linear |dLoss/dk| x unit"], fontsize=8, loc="lower right")
ax.grid(alpha=0.3, axis="x")
fig.suptitle("FP2 knob ranking, units equated:\n1% = 1 mm (dX, dY, WF1 length) = 0.5 deg (roll)", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(OUT_PNG, dpi=150)
print(f"\nwrote {OUT_PNG} and {OUT_CSV}")
