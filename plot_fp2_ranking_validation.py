"""
Compare the redone FP2 ranking (fp2_ranking_validation.py, new pipeline)
with the original oat_derivative_ranking.png (2026-10-04).

Three derivatives per knob, all chain-rule on the same 2-term loss at nominal:
  NEW    central difference from the new run (grid points closest to nominal)
  OLD-CD the same central difference, but from the ARCHIVED OAT points
  OLD    the original chart's value: OLS slope across the archived 10-point
         grid over the knob's full range
NEW vs OLD-CD isolates the pipeline (same points, should be identical).
OLD-CD vs OLD isolates the derivative method (small step vs wide-range fit),
i.e. any non-linearity of the response over the original range.

Also a point-by-point check: every (knob, value) evaluated in the new run
against the archived M12/M16 at the same point.

Output: Results/fp2_ranking_validation.png (the original layout, bars = NEW,
black dots = OLD) and a printed table.

Run:
    cd AgentSearch
    python3 plot_fp2_ranking_validation.py
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fp2_ranking_validation import (ARCHIVED_OAT, RESULTS_FILE, M12_TARGET, M16_TARGET,
                                    SIGMA_M12, SIGMA_M16, load_steps)

OUT_PNG = "Results/fp2_ranking_validation.png"
steps = load_steps()

arch, new = {}, {}
for r in csv.DictReader(open(ARCHIVED_OAT)):
    arch[(r["knob_id"], round(float(r["value"]), 8))] = (float(r["M12_sim"]), float(r["M16_sim"]))
for r in csv.DictReader(open(RESULTS_FILE)):
    new[(r["knob"], round(float(r["value"]), 8))] = (float(r["M12"]), float(r["M16"]))

# point-by-point check
diffs = [(max(abs(new[p][0] - arch[p][0]), abs(new[p][1] - arch[p][1])), p) for p in new if p in arch]
worst = max(diffs)
print(f"Point-by-point vs archived OAT: {len(diffs)} points, max |dM12|,|dM16| = {worst[0]:.2e} at {worst[1]}")
for d, p in sorted(diffs, reverse=True)[:5]:
    print(f"   {p[0]:<14} {p[1]:+.6g}: new {np.round(new[p], 8)}  archived {np.round(arch[p], 8)}")

M12n, M16n = new[('NOMINAL', 0.0)]
g12 = 2 * (M12n - M12_TARGET) / SIGMA_M12 ** 2
g16 = 2 * (M16n - M16_TARGET) / SIGMA_M16 ** 2


def unit(k):
    if k.endswith('_B_SC') or k.endswith('_Exit_dB'):
        return '%', 0.01
    if k.endswith('_Roll'):
        return 'deg', 1.0
    return 'mm', 0.001


rows = []
for k, (vm, vp) in steps.items():
    u, s = unit(k)
    pm, pp = (k, round(vm, 8)), (k, round(vp, 8))
    if pm not in new or pp not in new:
        continue
    cd = lambda D: (g12 * (D[pp][0] - D[pm][0]) + g16 * (D[pp][1] - D[pm][1])) / (vp - vm) * s
    grid = sorted((v, m) for (kk, v), m in arch.items() if kk == k)
    x = np.array([v for v, _ in grid])
    d12 = np.polyfit(x, [m[0] for _, m in grid], 1)[0]
    d16 = np.polyfit(x, [m[1] for _, m in grid], 1)[0]
    rows.append({"knob": k, "unit": u, "new": cd(new), "old_cd": cd(arch), "old": (g12 * d12 + g16 * d16) * s})
if not rows:
    raise SystemExit("No complete knobs yet.")

print(f"\n{len(rows)}/65 knobs complete. Nominal: M12 {M12n:.6f}, M16 {M16n:.6f}")
for u in ('%', 'mm', 'deg'):
    sel = [r for r in rows if r["unit"] == u]
    if not sel:
        continue
    pipe = max(abs(r["new"] - r["old_cd"]) / max(abs(r["old_cd"]), 1e-12) for r in sel)
    new_order = [r["knob"] for r in sorted(sel, key=lambda r: -abs(r["new"]))]
    old_order = [r["knob"] for r in sorted(sel, key=lambda r: -abs(r["old"]))]
    rk = {k: i for i, k in enumerate(old_order)}
    a, b = np.arange(len(new_order)), np.array([rk[k] for k in new_order])
    rho = 1 - 6 * np.sum((a - b) ** 2) / (len(a) * (len(a) ** 2 - 1)) if len(a) > 2 else float('nan')
    print(f"\n=== '{u}' ({len(sel)} knobs): pipeline max rel. diff (NEW vs OLD-CD) {pipe:.1e};"
          f"  Spearman(NEW ranking, original ranking) {rho:.3f}")
    print(f"   {'knob':<14} {'NEW':>10} {'OLD-CD':>10} {'OLD (chart)':>12}   rank new/old")
    for r in sorted(sel, key=lambda r: -abs(r["new"])):
        print(f"   {r['knob']:<14} {r['new']:+10.2f} {r['old_cd']:+10.2f} {r['old']:+12.2f}   "
              f"{new_order.index(r['knob'])+1:>3}/{old_order.index(r['knob'])+1:<3}")

fig, axes = plt.subplots(3, 1, figsize=(11, 18))
for ax, u in zip(axes, ('%', 'mm', 'deg')):
    sel = sorted([r for r in rows if r["unit"] == u], key=lambda r: -abs(r["new"]))
    y = np.arange(len(sel))
    v = np.array([r["new"] for r in sel])
    ax.barh(y, v, color=np.where(v > 0, "tab:red", "tab:blue"), label="new pipeline, central difference")
    ax.plot([r["old"] for r in sel], y, "ko", ms=4, label="original chart (OLS over full range)")
    ax.set_yticks(y)
    ax.set_yticklabels([r["knob"] for r in sel], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel(f"dLoss / d({u})  (at nominal)")
    ax.set_title(f"Knobs with native unit '{u}' -- sorted by |new derivative|")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
fig.suptitle("FP2 ranking redone with the FP3 methodology (2-term M12+M16 loss, at nominal)\n"
             "bars: new pipeline, central difference;  dots: original oat_derivative_ranking.png", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(OUT_PNG, dpi=150)
print(f"\nwrote {OUT_PNG}")
