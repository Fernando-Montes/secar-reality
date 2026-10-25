"""
Same analysis as fp3_lean_jacobian_variability_summary.py, but over all
46 candidates' full Jacobians (fp3_full_jacobian_raw_results.csv) instead
of the original 6-candidate variability check -- per user request
2026-08-18, to see how much the Jacobian entries actually vary across
the FULL 1-sigma family, not just a small sample. Same two metrics:
%RSD (spread relative to the entry's own mean) and std*step/sigma (the
entry's base-tune-driven uncertainty, in units of the FP3 acceptance
sigma for that target -- the metric that actually matters for whether a
shared/averaged Jacobian would be safe). Reports min/max instead of the
full per-candidate list (too long to print for 46 candidates).

Run (after fp3_full_jacobian_build.py has finished):
    cd AgentSearch
    python3 fp3_full_jacobian_summary.py
"""
import csv
import numpy as np

RESULTS_FILE = "Results/fp3_full_jacobian_raw_results.csv"

FP3_KNOBS = ['Q8_B_SC', 'Q9_B_SC', 'Q10_B_SC', 'Q11_B_SC', 'B5_Exit_dB', 'B6_Exit_dB', 'WF2_Exit_dB']
FP3_KNOB_STEP = {
    'Q8_B_SC': 0.02, 'Q9_B_SC': 0.02, 'Q10_B_SC': 0.02, 'Q11_B_SC': 0.02,
    'B5_Exit_dB': 0.005, 'B6_Exit_dB': 0.005, 'WF2_Exit_dB': 0.005,
}
TARGETS = ['M12', 'M16', 'Offset12', 'Offset16']
SIGMA_TARGET = {'M12': 0.04, 'M16': 0.1, 'Offset12': 0.5, 'Offset16': 1.5}

rows = list(csv.DictReader(open(RESULTS_FILE)))
by_key = {}
for r in rows:
    key = (r["source"], int(r["iteration"]), r["fp3_knob"], int(r["sign"]))
    by_key[key] = {t: float(r[t]) for t in TARGETS}

candidates = sorted(set((r["source"], int(r["iteration"])) for r in rows))
print(f"{len(candidates)} candidates found in results.\n")

# jac[candidate][knob][target] = central-difference slope
jac = {}
for cand in candidates:
    jac[cand] = {}
    for knob in FP3_KNOBS:
        plus = by_key.get((cand[0], cand[1], knob, 1))
        minus = by_key.get((cand[0], cand[1], knob, -1))
        if plus is None or minus is None:
            continue
        step = FP3_KNOB_STEP[knob]
        jac[cand][knob] = {t: (plus[t] - minus[t]) / (2 * step) for t in TARGETS}

print(f"{'knob':12s} {'target':10s} {'n':>4s} {'mean':>12s} {'std':>12s} {'%RSD':>8s} "
      f"{'std*step/sigma':>15s} {'min':>10s} {'max':>10s}")
print("-" * 100)
results = []
for knob in FP3_KNOBS:
    for t in TARGETS:
        vals = [jac[c][knob][t] for c in candidates if knob in jac[c]]
        if len(vals) < 2:
            continue
        vals = np.array(vals)
        mean, std = vals.mean(), vals.std()
        rsd = 100 * std / abs(mean) if mean != 0 else np.inf
        impact = std * FP3_KNOB_STEP[knob] / SIGMA_TARGET[t]
        results.append((knob, t, len(vals), mean, std, rsd, impact, vals.min(), vals.max()))
        print(f"{knob:12s} {t:10s} {len(vals):4d} {mean:12.4f} {std:12.4f} {rsd:7.1f}% "
              f"{impact:15.3f} {vals.min():10.4f} {vals.max():10.4f}")

print("\nRanked by std*step/sigma (largest base-tune-driven impact first):")
for knob, t, n, mean, std, rsd, impact, vmin, vmax in sorted(results, key=lambda r: -r[6])[:10]:
    print(f"  {knob:12s} {t:10s} impact={impact:.3f}  (%RSD={rsd:.1f}%, n={n})")
