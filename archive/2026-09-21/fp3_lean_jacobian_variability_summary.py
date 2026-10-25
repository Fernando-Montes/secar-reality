"""
Summarizes fp3_lean_jacobian_variability_results.csv: computes the
central-difference Jacobian entry (d[target]/d[fp3_knob]) at each
sampled FP2 base candidate, then reports the mean/std/%RSD across
candidates for every (target, knob) pair.

%RSD (std relative to the entry's OWN mean) answers "how much does this
Jacobian entry itself vary" but isn't the right yardstick for deciding
whether a shared Jacobian is safe to use -- per user request 2026-08-17,
also report std rescaled into TARGET units (std * the knob step used to
build the entry) as a fraction of that target's OWN acceptance sigma
from the FP3 loss (SIGMA_M12=0.04, SIGMA_M16=0.1, SIGMA_OFFSET12=0.5,
SIGMA_OFFSET16=1.5). That answers the actually-relevant question: if a
candidate-specific Jacobian entry differs from the "true" one by 1 std,
and the knob moves by about the step size tested here, how big is the
resulting target error as a fraction of the tolerance the loss itself
enforces? <<1 means the base-tune dependence is negligible for the
loss; ~1 or more means it isn't.

Run (after fp3_lean_jacobian_variability.py has produced data):
    cd AgentSearch
    python3 fp3_lean_jacobian_variability_summary.py
"""
import csv
import numpy as np

RESULTS_FILE = "Results/fp3_lean_jacobian_variability_results.csv"

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
print(f"{len(candidates)} candidates found in results: {candidates}\n")

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

print(f"{'knob':12s} {'target':10s} {'mean':>12s} {'std':>12s} {'%RSD':>8s} {'std*step/sigma':>15s}  per-candidate values")
print("-" * 115)
for knob in FP3_KNOBS:
    for t in TARGETS:
        vals = [jac[c][knob][t] for c in candidates if knob in jac[c]]
        if len(vals) < 2:
            continue
        vals = np.array(vals)
        mean, std = vals.mean(), vals.std()
        rsd = 100 * std / abs(mean) if mean != 0 else np.inf
        impact = std * FP3_KNOB_STEP[knob] / SIGMA_TARGET[t]
        vals_str = ", ".join(f"{v:.3f}" for v in vals)
        print(f"{knob:12s} {t:10s} {mean:12.4f} {std:12.4f} {rsd:7.1f}% {impact:15.3f}  [{vals_str}]")
