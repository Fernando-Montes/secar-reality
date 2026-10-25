"""
FP3 Jacobian for the user's 29-knob set (2026-10-04): every knob in the
equal-unit ranking (plot_fp3_equal_unit_ranking.py) down to and including
Q8_B_SC. No new COSY runs -- built from the 46-candidate derivative run
(fp3_knob_ranking_results.csv), which already holds each knob's observables
at +step and -step at every candidate.

Per candidate: J[i, k] = d obs_i / d knob_k (central difference), obs order
[M12, Offset12, M16, Offset16], knob in its native tuneChanges unit; plus the
base observables and residual r = obs - target.

Saved:
  Results/fp3_jacobian_29knobs.npz   J (46, 4, 29) native units, J_unit (per
                                     equivalent unit), obs0 (46, 4), knobs,
                                     candidates, unit sizes, steps
  Results/fp3_jacobian_29knobs.csv   long format, one row per (candidate, obs, knob)

Printed diagnostics (whitened: rows scaled by W = H^1/2 from fp3_loss, columns
per equivalent unit, so entries are "sigma of that observable per unit"):
  - per knob: median whitened column [M12, Off12, M16, Off16] and how
    linear the response is over the step (curvature ratio, median over 46);
  - singular values of the whitened 4x29 J (median over 46): the J has at
    most 4 independent directions; tiny singular values = observable
    combinations the knob set can barely move;
  - groups of nearly parallel knobs (|cos| > 0.98, median), i.e. knobs that
    are redundant with each other in a linear fit.

Run:
    cd AgentSearch
    python3 fp3_build_jacobian.py
"""
import csv
import numpy as np

import fp3_loss
from fp3_knob_ranking import KNOBS, RESULTS_FILE as SLOPE_FILE

KNOB_SET = [
    'B5_Exit_dB', 'WF2_Exit_dB', 'Q9_X', 'B6_Yaw', 'Q11_X', 'WF2_Yaw', 'B6_Exit_dB', 'B5_Yaw',
    'Q8_X', 'B5_X', 'Q10_Yaw', 'Q8_Yaw', 'Q11_B_SC', 'Q11_Yaw', 'Q9_Yaw', 'Q10_X', 'Q9_B_SC',
    'WF2_X', 'Q8_Y', 'Q9_Roll', 'B6_X', 'Q11_Roll', 'Q9_Y', 'Q8_Roll', 'Q10_B_SC', 'B6_Pitch',
    'B5_Roll', 'B6_Y', 'Q8_B_SC',
]
OBS = ['M12', 'Offset12', 'M16', 'Offset16']
OUT_NPZ = "Results/fp3_jacobian_29knobs.npz"
OUT_CSV = "Results/fp3_jacobian_29knobs.csv"


def unit_of(k):
    if k.endswith('_B_SC') or k.endswith('_Exit_dB'):
        return 0.01
    if k.endswith('_X') or k.endswith('_Y'):
        return 0.001
    if k.endswith('_Roll'):
        return 0.5
    return 0.1                                         # Yaw, Pitch


assert len(KNOB_SET) == 29 and all(k in KNOBS for k in KNOB_SET)

S = {}
for r in csv.DictReader(open(SLOPE_FILE)):
    S[((r["source"], int(r["iteration"])), r["knob"], int(r["sign"]))] = np.array([float(r[o]) for o in OBS])
cands = sorted({c for c, k, _ in S if k == 'base'})
steps = np.array([KNOBS[k][1] for k in KNOB_SET])
units = np.array([unit_of(k) for k in KNOB_SET])

J = np.zeros((len(cands), 4, len(KNOB_SET)))
curv = np.zeros((len(cands), len(KNOB_SET)))
obs0 = np.zeros((len(cands), 4))
for ci, c in enumerate(cands):
    obs0[ci] = S[(c, 'base', 0)]
    for ki, k in enumerate(KNOB_SET):
        p, m = S[(c, k, 1)], S[(c, k, -1)]
        J[ci, :, ki] = (p - m) / (2 * steps[ki])
        curv[ci, ki] = np.max(np.abs(p + m - 2 * obs0[ci]) / np.maximum(np.abs(p - m), 1e-15))
J_unit = J * units                                    # per equivalent unit
np.savez(OUT_NPZ, J=J, J_unit=J_unit, obs0=obs0, targets=fp3_loss.TARGETS, knobs=np.array(KNOB_SET),
         candidates=np.array([f"{s}:{i}" for s, i in cands]), units=units, steps=steps, obs=np.array(OBS))
with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["source", "iteration", "observable", "knob", "dobs_dknob_native", "dobs_per_unit", "unit_native", "step_native"])
    for ci, c in enumerate(cands):
        for oi, o in enumerate(OBS):
            for ki, k in enumerate(KNOB_SET):
                w.writerow([c[0], c[1], o, k, J[ci, oi, ki], J_unit[ci, oi, ki], units[ki], steps[ki]])

ev, evec = np.linalg.eigh(fp3_loss.H)
W = evec @ np.diag(np.sqrt(ev)) @ evec.T
Jw = np.einsum('ij,cjk->cik', W, J_unit)              # whitened, per unit

print(f"Jacobian: {len(cands)} candidates x 4 observables x {len(KNOB_SET)} knobs -> {OUT_NPZ}, {OUT_CSV}\n")
print("Median whitened column per knob (sigma per unit; note W mixes M12 and Offset12, which are correlated)")
print(f"and curvature over the step (|f(+h)+f(-h)-2f(0)| / |f(+h)-f(-h)|, max over observables; median over 46):")
print(f"  {'knob':<12} {'M12':>8} {'Off12':>8} {'M16':>8} {'Off16':>8}   {'|col|':>7}  curvature")
med = np.median(Jw, axis=0)
for ki, k in enumerate(KNOB_SET):
    c = np.median(curv[:, ki])
    flag = "  <- dominated by a quadratic term: a linear J barely sees it" if c > 1 else \
           ("  <- noticeably curved over the step" if c > 0.1 else "")
    print(f"  {k:<12} " + " ".join(f"{v:+8.3f}" for v in med[:, ki]) +
          f"   {np.linalg.norm(med[:, ki]):7.3f}  {c:8.3f}{flag}")

sv = np.array([np.linalg.svd(Jw[ci], compute_uv=False) for ci in range(len(cands))])
print(f"\nSingular values of the whitened J (sigma per unit), median [min-max] over candidates:")
for i in range(4):
    print(f"  s{i+1} = {np.median(sv[:, i]):9.3f}  [{sv[:, i].min():.3f} - {sv[:, i].max():.3f}]")
print(f"  condition number s1/s4: median {np.median(sv[:, 0] / sv[:, 3]):.0f}")
_, _, Vt = np.linalg.svd(np.median(Jw, axis=0))
U, s, _ = np.linalg.svd(np.median(Jw, axis=0))
print("  weakest direction in whitened observable space (median J): " +
      ", ".join(f"{o} {v:+.2f}" for o, v in zip(OBS, U[:, 3])))

cos = []
for ci in range(len(cands)):
    A = Jw[ci] / np.linalg.norm(Jw[ci], axis=0, keepdims=True)
    cos.append(A.T @ A)
cos = np.median(np.array(cos), axis=0)
print("\nNearly parallel knob pairs (|cos| > 0.98 between whitened columns, median over 46):")
pairs = [(abs(cos[i, j]), KNOB_SET[i], KNOB_SET[j], cos[i, j])
         for i in range(len(KNOB_SET)) for j in range(i + 1, len(KNOB_SET)) if abs(cos[i, j]) > 0.98]
for a, ki, kj, v in sorted(pairs, reverse=True):
    print(f"  {ki:<12} {kj:<12} cos {v:+.4f}")
print(f"  ({len(pairs)} pairs)")
