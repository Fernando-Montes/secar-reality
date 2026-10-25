"""
Definitive check (2026-08-16) for the suspected dipoleWFchanges-ordering
bug in process_single_ray (ray_functions.pyx:576-584): that function
walks the beamline with a single forward-only pointer and requires
dipoleWFchanges sorted by ascending element index, or every entry after
the first out-of-order one is silently skipped.

fp2_family_mc.py's v7 KNOBS list orders the dB knobs WF1, B1, B2, B3, B4
-- element indices [33, 7, 9, 22, 24], NOT ascending. This script takes
one real v7 1-sigma candidate's actual found knob values and computes
M12/M16 at FP2 (VD1638) TWICE: once with the dB entries in the buggy
(as-used) order, once with them correctly sorted ascending by element
index. If the bug is real, the two results will differ meaningfully
(B1-B4 would have zero effect in the buggy-order run).

Run:
    cd AgentSearch
    python3 verify_dipole_order_bug.py
"""
import csv
from fp2_metrics import compute_M12_M16

CHI2_1SIGMA_DF4 = 4.7196
V7_KNOBS_BUGGY_ORDER = ['Q1_B_SC', 'Q2_B_SC', 'Q3_B_SC', 'Q4_B_SC', 'Q5_B_SC', 'Q6_B_SC', 'Q7_B_SC',
                        'WF1_Exit_dB', 'B1_Exit_dB', 'B2_Exit_dB', 'B3_Exit_dB', 'B4_Exit_dB']
V7_KNOBS_CORRECTED_ORDER = ['Q1_B_SC', 'Q2_B_SC', 'Q3_B_SC', 'Q4_B_SC', 'Q5_B_SC', 'Q6_B_SC', 'Q7_B_SC',
                            'B1_Exit_dB', 'B2_Exit_dB', 'B3_Exit_dB', 'B4_Exit_dB', 'WF1_Exit_dB']


def knob_to_tunechange(knob_id, value):
    if knob_id.endswith('_Exit_dB'):
        elem = knob_id[: -len('_Exit_dB')]
        return [f'{elem} Exit', 'dB', float(value)]
    return [knob_id.replace('_B_SC', ''), 'B_SC', float(value)]


if __name__ == "__main__":
    rows = list(csv.DictReader(open("Results/fp2_family_mc_v7_results.csv")))
    cand = None
    for r in rows:
        if r["loss"] not in (None, "") and float(r["loss"]) <= CHI2_1SIGMA_DF4:
            cand = r
            break
    print(f"Using candidate iteration={cand['iteration']}, loss={cand['loss']}")
    knob_vals = {k: float(cand[k]) for k in V7_KNOBS_BUGGY_ORDER}
    print("Knob values:", knob_vals)

    tc_buggy = [knob_to_tunechange(k, knob_vals[k]) for k in V7_KNOBS_BUGGY_ORDER]
    tc_fixed = [knob_to_tunechange(k, knob_vals[k]) for k in V7_KNOBS_CORRECTED_ORDER]

    print("\n--- BUGGY order (WF1, B1, B2, B3, B4) ---")
    M12_bug, M16_bug, _, _ = compute_M12_M16(
        tuneChanges=tc_buggy, nRays=3000, cosyFile="pg_dipole_order_buggy", verbose=True)
    print(f"M12={M12_bug:.4f}  M16={M16_bug:.4f}")

    print("\n--- CORRECTED order (B1, B2, B3, B4, WF1 -- ascending element index) ---")
    M12_fix, M16_fix, _, _ = compute_M12_M16(
        tuneChanges=tc_fixed, nRays=3000, cosyFile="pg_dipole_order_fixed", verbose=True)
    print(f"M12={M12_fix:.4f}  M16={M16_fix:.4f}")

    print(f"\nDelta M12 = {M12_fix - M12_bug:.5f}   Delta M16 = {M16_fix - M16_bug:.5f}")
    print("If these deltas are essentially zero, the ordering doesn't matter here after all.")
    print("If they're meaningfully nonzero, the bug is confirmed real and B1-B4 were not being applied.")
