"""
Diagnostic: the WF1_Exit_dB row of the new offset-Jacobian scan is
erratic at the extreme end of the +-1% range (2026-08-10) -- group_3 and
group_4's per-group means are wildly non-monotonic for dB <~ -0.0078,
then smooth again elsewhere. Suspected cause: same low-ray-survival-count
noise as the earlier Q1_dY artifact (check_q1_dy_diagnostic.py), not real
physics -- since neither the OAT screen nor this scan use a fixed random
seed or repeat-run averaging, a group with few surviving rays after the
FP1-slit+transmission cut can have a very noisy mean from run to run.

This reruns WF1_Exit_dB=-0.01 (the worst point) twice: once at the
original NRAYS_GROUPS=1000, once at 10x (10000), printing per-group
mean/std/n (verbose=True) for both. If n is small (tens, not hundreds)
at 1000 rays for group_3/group_4 and the mean stabilizes/becomes smooth
at 10000, that confirms the low-statistics diagnosis.

Run:
    cd AgentSearch
    python3 check_wf1_stability.py
"""
from fp2_metrics import compute_M12_M16

TUNE_CHANGES = [['WF1 Exit', 'dB', -0.01]]

if __name__ == "__main__":
    print("=== NRAYS_GROUPS=1000 (matches the scan that showed instability) ===")
    compute_M12_M16(tuneChanges=TUNE_CHANGES, nRays=1000, cosyFile="pg_June2025_wf1stab_1k",
                     ta_y_offset=0.0, verbose=True)

    print("\n=== NRAYS_GROUPS=10000 (10x) ===")
    compute_M12_M16(tuneChanges=TUNE_CHANGES, nRays=10000, cosyFile="pg_June2025_wf1stab_10k",
                     ta_y_offset=0.0, verbose=True)
