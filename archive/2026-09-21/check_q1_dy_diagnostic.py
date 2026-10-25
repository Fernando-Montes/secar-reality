"""
Diagnostic for the Q1_dY anomaly: recompute the two extreme tested points
(-3mm, +3mm) plus nominal (0) and one interior point, WITH per-group
verbose output, to see whether the retained ray count (n) per E2 group
collapses at the extremes (would indicate an aperture/FP1-slit-cut
statistics artifact) or stays healthy (would indicate a real effect).

Run:
    cd AgentSearch
    python3 check_q1_dy_diagnostic.py
"""
from fp2_metrics import compute_M12_M16

NRAYS = 1000

for label, dY in [("nominal (dY=0)", 0.0), ("interior (dY=-0.001)", -0.001),
                   ("extreme low (dY=-0.003)", -0.003), ("extreme high (dY=+0.003)", 0.003)]:
    print("=" * 70)
    print(f"Q1 dY = {dY}  [{label}]")
    print("=" * 70)
    tuneChanges = [] if dY == 0.0 else [['Q1', 'Y', dY]]
    # tuneChanges=[] produces an EMPTY cosyChanges internally (same as any
    # dB-only change), and run_cosy_flag=True's fallback path runs COSY on
    # whatever name is passed in rather than deriving one -- must be the
    # real base name ("pg_June2025") in that case, not a made-up one.
    cosyFile = "pg_June2025" if dY == 0.0 else f"pg_June2025_q1dy_check_{dY}"
    M12, M16, per_group, usedFoxFile = compute_M12_M16(
        tuneChanges=tuneChanges, nRays=NRAYS, cosyFile=cosyFile,
        ta_y_offset=0.0, verbose=True)
    print(f"--> M12={M12:.4f}  M16={M16:.4f}\n")
