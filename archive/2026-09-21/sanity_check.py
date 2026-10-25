"""
One-off sanity check (not the full search): compute M12/M16 at FP2/VD1638
for (1) nominal and (2) the known "best overall" E2 tune, and print them
for comparison against the spreadsheet. Run directly, not via the (future)
parallel/checkpointed search script.
"""
from fp2_metrics import compute_M12_M16

NRAYS = 3000

print("=" * 70)
print("NOMINAL (tuneChanges=[], ta_y_offset=0.0)")
print("=" * 70)
M12_nom, M16_nom, groups_nom, _ = compute_M12_M16(
    tuneChanges=[], nRays=NRAYS, cosyFile="pg_June2025", ta_y_offset=0.0)
print(f"\nNominal: M12 (dX/dalphaX) = {M12_nom:.4f} mm/mrad   [spreadsheet: ~+0.207]")
print(f"Nominal: M16 (dX/dE)      = {M16_nom:.4f} mm/%       [spreadsheet: ~+0.0776]")

print()
print("=" * 70)
print("BEST OVERALL: Q6 -2.7%, WF1 Exit dB +0.0025, TA_Y +0.1mm")
print("=" * 70)
tuneChanges = [['Q6', 'B_SC', -0.027], ['WF1 Exit', 'dB', 0.0025]]
M12_best, M16_best, groups_best, _ = compute_M12_M16(
    tuneChanges=tuneChanges, nRays=NRAYS, cosyFile="pg_June2025_best", ta_y_offset=0.1/1000)
print(f"\nBest overall: M12 (dX/dalphaX) = {M12_best:.4f} mm/mrad   [spreadsheet: ~-0.2472]")
print(f"Best overall: M16 (dX/dE)      = {M16_best:.4f} mm/%       [spreadsheet: check E2 sheet]")

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"{'case':<15}{'M12 sim':>12}{'M16 sim':>12}")
print(f"{'nominal':<15}{M12_nom:>12.4f}{M16_nom:>12.4f}")
print(f"{'best overall':<15}{M12_best:>12.4f}{M16_best:>12.4f}")
