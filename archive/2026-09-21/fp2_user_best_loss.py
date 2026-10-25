"""
Computes M12/M16 (and the E3-style FP2/VD1638 widths) for the user's own
hand-found E2-best tune -- Q6 -2.7%, WF1 Exit dB +0.25%, TA_Y +0.1mm --
under the exact same group set and loss_fn (2-term, M12+M16 only, no Q6
prior) used by the v2 BO search. This tune uses different knobs (Q6,
WF1 Exit dB, TA_Y offset) than the BO's 3-knob family (Q6/Q2/Q5 B_SC), but
loss_fn only depends on the resulting M12_sim/M16_sim, so it's directly
comparable to every point in fp2_bo_search_v2_results.csv regardless of
which knobs produced it.

Run:
    cd AgentSearch
    python3 fp2_user_best_loss.py
"""
from fp2_metrics import compute_M12_M16, compute_fp2_width
from fp2_bo_search_v2 import loss_fn

TUNE_CHANGES = [['Q6', 'B_SC', -0.027], ['WF1 Exit', 'dB', 0.0025]]
TA_Y_OFFSET = 0.1 / 1000
NRAYS = 3000
COSY_FILE = "pg_June2025_userbest_loss"

if __name__ == "__main__":
    M12, M16, per_group, usedFoxFile = compute_M12_M16(
        tuneChanges=TUNE_CHANGES, nRays=NRAYS, cosyFile=COSY_FILE,
        ta_y_offset=TA_Y_OFFSET, verbose=True)

    loss = loss_fn(M12, M16)

    width = compute_fp2_width(
        tuneChanges=TUNE_CHANGES, nRays=NRAYS, cosyFile=usedFoxFile,
        run_cosy_flag=False, verbose=True)

    print(f"\nUser's E2 best (Q6-2.7%, WF1 Exit dB+0.25%, TA_Y+0.1mm):")
    print(f"  M12_sim={M12:.4f}  M16_sim={M16:.4f}")
    print(f"  loss (v2, M12+M16-only) = {loss:.3f}")
    print(f"  fp2_slits_std={width['fp2_std']:.3f} mm   vd1638_std={width['vd1638_std']:.3f} mm")
    print(f"\nFor comparison, the current BO v2 best is loss=4.575 "
          f"at Q6=+0.46% Q2=-7.70% Q5=+9.68%.")
