"""
Decomposes the user's E2-best tune (Q6-2.7%, WF1 Exit dB+0.25%, TA_Y+0.1mm,
loss=0.524) into its individual ingredients, now that Q6-only was found to
score WORSE (loss=6.891) than the BO v2's own best (4.575) -- meaning WF1
Exit dB and/or TA_Y, neither of which is in the BO's Q6/Q2/Q5 B_SC search
space, are doing the real work. Runs WF1-only and TA_Y-only (each with the
other two knobs left at nominal) to see which one matters.

Run:
    cd AgentSearch
    python3 fp2_user_best_decompose.py
"""
from fp2_metrics import compute_M12_M16, compute_fp2_width
from fp2_bo_search_v2 import loss_fn

CASES = {
    "WF1 Exit dB +0.25% only": {
        "tuneChanges": [['WF1 Exit', 'dB', 0.0025]], "ta_y_offset": 0.0,
        "cosyFile": "pg_June2025_userbest_wf1only_loss"},
    "TA_Y +0.1mm only": {
        "tuneChanges": [], "ta_y_offset": 0.1 / 1000,
        "cosyFile": "pg_June2025_userbest_tayonly_loss"},
}
NRAYS = 3000

if __name__ == "__main__":
    for name, cfg in CASES.items():
        M12, M16, per_group, usedFoxFile = compute_M12_M16(
            tuneChanges=cfg["tuneChanges"], nRays=NRAYS, cosyFile=cfg["cosyFile"],
            ta_y_offset=cfg["ta_y_offset"], verbose=False)
        loss = loss_fn(M12, M16)
        width = compute_fp2_width(
            tuneChanges=cfg["tuneChanges"], nRays=NRAYS, cosyFile=usedFoxFile,
            run_cosy_flag=False, verbose=False)
        print(f"{name}:")
        print(f"  M12_sim={M12:.4f}  M16_sim={M16:.4f}")
        print(f"  loss (v2, M12+M16-only) = {loss:.3f}")
        print(f"  fp2_slits_std={width['fp2_std']:.3f} mm   vd1638_std={width['vd1638_std']:.3f} mm\n")

    print("For comparison: Q6-2.7% only                       loss=6.891")
    print("                 full E2-best (Q6+WF1+TA_Y combo)   loss=0.524")
    print("                 BO v2 best (Q6/Q2/Q5)              loss=4.575")
