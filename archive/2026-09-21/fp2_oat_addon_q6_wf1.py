"""
Addon to fp2_oat_screen.py: adds Q6 B_SC and WF1 Exit dB as two more
one-at-a-time knobs, computed at the SAME baseline (TA_Y=0, nominal
everything else) and with the SAME ranges as their peer knob types
(B_SC +-10% like the other 9 B_SC knobs; dB +-1% like the other 4 dB
knobs), so they're directly comparable to the rest of the OAT screen --
unlike fp2_joint_search.py's characterization of these two, which was
run entirely at TA_Y=+0.1mm and would conflate that offset with
whatever Q6/WF1 does if reused directly (2026-08-08, user-flagged).

Appends to the SAME fp2_oat_screen_results.csv used by fp2_oat_screen.py
and plot_oat_screen.py -- no changes needed to either of those to pick
up these two extra knobs.

Run:
    cd AgentSearch
    python3 fp2_oat_addon_q6_wf1.py
"""
import numpy as np
from multiprocessing import Pool

import fp2_oat_screen as oat  # reuse run_one_point, loss_fn, append_result, RESULTS_FILE, etc.

ADDON_KNOBS = [
    ("Q6_B_SC", lambda v: [['Q6', 'B_SC', v]],
     np.round(np.linspace(-oat.BSC_RANGE, oat.BSC_RANGE, oat.N_POINTS), 6)),
    ("WF1_Exit_dB", lambda v: [['WF1 Exit', 'dB', v]],
     np.round(np.linspace(-oat.DB_RANGE, oat.DB_RANGE, oat.N_POINTS), 6)),
]

if __name__ == "__main__":
    oat.ensure_results_file_exists()
    completed = oat.load_completed()

    tasks = []
    for knob_id, builder, values in ADDON_KNOBS:
        for v in values:
            v = float(v)
            if (knob_id, round(v, 8)) in completed:
                continue
            tasks.append((knob_id, v, builder(v)))

    print(f"{len(ADDON_KNOBS)} addon knobs, {len(tasks)} points remaining to compute.")
    print(f"Appending to: {oat.RESULTS_FILE}\n")

    with Pool(processes=10) as pool:
        for msg in pool.imap_unordered(oat.run_one_point, tasks):
            print(msg, flush=True)

    print("\nDone. Re-run plot_oat_screen.py to include these in the ranking/plots.")
