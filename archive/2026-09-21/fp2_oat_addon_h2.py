"""
Addon to fp2_oat_screen.py: adds H2 B_SC as one more one-at-a-time knob,
for completeness in the derivative-based ranking (plot_oat_derivative_ranking.py),
per the user's request 2026-08-12. H2 was never included in the original
OAT screen's BSC_ELEMENTS list (only H2_Roll/H2_dX/H2_dY were), even
though H2 has a valid B_SC parameter in cosy_lines (confirmed directly)
-- same category as the Q6/WF1 addon (fp2_oat_addon_q6_wf1.py), same
+-10%/10-point range as the other B_SC knobs.

Appends to the SAME fp2_oat_screen_results.csv used by fp2_oat_screen.py
and plot_oat_derivative_ranking.py.

Run:
    cd AgentSearch
    python3 fp2_oat_addon_h2.py
"""
import numpy as np
from multiprocessing import Pool

import fp2_oat_screen as oat  # reuse run_one_point, RESULTS_FILE, etc.

ADDON_KNOBS = [
    ("H2_B_SC", lambda v: [['H2', 'B_SC', v]],
     np.round(np.linspace(-oat.BSC_RANGE, oat.BSC_RANGE, oat.N_POINTS), 6)),
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

    print(f"{len(tasks)} points remaining to compute for H2_B_SC.")
    print(f"Appending to: {oat.RESULTS_FILE}\n")

    with Pool(processes=10) as pool:
        for msg in pool.imap_unordered(oat.run_one_point, tasks):
            print(msg, flush=True)

    print("\nDone. Re-run plot_oat_derivative_ranking.py to include H2 in the ranking.")
