"""
One-at-a-time (OAT) sensitivity screen: vary each candidate knob alone
(everything else at nominal) across its own range, compute the same loss
used in fp2_joint_search.py, and rank knobs by how much they move the loss
away from the nominal baseline. Purpose: narrow ~62 candidate knobs down
to a handful for a follow-up joint optimization.

Baseline = pure nominal (tuneChanges=[]), per user's explicit preference
(NOT the Q6/WF1 joint-search "best" point) -- 2026-08-08.

Knob list, confirmed with user 2026-08-08:
  - Roll, dXY-X, dXY-Y for all 16 elements up to O1 that have them
    (including Q6 and WF1, NOT previously screened for these params)
  - B_SC for the 9 elements that have it, EXCLUDING Q6 (already fully
    characterized in fp2_joint_search.py)
  - dB for B1/B2/B3/B4 Exit (WF1 Exit dB already characterized)
  - WF1 length (new knob, 2026-08-08 -- see helper_functions.py's
    prepareCOSYchanges 'length'/'WF1' branch for the 3-line coordinated
    edit: WC line 593 + compensating DL18/DL19 lines 575/609)

Run:
    cd AgentSearch
    python3 fp2_oat_screen.py
Checkpointed/resumable exactly like fp2_joint_search.py: results appended
to fp2_oat_screen_results.csv as each point finishes; restart skips
already-completed (knob_id, value) pairs.
"""
import os
import csv
import time
import numpy as np
from multiprocessing import Pool

from config import matrix_directory, pkl_directory
from fp2_metrics import compute_M12_M16, compute_fp2_width

# ---------------------------------------------------------------- targets
# Same loss as fp2_joint_search.py. Q6 argument is always 0.0 here (no
# point in this screen touches the Q6 B_SC knob), so the Q6-prior term
# is a fixed offset (~11.1) added to every row -- irrelevant for ranking
# knobs against each other or against the nominal baseline, since it's
# identical everywhere; kept only so loss values stay numerically
# comparable to fp2_joint_search_results.csv.
M12_TARGET = -0.2472
M16_TARGET = -0.3842
Q6_E3_PRIOR = -0.005
SIGMA_M12 = 0.02
SIGMA_M16 = 0.03
SIGMA_Q6 = 0.0015


def loss_fn(M12_sim, M16_sim, Q6=0.0):
    t12 = ((M12_sim - M12_TARGET) / SIGMA_M12) ** 2
    t16 = ((M16_sim - M16_TARGET) / SIGMA_M16) ** 2
    tQ6 = ((Q6 - Q6_E3_PRIOR) / SIGMA_Q6) ** 2
    return t12 + t16 + tQ6


# ------------------------------------------------------------ knob registry
ROLL_DXY_ELEMENTS = ['Q1', 'Q2', 'B1', 'B2', 'H1', 'Q3', 'Q4', 'Q5',
                      'B3', 'B4', 'H2', 'Q6', 'Q7', 'WF1', 'H3', 'O1']
BSC_ELEMENTS = ['Q1', 'Q2', 'H1', 'Q3', 'Q4', 'Q5', 'Q7', 'H3', 'O1']  # Q6 excluded, already characterized
DB_ELEMENTS = ['B1', 'B2', 'B3', 'B4']  # WF1 Exit dB excluded, already characterized

ROLL_RANGE = 0.5      # degrees
DXY_RANGE = 0.003     # meters (3mm)
BSC_RANGE = 0.10      # fractional (10%)
DB_RANGE = 0.01       # fractional (1%)
WF1_LEN_RANGE = 0.02  # meters (2cm)
N_POINTS = 10

NRAYS_GROUPS = 1000
NRAYS_WIDTH = 1000

KNOBS = []  # list of (knob_id, tuneChanges_builder(value) -> list, values array)
for elem in ROLL_DXY_ELEMENTS:
    KNOBS.append((f"{elem}_Roll", lambda v, e=elem: [[e, 'Roll', v]],
                  np.round(np.linspace(-ROLL_RANGE, ROLL_RANGE, N_POINTS), 6)))
    KNOBS.append((f"{elem}_dX", lambda v, e=elem: [[e, 'X', v]],
                  np.round(np.linspace(-DXY_RANGE, DXY_RANGE, N_POINTS), 8)))
    KNOBS.append((f"{elem}_dY", lambda v, e=elem: [[e, 'Y', v]],
                  np.round(np.linspace(-DXY_RANGE, DXY_RANGE, N_POINTS), 8)))
for elem in BSC_ELEMENTS:
    KNOBS.append((f"{elem}_B_SC", lambda v, e=elem: [[e, 'B_SC', v]],
                  np.round(np.linspace(-BSC_RANGE, BSC_RANGE, N_POINTS), 6)))
for elem in DB_ELEMENTS:
    KNOBS.append((f"{elem}_Exit_dB", lambda v, e=elem: [[f"{e} Exit", 'dB', v]],
                  np.round(np.linspace(-DB_RANGE, DB_RANGE, N_POINTS), 6)))
KNOBS.append(("WF1_length", lambda v: [['WF1', 'length', v]],
              np.round(np.linspace(-WF1_LEN_RANGE, WF1_LEN_RANGE, N_POINTS), 6)))

RESULTS_FILE = "Results/fp2_oat_screen_results.csv"  # moved here 2026-08-08, OAT screen complete
FIELDNAMES = ["knob_id", "value", "M12_sim", "M16_sim", "loss",
              "fp2_slits_std", "vd1638_std", "fp2_slits_mean", "vd1638_mean",
              "elapsed_s"]

# -------------------------------------------------------------- checkpoint
def ensure_results_file_exists():
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()


def load_completed():
    completed = set()
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r", newline="") as f:
            for row in csv.DictReader(f):
                completed.add((row["knob_id"], round(float(row["value"]), 8)))
    return completed


def append_result(row):
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow(row)
        f.flush()


def run_one_point(task):
    knob_id, value, tuneChanges = task
    t0 = time.time()

    # dB-type knobs need NO new lattice (dB is applied at ray-propagation
    # time via dipoleWFchanges regardless of which matrices are loaded) --
    # reuse the nominal matrices directly instead of triggering a COSY run.
    # This is also a correctness fix, not just an optimization: for any
    # tuneChanges that produce an EMPTY cosyChanges internally (true for
    # dB-only changes, and for tuneChanges=[]), runECAT's run_cosy_flag=True
    # path tries to run COSY on whatever name is passed in as `foxFile`
    # rather than deriving one -- passing a made-up name there silently
    # fails and (if a stale pg_June2025.pkl happens to exist) falls back to
    # loading it anyway. Using the real base name avoids relying on that.
    is_db_knob = knob_id.endswith("_Exit_dB")

    try:
        if is_db_knob:
            M12_sim, M16_sim, per_group, usedFoxFile = compute_M12_M16(
                tuneChanges=tuneChanges, nRays=NRAYS_GROUPS, cosyFile="pg_June2025",
                ta_y_offset=0.0, verbose=False, run_cosy_flag=False)
        else:
            cosyFile = f"pg_June2025_oat_{knob_id}_{value:+.6f}".replace(".", "p")
            M12_sim, M16_sim, per_group, usedFoxFile = compute_M12_M16(
                tuneChanges=tuneChanges, nRays=NRAYS_GROUPS, cosyFile=cosyFile,
                ta_y_offset=0.0, verbose=False, run_cosy_flag=True)

        width = compute_fp2_width(
            tuneChanges=tuneChanges, nRays=NRAYS_WIDTH, cosyFile=usedFoxFile,
            run_cosy_flag=False, verbose=False)

        loss = loss_fn(M12_sim, M16_sim, Q6=0.0)
        elapsed = time.time() - t0

        append_result({
            "knob_id": knob_id, "value": value, "M12_sim": M12_sim, "M16_sim": M16_sim,
            "loss": loss, "fp2_slits_std": width["fp2_std"], "vd1638_std": width["vd1638_std"],
            "fp2_slits_mean": width["fp2_mean"], "vd1638_mean": width["vd1638_mean"],
            "elapsed_s": elapsed,
        })

        # Only clean up genuinely temporary per-point lattices -- never the
        # shared nominal matrices (usedFoxFile == "pg_June2025" for the
        # dB knobs and the nominal baseline itself; deleting those would
        # break every other in-flight or future point relying on them).
        if usedFoxFile != "pg_June2025":
            os.system(f"rm -rf {matrix_directory}/{usedFoxFile}")
            os.system(f"rm -rf {pkl_directory}/{usedFoxFile}.pkl")
        return f"[{knob_id} = {value:+.5f}] loss={loss:.1f}  ({elapsed:.0f}s)"
    except Exception as e:
        return f"[{knob_id} = {value:+.5f}] FAILED: {e}"


if __name__ == "__main__":
    ensure_results_file_exists()

    # Nominal baseline, computed first (not part of the KNOBS loop -- one
    # reference point, used to interpret every knob's sensitivity against).
    completed = load_completed()
    if ("NOMINAL", 0.0) not in completed:
        print("Computing nominal baseline...")
        t0 = time.time()
        M12_nom, M16_nom, _, usedFoxFile = compute_M12_M16(
            tuneChanges=[], nRays=NRAYS_GROUPS, cosyFile="pg_June2025",
            ta_y_offset=0.0, verbose=False, run_cosy_flag=True)
        width_nom = compute_fp2_width(
            tuneChanges=[], nRays=NRAYS_WIDTH, cosyFile=usedFoxFile,
            run_cosy_flag=False, verbose=False)
        loss_nom = loss_fn(M12_nom, M16_nom, Q6=0.0)
        append_result({
            "knob_id": "NOMINAL", "value": 0.0, "M12_sim": M12_nom, "M16_sim": M16_nom,
            "loss": loss_nom, "fp2_slits_std": width_nom["fp2_std"], "vd1638_std": width_nom["vd1638_std"],
            "fp2_slits_mean": width_nom["fp2_mean"], "vd1638_mean": width_nom["vd1638_mean"],
            "elapsed_s": time.time() - t0,
        })
        print(f"Nominal: M12={M12_nom:.4f} M16={M16_nom:.4f} loss={loss_nom:.1f}\n")
        completed = load_completed()

    # Flat task list across ALL knobs and ALL points -- no reuse
    # opportunity between points here (unlike the Q6/WF1 joint search),
    # so a flat list gives the best load balancing across workers.
    tasks = []
    for knob_id, builder, values in KNOBS:
        for v in values:
            v = float(v)
            if (knob_id, round(v, 8)) in completed:
                continue
            tasks.append((knob_id, v, builder(v)))

    n_knobs = len(KNOBS)
    n_total_points = sum(len(values) for _, _, values in KNOBS)
    print(f"{n_knobs} knobs, {n_total_points} total grid points, {len(tasks)} remaining to compute.")
    print(f"Results file: {os.path.abspath(RESULTS_FILE)}\n")

    with Pool(processes=10) as pool:
        for msg in pool.imap_unordered(run_one_point, tasks):
            print(msg, flush=True)

    print("\nAll points processed (or already complete).")
