"""
FP2 E2 observables with Q1-Q10 changed by the Hall-probe differences in the
user's "BTS35 quad Hallprobe field" plot (2026-10-07):
% = (archived 2024-06-15 mean - cycled snapshot) / cycled, signs as shown in
the plot (confirmed correct by the user; E3's sign convention is different).
Each quad's field is scaled by (1 + %): B_SC delta = nominal_B_SC * %
(additive convention; Q7's nominal is 0.981).

Only Q1-Q7 are upstream of FP2 (VD1638); Q8-Q10 are included for
completeness and cannot change FP2.

Compared against the 46 FP2 1-sigma family solutions (fp2_family_mc v1 and
v7), using the SAME 4-term FP2 loss that selected them (fp2_family_mc.py):
targets M12 -0.2472, M16 -0.3842, Offset12 1.543, Offset16 0.537; sigmas
0.02, 0.03, 0.2, 1.5; 1-sigma acceptance 4.72, 2-sigma 9.72. Beam: E2 FP2
groups (fp2_metrics), seeded, 3000 rays/group as in the family search
(1000 rays also printed, as in the FP2 ranking).

Lattice: pg_June2025_corrected (config.base); identical upstream of WF2, so
FP2 is unaffected by the WF2 correction. 1 COSY compile (nominal reuses the
cached maps).

Run yourself:
    cd AgentSearch
    /Users/fernandomontes/opt/anaconda3/envs/SECAR/bin/python3 fp2_hallprobe_check.py
"""
import csv
import numpy as np

HALLPROBE_PCT = {'Q1': +4.69, 'Q2': -0.75, 'Q3': -0.70, 'Q4': +0.32, 'Q5': -0.81,
                 'Q6': +2.96, 'Q7': -3.15, 'Q8': +1.80, 'Q9': -1.58, 'Q10': -0.07}
TARGETS = np.array([-0.2472, 1.543, -0.3842, 0.537])      # [M12, Off12, M16, Off16]
SIGMAS = np.array([0.02, 0.2, 0.03, 1.5])
ARCHIVE = "Results/archive/2026-09-21/"
FAMILY = [(ARCHIVE + "fp2_family_mc_results.csv", "v1"), (ARCHIVE + "fp2_family_mc_v7_results.csv", "v7")]
OUT_CSV = "Results/fp2_hallprobe_check.csv"


def loss4(o):
    return float(np.sum(((np.asarray(o) - TARGETS) / SIGMAS) ** 2))


def tune():
    from config import cosy_lines
    return [[q, 'B_SC', cosy_lines[q]['B_SC']['nom'] * p / 100] for q, p in HALLPROBE_PCT.items()]


def family():
    rows = []
    for path, src in FAMILY:
        for r in csv.DictReader(open(path)):
            if r["loss"] and float(r["loss"]) <= 4.7196:
                rows.append({**r, "src": src})
    assert len(rows) == 46, len(rows)
    return rows


if __name__ == "__main__":
    import fp3_eval
    out = []
    for name, tc in (("nominal (corrected .fox)", []), ("Hall-probe %", tune())):
        groups = fp3_eval.compile_maps(tc)
        for n in (3000, 1000):
            obs, _, counts = fp3_eval.observables_from_maps(groups, None, 'FP2', n)
            out.append({"case": name, "nrays": n, "M12": obs[0], "Offset12": obs[1], "M16": obs[2],
                        "Offset16": obs[3], "loss_4term": loss4(obs), "min_group_rays": min(counts.values())})
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    fam = family()
    fobs = np.array([[float(r[k]) for k in ("M12_sim", "Offset12_sim", "M16_sim", "Offset16_sim")] for r in fam])
    floss = np.array([float(r["loss"]) for r in fam])

    print("\nFP2 observables (targets: M12 -0.2472, Off12 1.543, M16 -0.3842, Off16 0.537)")
    print(f"{'case':<26} {'rays':>5} {'M12':>9} {'Off12':>8} {'M16':>9} {'Off16':>8} {'4-term loss':>12}")
    for r in out:
        print(f"{r['case']:<26} {r['nrays']:>5} {r['M12']:+9.4f} {r['Offset12']:8.3f} {r['M16']:+9.4f} "
              f"{r['Offset16']:8.3f} {r['loss_4term']:12.1f}")
    for lab, f_ in (("46 FP2 solutions: median", np.median), ("                   min", np.min), ("                   max", np.max)):
        v = f_(fobs, axis=0)
        print(f"{lab:<26} {'3000':>5} {v[0]:+9.4f} {v[1]:8.3f} {v[2]:+9.4f} {v[3]:8.3f} {f_(floss):12.2f}")

    print("\nKnob values: Hall-probe vs the 46 FP2 solutions (B_SC delta in %, dB in %)")
    knobs = ['Q1_B_SC', 'Q2_B_SC', 'Q3_B_SC', 'Q4_B_SC', 'Q5_B_SC', 'Q6_B_SC', 'Q7_B_SC',
             'WF1_Exit_dB', 'B1_Exit_dB', 'B2_Exit_dB', 'B3_Exit_dB', 'B4_Exit_dB']
    hp = {f"{q}_B_SC": v * 100 for q, _, v in tune()}
    print(f"  {'knob':<12} {'Hall-probe':>11} {'family median':>14} {'family min':>11} {'family max':>11}")
    for k in knobs:
        vals = np.array([float(r[k]) * 100 if k in r and r[k] != "" else 0.0 for r in fam])
        h = f"{hp[k]:+11.2f}" if k in hp else f"{'(not set)':>11}"
        print(f"  {k:<12} {h} {np.median(vals):+14.2f} {vals.min():+11.2f} {vals.max():+11.2f}")
    print("  (v1 solutions have no Q3/B1-B4 knobs: counted as 0)")
    print(f"\nwrote {OUT_CSV}")
