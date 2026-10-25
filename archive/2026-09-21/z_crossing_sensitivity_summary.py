"""
Reads z_crossing_sensitivity_results.csv (safe to run on a partial file
while the scan is still going), fits d(z_crossing)/d(knob) per knob for
both the M12 and M16 crossings (nominal + up to 4 swept points, linear
regression), and solves for the %/knob needed to shift each crossing by
+0.28m (downstream -- matching the user's E2-implied shift). Ranks knobs
by the smaller of the two required percentages (whichever crossing is
harder to move is the bottleneck for that knob), flagging anything
needing >5% (out of the tested range) or with too little data.

Run:
    cd AgentSearch
    python3 z_crossing_sensitivity_summary.py
"""
import csv
import numpy as np

RESULTS_FILE = "Results/z_crossing_sensitivity_results.csv"
SHIFT_TARGET = 0.28
MAX_REALISTIC_PCT = 0.05

rows = list(csv.DictReader(open(RESULTS_FILE)))
nominal = [r for r in rows if r["knob_id"] == "NOMINAL"]
if not nominal or nominal[0]["m16_crossing_z"] == "" or nominal[0]["m12_crossing_z"] == "":
    raise SystemExit("NOMINAL point missing or incomplete -- run the scan first.")
nom_m16 = float(nominal[0]["m16_crossing_z"])
nom_m12 = float(nominal[0]["m12_crossing_z"])
print(f"Nominal crossings: M16(z)=0 at z={nom_m16:.4f}m   M12(z)=0 at z={nom_m12:.4f}m\n")

knob_ids = sorted(set(r["knob_id"] for r in rows if r["knob_id"] != "NOMINAL"))
results = []
for knob in knob_ids:
    pts = [r for r in rows if r["knob_id"] == knob]
    vals = [0.0] + [float(p["value"]) for p in pts]
    m16s = [nom_m16] + [float(p["m16_crossing_z"]) for p in pts if p["m16_crossing_z"] != ""]
    m12s = [nom_m12] + [float(p["m12_crossing_z"]) for p in pts if p["m12_crossing_z"] != ""]
    vals16 = [0.0] + [float(p["value"]) for p in pts if p["m16_crossing_z"] != ""]
    vals12 = [0.0] + [float(p["value"]) for p in pts if p["m12_crossing_z"] != ""]

    row = {"knob_id": knob, "n_m16": len(vals16) - 1, "n_m12": len(vals12) - 1}

    if len(vals16) >= 3:
        slope16 = np.polyfit(vals16, m16s, 1)[0]
        req16 = SHIFT_TARGET / slope16 if slope16 != 0 else np.inf
        row["m16_slope_m_per_pct"] = slope16 / 100
        row["m16_required_pct"] = req16 * 100
    else:
        row["m16_slope_m_per_pct"] = None
        row["m16_required_pct"] = None

    if len(vals12) >= 3:
        slope12 = np.polyfit(vals12, m12s, 1)[0]
        req12 = SHIFT_TARGET / slope12 if slope12 != 0 else np.inf
        row["m12_slope_m_per_pct"] = slope12 / 100
        row["m12_required_pct"] = req12 * 100
    else:
        row["m12_slope_m_per_pct"] = None
        row["m12_required_pct"] = None

    results.append(row)

def realistic(pct):
    return pct is not None and abs(pct) <= MAX_REALISTIC_PCT * 100

results.sort(key=lambda r: max(abs(r["m16_required_pct"]) if r["m16_required_pct"] is not None else np.inf,
                                 abs(r["m12_required_pct"]) if r["m12_required_pct"] is not None else np.inf))

print(f"{'knob':14s} {'M16 req %':>12s} {'M12 req %':>12s} {'both <=5%?':>12s}")
print("-" * 55)
for r in results:
    m16_str = f"{r['m16_required_pct']:+.2f}%" if r["m16_required_pct"] is not None else "n/a"
    m12_str = f"{r['m12_required_pct']:+.2f}%" if r["m12_required_pct"] is not None else "n/a"
    both_ok = realistic(r["m16_required_pct"]) and realistic(r["m12_required_pct"])
    print(f"{r['knob_id']:14s} {m16_str:>12s} {m12_str:>12s} {'YES' if both_ok else 'no':>12s}")
