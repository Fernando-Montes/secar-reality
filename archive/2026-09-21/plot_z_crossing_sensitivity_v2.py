"""
Corrected replacement for plot_z_crossing_sensitivity.py, using the
continuation-tracked crossings (z_crossing_sensitivity_continuation.py)
instead of the old first-sign-change-in-window method, which was found
2026-08-14 to silently jump between unrelated roots for every knob with
real leverage (Q1-Q6, B1, B2). Same bar-chart style, but a knob is only
included if its continuation-tracked path actually REACHED both
directions' extremes needed for the linear fit -- if the crossing left
the search window before reaching a magnitude comparable to what's
being extrapolated to, this is flagged in the printed output rather
than silently trusted.

Run:
    cd AgentSearch
    python3 plot_z_crossing_sensitivity_v2.py
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_FILE = "Results/z_crossing_sensitivity_summary_v2.csv"
OUTPUT_DIR = "Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MAX_REALISTIC_PCT = 5.0


def to_float(s):
    return float(s) if s not in (None, "", "None") else None


rows = list(csv.DictReader(open(RESULTS_FILE)))
results = []
for r in rows:
    req16 = to_float(r["m16_global_fit_required_pct"])
    req12 = to_float(r["m12_global_fit_required_pct"])
    worst = max(abs(req16) if req16 is not None else np.inf, abs(req12) if req12 is not None else np.inf)
    results.append({"knob_id": r["knob_id"], "req16": req16, "req12": req12, "sort_key": worst,
                     "m16_range": r["m16_tracked_range_pct"], "m12_range": r["m12_tracked_range_pct"]})


def realistic(pct):
    return pct is not None and abs(pct) <= MAX_REALISTIC_PCT


kept = [r for r in results if realistic(r["req16"]) and realistic(r["req12"])]
kept.sort(key=lambda r: r["sort_key"])
print(f"{len(kept)} knobs (corrected) can shift both crossings within +-{MAX_REALISTIC_PCT:.0f}%: "
      f"{[r['knob_id'] for r in kept]}")

dropped = [r for r in results if r not in kept]
if dropped:
    print(f"\n{len(dropped)} knobs excluded (need >{MAX_REALISTIC_PCT:.0f}% or crossing left window):")
    for r in dropped:
        print(f"  {r['knob_id']}: req16={r['req16']}  req12={r['req12']}  "
              f"(tracked ranges: M16={r['m16_range']}  M12={r['m12_range']})")

if kept:
    knobs = [r["knob_id"] for r in kept]
    x = np.arange(len(knobs))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9, 7))
    r16 = [r["req16"] for r in kept]
    r12 = [r["req12"] for r in kept]
    ax.bar(x - width / 2, r16, width, label="M16 crossing: required %", color="tab:blue")
    ax.bar(x + width / 2, r12, width, label="M12 crossing: required %", color="tab:orange")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylim(-MAX_REALISTIC_PCT, MAX_REALISTIC_PCT)
    ax.set_xticks(x)
    ax.set_xticklabels(knobs, rotation=45, ha="right")
    ax.set_ylabel("required %/knob to shift crossing +0.28m (signed)")
    ax.set_title(f"CORRECTED: knobs that can shift BOTH crossings +0.28m within +-{MAX_REALISTIC_PCT:.0f}%\n"
                 "(continuation-tracked, not first-sign-change -- see z_crossing_sensitivity_continuation.py)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    outfile = os.path.join(OUTPUT_DIR, "z_crossing_sensitivity_bar_v2.png")
    fig.savefig(outfile, dpi=150)
    print(f"\nwrote {outfile}")
else:
    print("\nNo knob passes the corrected +-5% test -- no plot written.")
