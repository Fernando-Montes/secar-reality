"""
Fixes the root-jump bug (see z_crossing_sensitivity_scan_v2.py's
docstring) by TRACKING a single physical crossing step-by-step outward
from nominal, instead of trusting whichever root a fixed-window search
happens to find first.

For each knob, M12 and M16 crossing tracked SEPARATELY, and each
direction (increasing knob / decreasing knob) tracked SEPARATELY,
starting from the nominal crossing z-location:
  - walk outward through the (fine, 0.25%-starting) magnitude grid,
  - at each step, among ALL crossings found in the window at that knob
    value, pick the one CLOSEST to the previous step's tracked
    location (path continuation),
  - if NO crossings exist at some step, tracking stops there for that
    direction (the crossing has left the search window) -- reported,
    not silently dropped or replaced.

This directly replaces z_crossing_sensitivity_summary.py's flawed
"first sign-change, whatever it is" logic. Also reports the LOCAL slope
(from the single nearest step to nominal, ~0.25%) as an independent
cross-check against directly-measured M12/M16 derivatives (e.g.
q3_raw_fan_check.py's dM12/dQ3, dM16/dQ3 at nominal).

Run (after z_crossing_sensitivity_scan_v2.py has produced data):
    cd AgentSearch
    python3 z_crossing_sensitivity_continuation.py
"""
import csv
import numpy as np

RESULTS_FILE = "Results/z_crossing_sensitivity_results_v2.csv"
SHIFT_TARGET = 0.28
MAX_REALISTIC_PCT = 0.05

OUTPUT_SUMMARY = "Results/z_crossing_sensitivity_summary_v2.csv"
OUTPUT_TRACKED = "Results/z_crossing_sensitivity_tracked_v2.csv"


def parse_crossings(s):
    if not s:
        return []
    return [float(x) for x in s.split(";")]


MAX_JUMP_M = 2.0  # meters -- largest step-to-step move seen in genuine continuous
# tracking is ~0.6m (B1_Exit_dB M16, 1% step); the root-jump artifact this whole
# fix targets produces ~10m jumps. 2.0m is a safe, well-separated cutoff between
# the two: found 2026-08-15 after B1_Exit_dB's M12 tracking silently accepted a
# ~9.8m jump to an unrelated root (the ONLY crossing left in the window once the
# real one had exited it) because "nearest to previous" has no sense of "too far
# to be the same root" without an explicit cap -- this corrupted the fitted
# required-% enough to flip its sign, caught only by comparing against the
# beam-envelope plot's actual traced rays.


def track_direction(nominal_z, steps):
    """steps: list of (value, [crossings]) sorted by increasing |value|,
    all same sign. Returns list of (value, tracked_z) -- stops at the
    first step with no crossings, OR where the nearest available
    crossing is implausibly far from the last tracked point (the
    physical crossing has left the window; some other, unrelated root
    happening to still exist elsewhere doesn't count)."""
    tracked = []
    ref = nominal_z
    for value, crossings in steps:
        if not crossings:
            break
        chosen = min(crossings, key=lambda z: abs(z - ref))
        if abs(chosen - ref) > MAX_JUMP_M:
            break
        tracked.append((value, chosen))
        ref = chosen
    return tracked


if __name__ == "__main__":
    rows = list(csv.DictReader(open(RESULTS_FILE)))
    nominal = [r for r in rows if r["knob_id"] == "NOMINAL"][0]
    nom_m16 = parse_crossings(nominal["m16_crossings_z"])[0]
    nom_m12 = parse_crossings(nominal["m12_crossings_z"])[0]
    print(f"Nominal: M16(z)=0 at z={nom_m16:.4f}m   M12(z)=0 at z={nom_m12:.4f}m\n")

    knob_ids = sorted(set(r["knob_id"] for r in rows if r["knob_id"] != "NOMINAL"))
    tracked_rows = []
    summary_rows = []

    for knob in knob_ids:
        pts = [(float(r["value"]), parse_crossings(r["m16_crossings_z"]), parse_crossings(r["m12_crossings_z"]))
               for r in rows if r["knob_id"] == knob]
        pos = sorted([p for p in pts if p[0] > 0], key=lambda p: p[0])
        neg = sorted([p for p in pts if p[0] < 0], key=lambda p: -p[0])  # closest-to-zero first

        m16_pos = track_direction(nom_m16, [(v, c16) for v, c16, c12 in pos])
        m16_neg = track_direction(nom_m16, [(v, c16) for v, c16, c12 in neg])
        m12_pos = track_direction(nom_m12, [(v, c12) for v, c16, c12 in pos])
        m12_neg = track_direction(nom_m12, [(v, c12) for v, c16, c12 in neg])

        for value, z in m16_neg + [(0.0, nom_m16)] + m16_pos:
            tracked_rows.append({"knob_id": knob, "quantity": "M16", "value": value, "tracked_z": z})
        for value, z in m12_neg + [(0.0, nom_m12)] + m12_pos:
            tracked_rows.append({"knob_id": knob, "quantity": "M12", "value": value, "tracked_z": z})

        def local_slope(direction_pts, nom_z):
            """Slope from nominal to the single nearest tracked step."""
            if not direction_pts:
                return None
            v, z = direction_pts[0]
            return (z - nom_z) / v

        def global_fit_required_pct(neg_pts, pos_pts, nom_z):
            all_pts = list(reversed(neg_pts)) + [(0.0, nom_z)] + pos_pts
            if len(all_pts) < 3:
                return None, None
            vals = np.array([v for v, _ in all_pts])
            zs = np.array([z for _, z in all_pts])
            slope = np.polyfit(vals, zs, 1)[0]
            req = (SHIFT_TARGET / slope) * 100 if slope != 0 else np.inf
            return slope / 100, req

        m16_slope_local_pos = local_slope(m16_pos, nom_m16)
        m16_slope_local_neg = local_slope(m16_neg, nom_m16)
        m12_slope_local_pos = local_slope(m12_pos, nom_m12)
        m12_slope_local_neg = local_slope(m12_neg, nom_m12)

        m16_fit_slope, m16_req = global_fit_required_pct(m16_neg, m16_pos, nom_m16)
        m12_fit_slope, m12_req = global_fit_required_pct(m12_neg, m12_pos, nom_m12)

        max_pos_reach = m16_pos[-1][0] * 100 if m16_pos else 0.0
        max_neg_reach = m16_neg[-1][0] * 100 if m16_neg else 0.0

        summary_rows.append({
            "knob_id": knob,
            "m16_local_slope_pos": m16_slope_local_pos, "m16_local_slope_neg": m16_slope_local_neg,
            "m12_local_slope_pos": m12_slope_local_pos, "m12_local_slope_neg": m12_slope_local_neg,
            "m16_tracked_range_pct": f"[{-max_neg_reach:.2f}, {max_pos_reach:.2f}]",
            "m12_tracked_range_pct": f"[{-(m12_neg[-1][0]*100 if m12_neg else 0):.2f}, "
                                      f"{(m12_pos[-1][0]*100 if m12_pos else 0):.2f}]",
            "m16_global_fit_required_pct": m16_req,
            "m12_global_fit_required_pct": m12_req,
        })

        print(f"{knob}:")
        print(f"  M16 local slope: neg-side={m16_slope_local_neg}  pos-side={m16_slope_local_pos}  "
              f"m/pct tracked range={summary_rows[-1]['m16_tracked_range_pct']}")
        print(f"  M12 local slope: neg-side={m12_slope_local_neg}  pos-side={m12_slope_local_pos}  "
              f"tracked range={summary_rows[-1]['m12_tracked_range_pct']}")
        m16_req_str = f"{m16_req:+.2f}%" if m16_req is not None else "n/a (crossing left window immediately)"
        m12_req_str = f"{m12_req:+.2f}%" if m12_req is not None else "n/a (crossing left window immediately)"
        print(f"  Global-fit required % for +0.28m: M16={m16_req_str}  M12={m12_req_str}\n")

    with open(OUTPUT_TRACKED, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["knob_id", "quantity", "value", "tracked_z"])
        writer.writeheader()
        writer.writerows(tracked_rows)
    print(f"wrote {OUTPUT_TRACKED}")

    with open(OUTPUT_SUMMARY, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"wrote {OUTPUT_SUMMARY}")
