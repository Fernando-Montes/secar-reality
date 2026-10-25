"""
FP2 (VD1638) E2 slopes with the magnet fields measured during E2 (2026-10-24).

Settings: Results/e2_hall_legdown_settings.csv. Q1-Q7, H1, H2, O1 at nominal SC x
(1 + % vs leg down), from the E2 Hall probes compared with the leg-down field of
the measured hysteresis loops (secar-pv-desk tables, segment B). H3, Q1H and
everything else at the deck nominal. Fernando's decisions are in docs/kb/runs.md,
2026-10-24.
Lattice pg_June2025_corrected, MISALIGN 1, WFs on. E2 FP2 groups, 3000
rays/group, seed 42 (fp2_metrics.build_initial_distribution). Readout
X [mm] = -x*1000 at VD1638, with the FP1-slit cut per group.
Cases:
  data     measured group means (fp2_metrics *_MEASURED_X)
  nominal  deck as is: traced through the cached base map, no compile
  hall     one COSY compile (queue_eval private compile, 12-digit deck values)
Observables [M12 mm/mrad, Offset12 mm, M16 mm/%, Offset16 mm]; FP2 4-term loss
(targets -0.2472, 1.543, -0.3842, 0.537; sigmas 0.02, 0.2, 0.03, 1.5).

Run (from secar-reality/, SECAR python): python3 e2_hall_fp2_slopes.py [--plot-only]
(--plot-only replots from the CSV, no compile; --digitaltwin plots DigitalTwin's results, E2 settings in orange)
Output: Results/e2_hall_fp2_slopes.csv, Results/e2_hall_fp2_slopes.png
"""
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import fp2_metrics as m2
import fp3_eval
import queue_eval
from helper_functions import load_optics_data

SETTINGS = "Results/e2_hall_legdown_settings.csv"
OUT_CSV = "Results/e2_hall_fp2_slopes.csv"
OUT_PNG = "Results/e2_hall_fp2_slopes.png"
DT_CSV = ("/Users/fernandomontes/Research/COSY/Simulations/Agents/DigitalTwin/exports/"
          "secar-reality_e2-fp2-hall_2026-10-24/e2_fp2_observables.csv")
DT_PNG = "Results/e2_hall_fp2_slopes_digitaltwin.png"
TARGETS = np.array([-0.2472, 1.543, -0.3842, 0.537])
SIGMAS = np.array([0.02, 0.2, 0.03, 1.5])
OBS = ["M12", "Offset12", "M16", "Offset16"]


def loss(obs):
    return float(np.sum(((np.asarray(obs) - TARGETS) / SIGMAS) ** 2))


def hall_upstream():
    rows = list(csv.DictReader(l for l in open(SETTINGS) if not l.startswith("#")))
    return [{"name": f"{r['magnet']}_B_SC", "tune_change": [r["magnet"], "B_SC"], "unit": "fraction",
             "applied": "deck", "value": float(r["B_SC_offset"])} for r in rows]


def load_csv(path=OUT_CSV):
    res = {}
    for r in csv.DictReader(l for l in open(path) if not l.startswith("#")):
        res[r["case"]] = (np.array([float(r[o]) for o in OBS]),
                          {g: float(r[f"meanx_g{g}"]) for g in m2.ALL_GROUPS if r[f"meanx_g{g}"] != ""})
    return res


def main():
    if "--plot-only" in sys.argv:
        return plot(load_csv())
    if "--digitaltwin" in sys.argv:
        # simulations computed by DigitalTwin (2026-10-24); data from fp2_metrics
        sim = load_csv(DT_CSV)
        mx = {**m2.DE_GROUPS_MEASURED_X, **m2.DALPHAX_GROUPS_MEASURED_X}
        res = {"data": (fp3_eval.fit_observables(mx, "FP2"), mx), "nominal": sim["nominal"], "hall": sim["hall"]}
        return plot(res, out=DT_PNG, hall_color="#ff7f0e", source=" | simulations by DigitalTwin")
    res = {}
    # data
    mx = {**m2.DE_GROUPS_MEASURED_X, **m2.DALPHAX_GROUPS_MEASURED_X}
    res["data"] = (fp3_eval.fit_observables(mx, "FP2"), mx)
    # nominal: cached base map of the unedited deck
    maps = load_optics_data(config.fox_file, config.pkl_directory)[1]
    obs, mean_x, _ = fp3_eval.observables_from_maps(maps, None, "FP2", 3000)
    res["nominal"] = (obs, mean_x)
    # E2 Hall settings: one compile
    settings = {"evaluation": {"planes": ["FP2"], "nrays_per_group": 3000, "cosy_timeout_s": 300,
                               "deck_cut": "none"},
                "parameters": [], "upstream": hall_upstream()}
    queue_eval.init(settings, os.path.join(config.results_directory, "e2_hall_run"))
    rec = queue_eval.evaluate_group({}, [("hall", {})], settings)[0]
    if rec["status"] != "ok":
        raise SystemExit(f"compile failed: {rec['status']}")
    res["hall"] = (np.array([rec[f"FP2_{o}"] for o in OBS]),
                   {g: rec[f"FP2_meanx_g{g}"] for g in m2.ALL_GROUPS})

    with open(OUT_CSV, "w") as f:
        f.write("# FP2 E2 observables: data, nominal deck, E2 Hall settings (e2_hall_fp2_slopes.py, 2026-10-24)\n"
                "# lattice pg_June2025_corrected; settings Results/e2_hall_legdown_settings.csv; 3000 rays/group, "
                "seed 42; X = -x*1000 mm at VD1638; M12 mm/mrad, M16 mm/%\n")
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["case"] + OBS + ["loss"] + [f"meanx_g{g}" for g in m2.ALL_GROUPS])
        for case, (o, mx_) in res.items():
            w.writerow([case] + [repr(float(v)) for v in o] + [repr(loss(o))] +
                       [repr(float(mx_[g])) if g in mx_ else "" for g in m2.ALL_GROUPS])
    for case, (o, _) in res.items():
        print(f"{case:8s} M12 {o[0]:+.4f}  Off12 {o[1]:+.3f}  M16 {o[2]:+.4f}  Off16 {o[3]:+.3f}  loss {loss(o):.1f}")
    plot(res)


def plot(res, out=OUT_PNG, hall_color="#d62728", source=""):

    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
    style = {"data": dict(color="black", marker="o", ls="-"),
             "nominal": dict(color="#1f77b4", marker="s", ls="--"),
             "hall": dict(color=hall_color, marker="^", ls="-")}
    label = {"data": "E2 data", "nominal": "simulation, nominal settings",
             "hall": "simulation, E2 Hall-probe settings"}
    panels = [(axes[0], m2.DALPHAX_GROUPS, 0, "dalphaX [mrad]", "M12", "mm/mrad"),
              (axes[1], m2.DE_GROUPS, 2, "dE [%]", "M16", "mm/%")]
    for ax, groups, k, xl, name, unit in panels:
        xs = np.array([groups[g] for g in groups])
        xx = np.linspace(xs.min() - 1, xs.max() + 1, 2)
        for case, (o, mx_) in res.items():
            ys = [mx_[g] for g in groups]
            ax.plot(xs, ys, marker=style[case]["marker"], color=style[case]["color"], ls="none", ms=8)
            ax.plot(xx, o[k] * xx + o[k + 1], color=style[case]["color"], ls=style[case]["ls"], lw=2,
                    label=f"{label[case]}: {name} = {o[k]:+.3f} {unit}, offset {o[k + 1]:+.2f} mm")
        ax.set_xlabel(xl)
        ax.set_ylabel("mean X at FP2 (VD1638) [mm]")
        ax.set_title(f"FP2 {name}: X vs {xl.split()[0]}")
        ax.grid(alpha=0.4, ls="--")
        ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    fig.suptitle("E2 at FP2: data vs simulation with nominal and E2 Hall-probe (vs leg down) magnet settings "
                 f"| FP2 loss: nominal {loss(res['nominal'][0]):.0f}, Hall {loss(res['hall'][0]):.0f}{source}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(out)


if __name__ == "__main__":
    main()
