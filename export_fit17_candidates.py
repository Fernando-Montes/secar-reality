"""
Export the fit17 accepted FP2 candidates (knob values and observables) to a self-describing CSV (2026-10-24).
Analysis only; no simulation. Same inputs and acceptance as plot_fit17_family.py.

Run: python3 export_fit17_candidates.py   Output: Results/fit17_fp2_candidates.csv
"""
import json

import numpy as np

import fp2_metrics as m2
from plot_fit17_family import CHI2_1S, CHI2_2S, H, OBS, REQ, RES, rows
from jac_loop import FP2_TARGETS

OUT = "Results/fit17_fp2_candidates.csv"
CONV = {"fraction": (100, "pct"), "m": (1000, "mm"), "deg": (1, "deg")}


def main():
    spec = json.load(open(f"{REQ}/spec.json"))
    params = spec["parameters"]
    req, res = rows(f"{REQ}/req_*.csv"), rows(f"{RES}/res_*.csv")
    out = []
    for cid, r in res.items():
        if r["status"] != "ok" or "s" in cid[2:]:
            continue
        o = np.array([float(r[f"FP2_{k}"]) for k in OBS])
        d = o - FP2_TARGETS
        loss = float(d @ H @ d)
        if loss < CHI2_2S:
            out.append((loss, cid, r, o))
    out.sort()
    cols = ([f"{p['name']}[{CONV[p['unit']][1]}]" for p in params] + [f"FP2_{k}" for k in OBS]
            + [f"FP2_meanx_g{g}[mm]" for g in m2.ALL_GROUPS])
    with open(OUT, "w") as f:
        f.write("# fit17: FP2 candidates accepted at 2 sigma (FP2 Mahalanobis loss < 9.7156, 4 dof; 1 sigma = 4.7196)\n"
                "# Lattice COSY10.2/pg_June2025_corrected.fox (sha1 fca3af5780afba3e60dbbdea52f891cde18b45ca), MISALIGN 1, "
                "WF1/WF2 on. Engine: COSY 10.2 + ECAT via DigitalTwin's eval queue (protocol sr-0.4, evaluator dt_eval dte-1), "
                "campaign sr_fp2_fit17, 2026-10-24.\n"
                "# Base: E2 Hall-probe settings (Results/e2_hall_legdown_settings.csv) on Q1-Q7, H1, H2, O1. Knob values are "
                "OFFSETS added to that base/deck: *_dB = exit field kick [% of the dipole/WF field]; *_X = horizontal shift "
                "added to the deck's SA x [mm, COSY x]; *_Yaw added to the deck's TA yaw [deg]; WF1_rhoE/rhoB/length [mm]; "
                "H3_B_SC [% of nominal SC].\n"
                "# Observables at FP2 (VD1638), readout X = -x*1000 mm (ECAT sign), FP1-slit cut per group, 3000 rays/group, "
                "seed 42: M12 [mm/mrad], Offset12 [mm], M16 [mm per % dE/E], Offset16 [mm]. "
                "Targets (E2 data): -0.2472, 1.543, -0.3842, 0.537.\n"
                "# Loss: r^T C^-1 r, sigmas 0.027, 0.36, 0.3, 1.5; cov(M12,Off12) -0.00090, cov(M16,Off16) -0.0084.\n")
        f.write("cand_id,loss,accepted," + ",".join(cols) + "\n")
        for loss, cid, r, o in out:
            vals = [float(req[cid][p["name"]]) * CONV[p["unit"]][0] for p in params]
            vals += list(o) + [float(r[f"FP2_meanx_g{g}"]) for g in m2.ALL_GROUPS]
            f.write(f"{cid},{loss:.4f},{'1sigma' if loss < CHI2_1S else '2sigma'},"
                    + ",".join(f"{v:.6g}" for v in vals) + "\n")
    print(f"{OUT}: {len(out)} candidates")


if __name__ == "__main__":
    main()
