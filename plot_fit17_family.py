"""
fit17 accepted family at FP2 (2026-10-24): slope plot and knob plot. Analysis only; no simulation.

Input: DigitalTwin eval-queue campaign sr_fp2_fit17 (dt_eval dte-1, lattice pg_June2025_corrected, MISALIGN 1, WFs on,
E2 Hall base, 3000 rays/group, seed 42, X = -x*1000 mm at VD1638 with the FP1-slit cut): requests in
eval_queue/outbox/sr_fp2_fit17/req_*.csv, results in DigitalTwin's outbox res_*.csv. Nominal and E2 Hall rows from
DigitalTwin's export e2_fp2_observables.csv. Data: fp2_metrics measured group means.
Accepted = FP2 Mahalanobis loss (sigmas 0.027, 0.36, 0.3, 1.5 + OLS covariances) below the 4-dof 1 sigma (4.7196)
or 2 sigma (9.7156) quantile.

Run: python3 plot_fit17_family.py   Output: Results/fit17_family_slopes.png, Results/fit17_family_knobs.png
"""
import csv
import glob
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fp2_metrics as m2
from e2_hall_fp2_slopes import DT_CSV, load_csv
from jac_loop import FP2_COV, FP2_TARGETS

REQ = "eval_queue/outbox/sr_fp2_fit17"
RES = "/Users/fernandomontes/Research/COSY/Simulations/Agents/DigitalTwin/eval_queue/outbox/sr_fp2_fit17"
OBS = ["M12", "Offset12", "M16", "Offset16"]
CHI2_1S, CHI2_2S = 4.7196, 9.7156
H = np.linalg.inv(FP2_COV)


def rows(path):
    out = {}
    for f in sorted(glob.glob(path)):
        for r in csv.DictReader(l for l in open(f) if not l.startswith("#")):
            out[r["cand_id"]] = r
    return out


def main():
    spec = json.load(open(f"{REQ}/spec.json"))
    params = spec["parameters"]
    req, res = rows(f"{REQ}/req_*.csv"), rows(f"{RES}/res_*.csv")
    fam = []
    for cid, r in res.items():
        if r["status"] != "ok" or "s" in cid[2:]:  # candidates only, not stencil points
            continue
        o = np.array([float(r[f"FP2_{k}"]) for k in OBS])
        d = o - FP2_TARGETS
        fam.append(dict(cid=cid, obs=o, loss=float(d @ H @ d),
                        mx={g: float(r[f"FP2_meanx_g{g}"]) for g in m2.ALL_GROUPS},
                        knobs={p["name"]: float(req[cid][p["name"]]) for p in params}))
    one = [f for f in fam if f["loss"] < CHI2_1S]
    two = [f for f in fam if CHI2_1S <= f["loss"] < CHI2_2S]
    best = min(fam, key=lambda f: f["loss"])
    print(f"{len(fam)} candidates, {len(one)} within 1 sigma, {len(two)} more within 2 sigma; best {best['cid']} "
          f"loss {best['loss']:.2f}")

    # --- slope plot
    sim = load_csv(DT_CSV)
    mx_data = {**m2.DE_GROUPS_MEASURED_X, **m2.DALPHAX_GROUPS_MEASURED_X}
    fig, axes = plt.subplots(2, 1, figsize=(9, 12))
    for ax, groups, k, xl, name in [(axes[0], m2.DALPHAX_GROUPS, 0, "dalphaX [mrad]", "M12"),
                                    (axes[1], m2.DE_GROUPS, 2, "dE [%]", "M16")]:
        xs = np.array([groups[g] for g in groups])
        xx = np.array([xs.min(), xs.max()])
        for fset, c, lab in [(two, "#e8a0a0", f"fit17 1-2σ ({len(two)})"), (one, "#6fa8dc", f"fit17 <1σ ({len(one)})")]:
            for i, f in enumerate(fset):
                ax.plot(xx, f["obs"][k] * xx + f["obs"][k + 1], color=c, lw=0.8, alpha=0.6,
                        label=lab if i == 0 else None)
        for key, c, lab in [("nominal", "#ff7f0e", "Nominal"), ("hall", "#7f7f7f", "E2 Hall base")]:
            o, mx = sim[key]
            ax.plot(xx, o[k] * xx + o[k + 1], "--", color=c, lw=2,
                    label=f"{lab} (y={o[k]:+.3f}x{o[k + 1]:+.2f})")
            ax.plot(xs, [mx[g] for g in groups], "s", color=c, ms=6)
        ax.plot(xx, best["obs"][k] * xx + best["obs"][k + 1], "--", color="#2ca02c", lw=2,
                label=f"fit17 best {best['cid']}, loss {best['loss']:.2f} (y={best['obs'][k]:+.3f}x{best['obs'][k + 1]:+.2f})")
        ax.plot(xs, [best["mx"][g] for g in groups], "s", color="#2ca02c", ms=6)
        tk = 0 if name == "M12" else 2
        ax.plot(xx, FP2_TARGETS[tk] * xx + FP2_TARGETS[tk + 1], "--", color="black", lw=2,
                label=f"Exp. (y={FP2_TARGETS[tk]:+.3f}x{FP2_TARGETS[tk + 1]:+.2f})")
        ax.plot(xs, [mx_data[g] for g in groups], "o", color="black", ms=8)
        ax.set_xlabel(xl)
        ax.set_ylabel("X mean [mm]")
        ax.set_title(f"{name}: X mean vs {xl.split()[0]} at VD1638 (FP2)")
        ax.grid(alpha=0.4)
        ax.legend(fontsize=8)
    fig.suptitle("FP2 M12/M16: E2 data vs Nominal, E2 Hall base and fit17 accepted family (17 knobs)\n"
                 "circle = measured, square = simulated | simulations by DigitalTwin", fontsize=11)
    fig.tight_layout()
    fig.savefig("Results/fit17_family_slopes.png", dpi=130)

    # --- knob plot: knobs that moved (family mean beyond 10% of the trust half-range)
    conv = {"fraction": (100, "%"), "m": (1000, "mm"), "deg": (1, "deg")}
    moved = []
    for p in params:
        v = np.array([f["knobs"][p["name"]] for f in one])
        if abs(v.mean()) > 0.1 * p["trust"][1]:
            moved.append(p)
    panels = [("%", "exit field kick dB [%]"), ("mm", "horizontal shift X [mm]")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw=dict(width_ratios=[1, 1]))
    for ax, (u, yl) in zip(axes, panels):
        ps = [p for p in moved if conv[p["unit"]][1] == u]
        for j, p in enumerate(ps):
            s = conv[p["unit"]][0]
            for fset, c, dx, lab in [(one + two, "#ff7f0e", -0.12, f"≤2σ accepted ({len(one) + len(two)})"),
                                     (one, "#1f77b4", 0.12, f"≤1σ accepted ({len(one)})")]:
                v = np.array([f["knobs"][p["name"]] for f in fset]) * s
                ax.errorbar(j + dx, v.mean(), yerr=[[v.mean() - v.min()], [v.max() - v.mean()]], fmt="o",
                            color=c, capsize=4, label=lab if j == 0 else None)
            ax.hlines(np.array(p["trust"]) * s, j - 0.3, j + 0.3, color="gray", ls=":", lw=1,
                      label="trust range" if j == 0 else None)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks(range(len(ps)))
        ax.set_xticklabels([p["name"].rsplit("_", 1)[0] for p in ps])
        ax.set_ylabel(yl)
        ax.grid(axis="y", alpha=0.4)
        ax.legend(fontsize=8)
    still = [p["name"] for p in params if p not in moved]
    fig.suptitle("fit17 accepted family: per-knob mean ± observed range (offsets from the E2 Hall base)\n"
                 f"not shown (family mean within 10% of trust range): {', '.join(still)}", fontsize=10)
    fig.tight_layout()
    fig.savefig("Results/fit17_family_knobs.png", dpi=130)
    print("Results/fit17_family_slopes.png Results/fit17_family_knobs.png; not shown:", still)


if __name__ == "__main__":
    main()
