# secar-reality: matching the SECAR beam-transport model to measured data

SECAR is the recoil separator at FRIB. This repository holds the method used to find which physical
parameters ("knobs": magnet fields, exit-field kicks, misalignments) bring the COSY Infinity + ECAT
simulation of SECAR into agreement with commissioning data (experiment E2), and the parameter values
found so far.

## Data and observables

- E2: a thick target at the start of SECAR creates energy and angle spread; target apertures (pepperpot)
  and the FP1 slits select groups of rays. For each group, the mean horizontal position X is measured on
  viewers (viewer images). Group definitions and measured means: `fp2_metrics.py`, `fp3_metrics.py`.
- Observables per focal plane, from linear fits over groups: M12 = dX/d(alpha_x) [mm/mrad] with Offset12,
  and M16 = dX/d(dE/E) [mm/%] with Offset16. FP2 = viewer VD1638, FP3 = VD1783.
- Readout sign: X = -x * 1000 mm (ECAT convention), FP1-slit cut applied per group.

## Method

1. **Fix what was measured.** Magnets with E2 Hall-probe readings are set from the deviation of the
   reading from the down leg of their hysteresis loop: new SC = nominal SC x (1 + %/100)
   (`Results/e2_hall_legdown_settings.csv`).
2. **Loss.** Mahalanobis distance on (M12, Offset12, M16, Offset16) with the covariance of the data fits
   (FP2: sigmas 0.027, 0.36 mm, 0.3, 1.5 mm; FP3: `fp3_loss.py`). Acceptance: 4-dof chi-square quantiles,
   4.7196 (1 sigma) and 9.7156 (2 sigma). See `docs/kb/measurements/datasets.md`.
3. **Rank the knobs.** Central-difference derivatives of the observables for every candidate knob, in
   equal units (1% field, 1 mm shift, 0.5 deg roll, 0.1 deg yaw/pitch), give dLoss per unit and a
   curvature check (`rank_fp2_knobs.py`, `Results/fp2_rank69.*`).
4. **Fit with a Jacobian loop** (`jac_loop.py`): build the Jacobian by a stencil around the base; draw M
   candidate targets from the data covariance; solve a ridge-regularized linear correction for each (prior
   1/trust^2 on each knob, clipped to its trust range); evaluate every candidate with a full simulation;
   compare real with predicted improvement (rho) and re-linearize at the best candidate when rho leaves
   [0.7, 1.3]. Stops: enough accepted candidates, no progress, a best candidate at a trust bound (a
   "physics stop": the model may need a different parameter), or the evaluation budget.
5. **Report the family** of accepted candidates, not a single best point (`plot_fit17_family.py`,
   `export_fit17_candidates.py`).

Simulations are run by a separate service ("DigitalTwin", not in this repository) through a folder-based
evaluation queue; the protocol is `eval_queue/SPEC.md`, and each campaign's spec, requests and results are
in `eval_queue/`. Every campaign pins the sha1 of each file the evaluator imports, so results are
traceable; `jac_loop.py` lists them.

## Results so far

- `Results/fit17_fp2_candidates.csv`: 32 FP2 candidates accepted at 2 sigma (16 at 1 sigma) from the
  17-knob fit on the E2 Hall-probe base, with knob offsets, observables and per-group means. The header
  states lattice, units and conventions.
- `Results/fit17_family_slopes.png`, `Results/fit17_family_knobs.png`: the family against the data.
- `docs/kb/runs.md`: the log of every analysis, with assessments (older entries on a superseded lattice
  are marked).

## Lattice

`COSY10.2/pg_June2025_corrected.fox` (sha1 fca3af5780afba3e60dbbdea52f891cde18b45ca), with measured
misalignments (`MISALIGN 1`). Knob values are offsets from this deck (or from the Hall-probe base, as
stated in each file).

## Not included

COSY Infinity executables and manuals, the ECAT ray module (`ray_functions`), generated transfer maps,
DigitalTwin's evaluator, archived machine-data tables, and textbook-derived notes. The code here is
therefore readable and its results traceable, but the simulation chain must be supplied separately.

## Author

Fernando Montes. Analysis carried out with AI agents (Claude Code); decisions on physics and data are the
author's.
