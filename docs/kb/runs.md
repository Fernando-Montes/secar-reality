# Optimization run log

Append-only. One entry per attempt, successful or not. Format:

```markdown
## YYYY-MM-DD — <short description>
Dataset: measurements/<file>
Free parameters: ...
Fixed: everything else
Baseline: <metric> = <value>
Result: <metric> = <value> at <param=value, ...>
Assessment: ...
```

## 2026-08-07 — Agent pipeline validation (AgentSearch), FP2 M12/M16
Dataset: E2 groups {3,4,5,7,8,9,10} (M16/dX-dE) and {10,13,14,15,16} (M12/dX-dalphaX), per user confirmation.
Purpose: validate a new, isolated simulation pipeline (`AgentSearch/`, copies of config.py/helper_functions.py/ViewerAnalysis.py, redirected to isolated Matrix/pkl/Results/COSY10.2 dirs) reproduces the user's own established E2 results before any new fitting is attempted.
Method: `AgentSearch/fp2_metrics.py` reproduces the exact per-group aperture-type initial distribution (target/pepperpot alignment from `groupsInfo`, generic angles/dE, TA_Y realized as `target_alignment.Y + 0.1mm`) and the FP1-slits-then-VD1638 ray filtering pattern from `ViewerAnalysis.py:484-499` (a real bug — omitting the FP1-slits filter — was caught by the user from the first run, since it made distinct groups sharing the same pepperpot alignment produce identical simulated output; fixed by filtering on `-x*1000` at the "FP1 Slits" element index against each group's `slits_mm` window, same sign convention as the original code).
Result (user compared against original E2 spreadsheet values, new comparison recorded in `Results/OpticsOptimization.xlsx` sheet `AgentResults`):
- M12 (dX/dalphaX), nominal: original sim +0.2070, agent pipeline +0.2075 (Δ≈0.0005) — match.
- M12, best-overall (Q6 −2.7%, WF1 Exit dB +0.0025, TA_Y +0.1mm): original sim −0.2324, agent −0.2330 (Δ≈0.0006) — match.
- M16 (dX/dE), best-overall: original sim −0.3851, agent −0.3887 (Δ≈0.004) — match.
- M16, nominal: original sim **+0.0776**, agent **+0.1981** — does not match (factor >2 off). User's assessment: not a low-statistics artifact (~300 surviving rays for the affected groups); user suspects an error in the *original* E2 hand-calculation rather than the new pipeline. Not investigated further — user's call, since the case that matters (best-overall) agrees well and nominal-M16 isn't a target of the ongoing fit.
Assessment: pipeline validated as reliable for M12 in both cases and M16 in the best-overall case (the regime actually relevant to the joint E2/E3 fit). One unresolved, deliberately-not-chased discrepancy (nominal M16) noted for the record in case it matters later.

## Target established 2026-08-07 (going forward — new round of joint fitting)

Per the optimization protocol's "establish the target" step, before any
new `tuneChanges` are attempted:

- **Datasets**: both E2 and E3, used jointly (see below), not just one.
- **Coupling between E2 and E3**: **[VERIFIED, user]** hardware-level
  knobs (magnet field strengths, misalignments) must take the **same**
  value to explain both E2 and E3 — they're the same physical magnets.
  Only initial-beam-distribution parameters may differ between the two
  (E2 and E3 use different beams: 20Ne(p,γ) vs ¹⁴N⁶⁺). The current
  E2≠E3 disagreement (see the E3 section above) is treated as a real
  problem to solve, not an artifact of using different beams.
- **Metric — search vs. acceptance, agreed 2026-08-07**:
  - *Search signal* (cheap, drives the fit): E2's M12/M16 slopes, and
    E3's derived magnet values (Q6 +0.5%, and the FP3 Q9/Q11 pair).
  - *Acceptance test* (required before calling a candidate converged):
    the resulting simulated histograms/projections must also
    reasonably match the measured ones (visual and/or chi-squared, in
    the style already used in the presentations' per-group plots) —
    not just the derived scalar slopes. Rationale: a prior slope-only
    fit (E2) converged but then failed to predict E3, and the still-
    unexplained shape anomalies (unusual VD1515/VD1542 shape, spread
    FP3 spot) are exactly the kind of thing a slope-only metric is
    blind to. If scalar-matching plateaus without passing the
    histogram acceptance test, the fallback is reviving full
    histogram/chi-squared fitting (the earlier Bayesian/MCMC approach,
    `ViewerAnalysis.py`'s `optimizeMCMC`) with a **wider knob range**
    than before — its earlier failure was attributed to range, not
    method (see the Bayesian/MCMC entry below).
- **Free parameters, this round**: start with field-strength knobs
  (`B_SC`, Wien filter E/B) **and** misalignments (`Roll`, `dXY`) —
  both already known to matter (Q6 field, TA_Y misalignment at FP2;
  Q9's dXY misalignment for transmission). Extend beyond `cosy_lines`
  (e.g. higher-order/fringe-field terms) only if this scope proves
  insufficient — user explicitly deferred that decision rather than
  ruling it in or out now.
- **Convergence criterion**: not yet numerically defined — still
  informal ("close," histogram agreement "looks right"). Revisit once
  a concrete fit is underway.
Source: user, 2026-08-07 conversation
Recorded: 2026-08-07

## Methodology (established prior to this KB, reconstructed 2026-08-07)

**[VERIFIED, source: user]** Before this knowledge base existed, the
user performed a manual sequential optimization, recorded in
`Results/OpticsOptimization.xlsx` sheet `E2` (other sheets `Optics`,
`E3` in that workbook are not part of this — user specified only `E2`
is useful). Read into this KB 2026-08-07.

**Figure of merit**: match the slopes **dX/dE** and **dX/dalphaX**
between simulation and measurement at each viewer — not absolute mean
position. dX = shift in measured/simulated mean spot position at a
viewer; dE and dalphaX are derived per-`Group` from experimental
conditions (aperture position, collimator size, dE/E, FP1 slit
setting) recorded in the sheet's leftmost columns. In the sheet layout,
measured (data) values occupy roughly columns A-L, simulation (ECAT)
results start around column L onward, arranged as one column-block per
tune variant tried (e.g. "Nominal", "Nominal Q6 - 2.7%", ...).

**Dataset**: `PartD_ViewerAnalysis/Group{1..19}/`, values manually
read from the per-viewer PNG figures and curated (several data points
flagged/excluded as "looks wrong" during review — see
`measurements/datasets.md`).

**[VERIFIED]** Corroborating source added 2026-08-07:
`sources/OpticsComparison-MatrixElements.pdf`, a 24-slide presentation
of this same work ("Characterization and optimization of optical
matrix elements"), which the user confirmed matches this
reconstruction. It adds detail not present in the spreadsheet alone:

- **Experiment**: data taken as part of **E22503D** ("full acceptance
  measurement"). 103 viewer images recorded; ~20% unusable (too dim,
  "leaky" light, incorrect settings). "Next steps" slide references a
  planned **20Ne(p,γ) run** — i.e. a proton beam incident on a 20Ne
  target — consistent with the `20Ne` filename fragments seen in
  `E3_optimization/` (see `measurements/datasets.md`).
  Source: sources/OpticsComparison-MatrixElements.pdf, slide 1, slide 24
- **Method**: a foil at the target induces angle and energy spread;
  apertures at the target select angle sub-ranges; FP1 slits select
  energy sub-ranges; viewer images are compared against ECAT
  simulation output. Angle-restricting-aperture data shown has "Angle
  ≤ 12.5 mrad" as one example cut.
  Source: sources/OpticsComparison-MatrixElements.pdf, slide 1
- **Explicit fixed assumption**: "Quads and dipoles up to FP1 work as
  intended" — i.e. everything upstream of FP1 is trusted and untuned;
  this is the stated justification for starting the sequential
  optimization at VD1515/VD1542 rather than further upstream.
  Source: sources/OpticsComparison-MatrixElements.pdf, slide 1
- **dE/E calibration**: VD1515 used to obtain dE/E after the foil by
  matching the viewer distribution to the optics distribution assuming
  a Gaussian energy spread. Result: **dE/E = 2.0 ± 7.7%**.
  Source: sources/OpticsComparison-MatrixElements.pdf, slide 2
- **Matrix element naming**: the sheet's `dX/dE` and `dX/dalphaX`
  columns are named using standard TRANSPORT-notation matrix elements
  `M_ij` (1=x, 2=a, 3=y, 4=b, 5=l, 6=δ — the same numbering COSY's own
  `SB` command references, per `cosy/language-reference.md`): **M12 =
  dX/dalphaX**, **M16 = dX/dE**. Original slide-1 goal was to obtain
  M11, M33 (crossed out — not pursued), M12, M16 at FP2 and FP3, then
  extend to FP4.
  Source: sources/OpticsComparison-MatrixElements.pdf, slide 1, slide 3

**Order/strategy**: sequential, one or two knobs at a time, one focal
plane at a time, each stage building on the previous stage's result:
1. Optimize incoming beam properties using VD1515 and VD1542, holding
   all downstream knobs at their assumed-correct values.
2. Move to VD1638 (FP2) and adjust quads (and other knobs) a few at a
   time until the dX/dE, dX/dalphaX slopes are close to measured.
3. Move to VD1783 (FP3), repeat, on top of the FP2 solution.
4. Move to VD1836/VD1879 (FP4), repeat, on top of the FP2+FP3 solution.

This is a knob-scope departure from the KB's default optimization
protocol (which says "optimize only parameters the user has explicitly
designated as free, everything else fixed") — here the *scope itself*
sequentially expands (each stage's knobs become fixed for the next
stage), which is a valid and explicitly user-directed strategy, not a
protocol violation, but it does mean a later-stage "fixed" knob was
itself a free parameter one stage earlier. Worth keeping in mind when
assessing whether an early-stage knob choice is still optimal after
later stages changed things downstream of it.

## (predates E2, exact date unknown) — Bayesian/MCMC full-histogram matching — ATTEMPTED, UNDERPERFORMED, NOT ABANDONED
Dataset: `PartD_ViewerAnalysis/Group{N}/` CSV projections (`_xf_cosy.csv`/`_yf_cosy.csv`) — the same measurements as the E2 work below, but read programmatically rather than digitized by hand from the PNGs (see `measurements/datasets.md`).
Free parameters: not fully enumerated here — driven by whatever `params` list is passed to `optimizeMCMC` (see e.g. the commented-out call at `main_new2.py:212-221`, which lists `initialDist` params plus `Q1`/`Q2` `Y`/`B_SC`, `B2 Exit dB`, etc.)
Method: **[VERIFIED, user]** Before the E2 slope-matching approach, the user attempted to directly match the **full measured histograms/projections** (not just a derived slope) using a Bayesian approach — this is what `ViewerAnalysis.py`'s `optimizeMCMC` function actually is (built on `emcee`, imported at `ViewerAnalysis.py:5-6`).
Result: **[VERIFIED, user]** "the range of the knobs was not large enough and the results were not that great." I.e. the fit under-performed — consistent with the independently-noted FP3 slide caveat "MCMC approach did not extend further than ±3% for all quads" (see the FP3 entry below), which now reads as a symptom of this same range limitation rather than a one-off note.
Assessment: **[VERIFIED, user]** Not abandoned — the user explicitly flagged this as "a possible venue to continue the optimization," i.e. a candidate direction for future work (e.g. widening the knob search range) rather than a dead end. The E2 slope-matching approach was adopted afterward, likely *because* of this shortfall, though that causal ordering is my inference, not stated explicitly.
Source: user, 2026-08-07 conversation; ViewerAnalysis.py:1-15, 394
Recorded: 2026-08-07

## Sign convention: simulation correction vs. real-hardware correction (clarified 2026-08-07, user)
**[VERIFIED, user]** A `tuneChanges` correction found by fitting the
*simulation* to match *measured* beam behavior, and a correction
applied to the *real magnet* to bring it up to what *nominal* (as-
designed) optics predicts, are **inverse operations, not the same
number with the same sign** — this is not an error, it's two different
reference frames:
- **Simulation correction** answers "what do I change in ECAT so the
  simulated beam matches the real observed beam?" E.g. E2 found Q6
  `B_SC` **−2.7%** in the simulation matches reality. This means the
  *real* Q6 magnet is physically running about 2.7% **weaker** than
  its nominal/mapped field strength (that's the only way decreasing
  the simulated field reproduces what the weaker real magnet does).
- **Hardware correction** answers "what do I change on the real magnet
  to bring the real beam back to what nominal optics predicts?" Since
  the real magnet is running weak, you'd **increase** it: **+2.7%**.
  This is exactly what slide 16 of
  `sources/OpticsComparison-update_20Ne.pdf` (E3 pilot run) reports as
  the "From E2" starting point for the online Q6 correction.
- Same logic, confirmed consistent for all three FP2/FP3 knobs:
  - Q6: sim −2.7% ⇒ real magnet weak ⇒ hardware correction **+2.7%**.
  - Q9: sim +9.49% ⇒ real magnet strong ⇒ hardware correction **−9.5%**.
  - Q11: sim −0.67% ⇒ real magnet weak ⇒ hardware correction **+0.7%**.
  All three match `sources/OpticsComparison-update_20Ne.pdf` slide 16
  exactly under this reading.
**Rule of thumb for future entries in this log**: always state
explicitly which direction a correction is (simulation-to-match-
reality, or hardware-to-match-nominal) — do not assume a sign carries
over between the two without applying this inversion.
Source: user, 2026-08-07 conversation
Recorded: 2026-08-07

## 2026-08-07 (retroactive) — FP2 (VD1638) tuning
Dataset: PartD_ViewerAnalysis groups (subset used in E2 sheet rows ~18-64)
Free parameters: Q6 `B_SC`, Q7 `B_SC`, WF1 E/B field strength, `TA_Y` misalignment, `PP_Y` (pepperpot Y alignment)
Fixed: incoming beam distribution (from VD1515/VD1542 stage), everything downstream of FP2
Baseline: not numerically recorded in this KB yet — sheet has nominal-tune slope values (e.g. row 61, row 81) not yet transcribed here
Result: **"Best overall: TA_Y = surv + 0.1mm, Q6 - 2.7%, WF1 + 0.25%"**
Assessment (user's own, from sheet notes): angle_Y and PP_Y misalignment don't have a large effect; TA_Y solves the mean-Y offset issue; Q7 doesn't significantly change Y_sigma. X-mean is not perfect for some groups after this tune, but histogram comparisons (`Results/OpticsComparison-MatrixElements.pdf` per sheet note referencing "OpticsComparison-MatrixElements.key") look like good agreement regardless.

**[VERIFIED]** Quantitative detail from `sources/OpticsComparison-MatrixElements.pdf`:
- **Measured targets**: M12(FP2) = -0.25 mm/mrad, M16(FP2) = -0.45 mm/%.
- **Baseline was qualitatively wrong, not just quantitatively off**: nominal (untuned) ECAT gave a dX/dalphaX slope of **+0.207 mm/mrad — the opposite sign** from the measured -0.25 mm/mrad. This is a sign flip, not a small offset — worth remembering as a strong prior next time nominal-vs-measured disagreement is being triaged elsewhere in the beamline.
- Q7 ±1% tried first: negligible effect on either slope.
- **Q6 -2.7%** flips the simulated slope to the correct sign and brings it close to the measured M12 value.
- Q6 -2.7% still leaves an offset (not a slope error) on the dX/dE (M16) line; hypothesized cause: WF1 field-strength mismatch.
- **Q6 -2.7% + WF1(E/B) +0.25%**: matches M12 slope well; M16 offset reduced further, remaining mismatch judged "within uncertainty of tuning" (i.e. not chased further).
- **Independent cross-check**: the Q6 correction was separately reproduced via an MCMC fit approach (the commented-out `optimizeMCMC` machinery in `ViewerAnalysis.py`) — same correction found two different ways. This is a genuine corroboration, not just a single-method fit artifact.
Source: sources/OpticsComparison-MatrixElements.pdf, slides 3-6
Recorded: 2026-08-07

## 2026-08-07 (retroactive) — FP3 (VD1783/VD1688) tuning
Dataset: PartD_ViewerAnalysis groups (subset used in E2 sheet rows ~68-118)
Free parameters: Q8, Q9, Q10, Q11 `B_SC`, WF2 E/B field strength, Q9 `X` (transverse offset)
Fixed: FP2 solution above (Q6, WF1, TA_Y) plus everything upstream
Baseline: not numerically recorded in this KB yet
Result: **"Nominal + FP2 opt. + Q11-0.67% + Q9+9.49% + WF2-0.077%"**, with a further "+ Q8+7%" variant also tried and marked "Best?"
Assessment: not yet captured in detail — sheet has per-group slope numbers not yet transcribed.

**[VERIFIED]** Quantitative detail from `sources/OpticsComparison-MatrixElements.pdf`:
- **Measured targets**: M12(FP3) = -0.24 mm/mrad, M16(FP3) = -2.1 mm/%.
- Q8, Q10 tried first: minimal effect on either slope.
- **Q9 +9.49% + Q11 -0.67%** (on top of the FP2 solution) brings the M12 (dX/dalphaX) slope close to measured.
- Residual offset remains on the M16 (dX/dE) line; hypothesized cause: WF2 mismatch. Adding **WF2(E/B) -0.077%** narrows it further; "WF2 mismatch within uncertainty of tuning" (same judgment call as WF1 at FP2).
- Cross-check: VD1688 vs VD1783 (two viewers near FP3) show consistent agreement between measurement and the adapted simulation — a same-focal-plane, different-viewer sanity check.
- Caveat stated on-slide: "MCMC approach did not extend further than ±3% for all quads" — i.e. the automated MCMC cross-check (that corroborated Q6 at FP2) was bounded to ±3% and so could **not** reach or independently corroborate the +9.49% Q9 correction. The Q9 correction here is **only** a manual/grid-search result, unlike Q6 at FP2.
Source: sources/OpticsComparison-MatrixElements.pdf, slides 9-11
Recorded: 2026-08-07

**[VERIFIED] — flagged per optimization protocol ("report fit quality honestly")**:
The +9.49% Q9 correction was independently investigated via a **Hall
probe measurement of Q9's actual field vs. current**, specifically
*because* a ~10% field-strength correction is large enough to warrant
checking against real hardware. Result, verbatim from the slide: "Q9
field vs current was investigated using a hall probe and **within
measurement uncertainty a factor of 10% was not observed** between
mapped field and observed value." I.e. **the Hall probe did NOT
confirm a 10% field discrepancy in Q9** — the correction that makes
the simulation match measured slopes is *not* explained by an actual
hardware field error. This is precisely the situation the optimization
protocol calls out: a knob driven to fit the data without independent
physical corroboration is a symptom of a model error elsewhere (an
unmodeled effect being absorbed by Q9), not a validated physical
finding. **Do not treat the +9.49% Q9 `B_SC` correction as a confirmed
hardware fact** — treat it as an unresolved discrepancy. See also the
separate Q9 X-misalignment finding below, which is a different
Q9 parameter and does NOT explain this either (misalignment affects
intercept, not slope).
Source: sources/OpticsComparison-MatrixElements.pdf, slide 11
Recorded: 2026-08-07

**[VERIFIED] — Q9 X-misalignment (separate from the Q9 field-strength
issue above)**: "Q9 X-misalignment has the largest effect on beam
behavior" of any single knob investigated in this whole study.
Changing the "current (2022) measurement misalignment" from **0.834
mm to no misalignment lowers transmission by 10%.** This 0.834 mm
value is not a free/tuned parameter — cross-referenced against
`config.py`: `cosy_lines['Q9']['dXY']['nom'] = [0.000834, -0.000234]`
(config.py:133) already encodes exactly this 0.834 mm X-misalignment
as the *nominal* (pre-existing, not tuned) value in the SECAR `.fox`
file. This upgrades that number from "as encoded in the simulation,
unverified against a source" (per `secar/layout.md`'s general caveat)
to **independently confirmed by a 2022 physical alignment
measurement**, for this specific value only.
However — important scope limit: **"Q9 X-misalignment has no effect
on slope. It has effect on intercept: moves the beam left or right."**
So this large, real, measured misalignment explains transmission and
beam-position offset, but does **not** explain the M12/M16 slope
mismatches that motivated the Q9 `B_SC` +9.49% correction above. These
are two independent Q9-related findings; don't conflate them.
Source: sources/OpticsComparison-MatrixElements.pdf, slide 18; config.py:131-135
Recorded: 2026-08-07

**[VERIFIED] — robustness check via misalignment Monte Carlo**: to
assess whether the FP2/FP3 corrections found above are within the
range plausibly explained by ordinary magnet misalignment uncertainty
(rather than being an arbitrarily-sized fit artifact), the user ran
three Monte Carlo studies (1000 iterations each, 2019 vs 2022 survey
alignment data) randomly perturbing quads 1-11's X/Y position and roll
angle and recomputing the M12/M16 slopes at VD1638 and VD1783:
- Survey data itself: average difference between 2019 and 2022
  alignment surveys was -0.33 mm (X) and -0.11 mm (Y), with per-quad
  scatter (σ) of 0.32 mm (X) and 0.34 mm (Y); roll angle showed
  negligible average difference, σ = 0.15 mrad.
- "Optimistic" MC: N(0, 0.34mm) for X/Y (zero-mean, survey-scatter
  only), N(0, 0.15mrad) for roll.
- "Pessimistic" MC: N(±0.33mm, 0.31mm) for X/Y (includes the observed
  systematic 2019→2022 shift), N(0, 0.15mrad) for roll.
- "Worse" MC: N(0, 0.45mm) for X/Y (inflated σ), N(0, 0.15mrad) for roll.
- Resulting slope distributions across all three scenarios were narrow
  and mutually consistent: VD1638 dX/dE mean ≈ -0.392 to -0.397
  (σ≈0.015-0.021), VD1638 dX/dalphaX mean ≈ -0.245 to -0.251
  (σ≈0.019-0.029), VD1783 dX/dE mean ≈ -2.09 to -2.12 (σ≈0.085-0.12),
  VD1783 dX/dalphaX mean ≈ -0.254 to -0.256 (σ≈0.019-0.022; "Worse"
  scenario's histogram reported `nan` mean — likely a fit failure for
  that one case, not a real value).
**Correction/resolution, 2026-08-07 (user confirmed directly)**: the
MC was seeded from **nominal** ECAT, not the Q6/WF1-adapted tune. This
upgrades the observation above from a flagged inference to a genuinely
notable, user-confirmed result: **randomly perturbing quads 1-11's
alignment by realistic, survey-measured amounts (zero-mean or
including the observed 2019→2022 systematic shift) around *nominal*
ECAT reproduces a VD1638 dX/dalphaX mean of ≈-0.245 to -0.251 —
matching the measured M12(FP2) = -0.25 mm/mrad — even though nominal
ECAT itself (zero perturbation) gives +0.207 mm/mrad, the wrong
sign.** In other words: realistic magnet-misalignment-scale
perturbations, applied broadly across many quads rather than as a
single deliberately-tuned Q6 correction, are on their own enough to
flip the sign and land near the measured slope. This is a meaningful
piece of corroborating evidence that the FP2 mismatch is physically
plausible (consistent with real-world misalignment magnitudes) rather
than requiring an implausibly large or ad-hoc correction — a stronger
result than the Q6-specific MCMC cross-check alone. Note this doesn't
mean Q6 specifically is "the" answer — the MC perturbed 11 quads
jointly, so it demonstrates plausibility of *some* combination of
misalignments producing the effect, not that Q6 alone is uniquely
responsible.
Source: user, 2026-08-07 conversation; sources/OpticsComparison-MatrixElements.pdf, slides 19-21
Recorded: 2026-08-07

## 2026-08-07 (retroactive) — FP4 (VD1836/VD1879) tuning — IN PROGRESS, not converged
Dataset: PartD_ViewerAnalysis groups (subset used in E2 sheet rows ~139-167)
Free parameters tried: Q12, Q13, Q14 `B_SC`
Fixed: FP2+FP3 solution above, plus everything upstream
Result: incomplete. User's sheet notes: "Q12 does not have a significant effect", "Q13 does not have a significant effect" — Q14 not yet resolved. Trailing sheet notes read "Finish Q14" and "Check transmission", indicating this stage was left unfinished.
Assessment: this stage is the active/current front of the optimization work as of 2026-08-07 — confirmed unfinished by the user directly.

**[VERIFIED]** Additional detail from `sources/OpticsComparison-MatrixElements.pdf`:
- **Past B8 (VD1836)**: "Observations show change between restricted
  angle and large aperture. Difference not observed in simulations" —
  a qualitative measurement behavior (position/shape changing with
  aperture size) that the simulation does not reproduce at all,
  independent of any slope-matching question. One VD1836 dX/dalphaX
  data point is explicitly flagged on-slide as "Exp. but questionable"
  — i.e. the user does not trust that particular measured slope, not
  just the simulation.
- **FP4 (VD1879)**: both dX/dE and dX/dalphaX slopes show a sizable,
  unresolved gap between the FP2+FP3-adapted simulation and
  measurement — no knob tried at FP4 (Q12, Q13, or Q14+2%) closes it.
- **Full-beamline ray-tracing comparison** (nominal vs. FP2+FP3-adapted
  tune, colored by energy): the adapted tune fixes the crossing/focus
  behavior at FP2 and FP3 as intended, but shows **"large differences
  past FP3"** in the traced rays — however, **"same transmission
  reaching DSSD for nominal and adapted"**, i.e. despite the
  significantly different ray paths downstream of FP3, the final
  transmitted fraction to the DSSD detector is unchanged between the
  nominal and adapted tunes. Worth remembering: transmission agreement
  at the final detector is not evidence that the downstream optics
  (FP4 and beyond) are correctly matched — the ray paths differ
  substantially even though the aggregate transmission doesn't.
  A further "Adapted up to FP3 and Q14 changed by 2%" ray-trace was
  tried and visibly shifts the FP4-region focus, but this was not
  carried to a quantitative slope match in the sheet.
Source: sources/OpticsComparison-MatrixElements.pdf, slides 13, 15, 17
Recorded: 2026-08-07

## Stated next steps (as of the 2026-08-07 presentation — not independently re-confirmed with the user beyond acknowledging FP4 is unfinished)
**[VERIFIED]** From the presentation's final "Next steps" slide:
1. Include, in the **20Ne(p,γ) run plan**, use of the "reverse adapted"
   optics with modified Q6, Q9, Q11 — i.e. the FP2/FP3 corrections
   found here are intended to be propagated into an actual upcoming
   beamtime tune, not left as simulation-only findings.
2. Find/confirm the Q14 and Q15 correction needed to fit the beam into
   the DSSD using nominal optics; fall back to a **phenomenological
   correction** ("use of phenomenological correction if nothing else")
   if a physically-motivated one can't be found — i.e. the user has
   already anticipated that FP4 might not yield a clean single-cause
   explanation.
3. Continue investigating the source of the large discrepancies
   between measurement and simulation past FP3, specifically:
   high-order (nonlinear) magnet effects, and the effect of Q14/Q15
   specifically.
Source: sources/OpticsComparison-MatrixElements.pdf, slide 24
Recorded: 2026-08-07

**Note on completeness of this log**: these three entries are a
reconstruction from the spreadsheet's "best overall" annotations, not
a full transcription of every knob value/slope number tried at each
stage. The sheet contains many more intermediate attempts (e.g. Q6+1%,
Q6-1% at FP2; Q10+1%, Q11+1%, Q11+7%, Q9+1%, Q9+5%, Q9+7% at FP3) whose
individual results have not been transcribed here. If precise numeric
values are needed later (e.g. to reproduce a `tuneChanges` list), go
back to `Results/OpticsOptimization.xlsx` sheet `E2` directly rather
than relying on this summary.

---

## 2026-08-07 (retroactive) — E3: ¹⁴N⁶⁺ pilot beamtime, real-hardware validation of the E2 corrections — PARTIALLY DISAGREES WITH E2

**[VERIFIED, user]** Context: E3 is **real online beamtime data**
using a **¹⁴N⁶⁺ beam** (a resonance-measurement pilot run — different
purpose and different species from E2's 20Ne(p,γ) work), taken using
corrections learned from E2 as a starting point. Recorded from
`Results/OpticsOptimization.xlsx` sheet `E3` +
`sources/OpticsComparison-update_20Ne.pdf` (17-slide update; slides
1-14 are identical to `OpticsComparison-MatrixElements.pdf`, slides
15-17 are new).
Source: user, 2026-08-07 conversation; sources/OpticsComparison-update_20Ne.pdf slides 15-17
Recorded: 2026-08-07

**Beam differs from E2, confirmed at FP1**: unusual angle/shape at
VD1515 and VD1542 (elongated/tilted spot, visible on slide 15) —
attributed to the ¹⁴N⁶⁺ beam genuinely having different incoming beam
properties than the foil-induced synthetic distribution used in E2,
not treated as a simulation failure.
Source: sources/OpticsComparison-update_20Ne.pdf, slide 15

**Routine steering (expected, not anomalous)**: B3, B4, B5, B6 (dipoles)
and WF1, WF2 (Wien filters) were tweaked by small amounts, within
expected experimental uncertainty, to center the real beam on the
FP2/FP3 viewer axis — this is standard beam tuning, not a finding.
Source: user, 2026-08-07 conversation

**Genuine anomaly**: the beam spot at FP3 (VD1783) was unexpectedly
spread out in X, well beyond anticipated width — visible in the bottom
figure of slide 15. Unlike the routine steering above, this was not
expected and remains only partially explained (see Q9/Q11 findings
below). Beam behavior at FP1/FP3/FP4 "agrees qualitatively with ECAT
(beam left/right) but not quantitatively" (verbatim, slide 15).
Source: sources/OpticsComparison-update_20Ne.pdf, slide 15

**Method — spot-width minimization at the true focal plane, not the
viewer**: nominal (as-designed) optics minimizes beam spot width
exactly at the true FP2/FP3 locations (slide 14's ray-trace). Since
only the viewer position can be directly measured, and VD1638 sits
0.28 m downstream of the true FP2 (see `secar/layout.md`), the E3
sheet (columns A-F: `Q6 [%]`, mean/std at "FP2 slits", mean/std at
"VD1638") scans Q6 in **simulation** to establish the mapping between
"spot minimized at VD1638" and "spot minimized at true FP2," then uses
that mapping to back out what real-hardware Q6 correction actually
minimizes the spot at the true FP2 — a fundamentally different method
from E2's slope-matching (M12/M16) approach. Same logic applies to
FP3/VD1783 (offset present but not broken out in `elements_dict` —
confirmed small by the user).
Source: user, 2026-08-07 conversation; Results/OpticsOptimization.xlsx sheet E3, columns A-F

**Result for FP2 — disagrees with E2**: this method gives a real
hardware Q6 correction of **+0.5%**, vs. E2's implied hardware
correction of **+2.7%** (see the sign-convention note above) — a real,
substantial disagreement between the two independent methods, not
rounding noise.

The 0.5% itself has a documented internal wrinkle (an open "to do"
item, not yet resolved):
- **1.4%** = a rough estimate used *live* online during the pilot run
  for the VD1638→true-FP2 translation.
- **1.22%** = the *corrected*, careful version of that same
  translation, computed after the fact in the E3 sheet (the "Diff.
  0.01224" note).
- **−0.9%** = an additional real-hardware Q6 tweak applied online to
  actually minimize the spot on the viewer itself.
- **0.5%** reported on the slides = 1.4% + (−0.9%), built from the
  *rough* 1.4% estimate. Using the corrected 1.22% instead would give
  1.22% + (−0.9%) = **0.32%**, not 0.5%. This inconsistency is
  unresolved — flagged directly by the user as a to-do item (see
  below), not something to silently paper over.

**Robustness checks** (E3 sheet, past column F): the optimum-Q6
finding was checked for sensitivity to other assumptions —
"max change 0.2% in optimum Q6 field for a ~1% change in rhoE
[Wien filter electric-bend radius parameter]", "no change in optimum
Q6 field for a ~±1% change in rhoB [Wien filter magnetic-bend radius]",
"no change in optimum Q6 field for a ~±2mm change in dX". I.e. the
0.5% (or 1.22%) finding is robust to those particular uncertainties —
the disagreement with E2's 2.7% is not explained by rhoE/rhoB/dX
assumptions.
Source: Results/OpticsOptimization.xlsx sheet E3, rows 60/89 (bottom summary notes)

**Result for FP3 — also disagrees with E2**: after applying the Q6
+0.5% correction, E2's predicted Q9/Q11 values do **not** minimize the
real VD1783 X-width. Two alternative combinations do:
- **Q9 = −21.8%, Q11 = −6.9%**, or
- **Q9 = −9.5%** (this matches E2's hardware-direction value exactly)
  **with Q11 = −11.4%** instead of E2's implied +0.7% — this second
  combination is marked **"Favorite settings"** on slide 16.
Source: sources/OpticsComparison-update_20Ne.pdf, slide 16

**Stated next steps, from slide 17 ("To do") — supersedes/extends the
"Stated next steps" section above for FP2/FP3 specifically**:
1. Re-investigate the E2 data using Q6 +0.5% instead of +2.7% — try
   combinations involving Q7, possibly Q3-Q5, to see if a smaller Q6
   correction plus other quads can still explain the E2 slope results.
2. Check the 0.5% online correction discrepancy (the 1.4%-vs-1.22%
   issue above) — "why did it change online? check it makes sense."
3. Needs to fit slopes dX/dAx, dX/dE — i.e. redo the E2-style
   slope-matching analysis using the E3-derived knob values, to check
   whether they also satisfy the slope-matching criterion.
4. What to do with FP3 data — two open sub-items, both referencing
   "entry 689" (an external logbook/run entry, not further identified
   in this KB): find solutions that match the weird beam shape at
   VD1515/VD1542, and reproduce the Q9/Q11 behavior.
Source: sources/OpticsComparison-update_20Ne.pdf, slide 17
Recorded: 2026-08-07

**Overall assessment (per the optimization protocol's "report fit
quality honestly" principle)**: E2 (simulation-side slope-matching
against a foil-induced synthetic beam) and E3 (real-hardware
spot-width minimization against an actual ¹⁴N⁶⁺ beam) are two
independent methods that should, if the model were complete and
correct, converge on the same physical magnet corrections. They do
not — Q6 disagrees by a factor of ~5 (0.5% vs 2.7%), and Q9/Q11 only
partially agree (Q9 can match at −9.5% but Q11 needs −11.4% instead of
+0.7%). This is a genuine, currently-unresolved model/measurement
discrepancy, not yet attributable to a single cause. Do not treat
either E2's or E3's quadrupole values as "the" answer without further
reconciliation work (item 1-3 above).

---
# Entries added 2026-10-23 (backfill of 2026-08-08 .. 2026-10-19)

Work between 2026-08-08 and 2026-10-19 was done but not logged here at the
time. Entries below are reconstructed from the scripts' own docstrings and
results files (paths given) and from the agent's session notes. Where a
detail could not be re-checked, it says so.

## 2026-08-08 .. 2026-08-14 — FP2 family search (fp2_family_mc v1..v7) -> 46 FP2 solutions
Dataset: E2 FP2 groups at VD1638 (measurements/datasets.md, "E2 beam and loss definitions").
Free parameters: v1: Q1, Q2, Q4, Q5, Q6, Q7 B_SC, WF1 Exit dB (7). v7: adds Q3 B_SC and B1-B4 Exit dB (12).
Fixed: everything else (lattice pg_June2025; FP2 is upstream of WF2, so unaffected by the 2026-10-04 correction).
Method: sample "true" target residuals from the E2 fit uncertainties, solve a regularized (ridge, per-knob trust range) linear correction from a Jacobian, verify each candidate with real COSY on the 4-term FP2 loss. Earlier BO/OAT/joint searches led to this design (see the docstrings of the scripts in AgentSearch/archive/2026-09-21/, e.g. fp2_family_mc.py, fp2_oat_screen.py, fp2_bo_search*.py).
Result: 46 candidates with 4-term loss <= 4.72 (1 sigma): 24 from v1 (Results/archive/2026-09-21/fp2_family_mc_results.csv), 22 from v7 (fp2_family_mc_v7_results.csv). Q6 B_SC in all 46 is -0.12 .. -4.03% (median -2.81%).
Assessment: a family, not a unique solution; degenerate directions among the knobs are expected. Source: AgentSearch/archive/2026-09-21/fp2_family_mc.py docstring and results files.

## 2026-08-14 .. 2026-08-18 — FP3 family on the OLD lattice (superseded)
Dataset: E2 FP3 groups at VD1783.
Free parameters: Q8-Q11 B_SC, B5/B6/WF2 Exit dB (7), per FP2 candidate.
Method: candidate-specific 4x7 Jacobians (fp3_full_jacobian_build.py), FP3 Mahalanobis loss (introduced here, 2026-08-18, covariance /4), ridge-regularized solve, COSY verification (fp3_family_mc.py).
Assessment: **superseded** — run on pg_June2025 with the spurious WF2 misalignment (fixed 2026-10-04). Old-lattice FP3 values for the 46 candidates are in Results/archive/2026-09-21/fp3_family_1sigma_results.csv; do not reuse them.

## 2026-09-21 .. 2026-09-22 — Q6/Q14/Q15/all-quad studies and a 15-quad mistune Monte Carlo (OLD lattice)
Purpose: sensitivity studies, not fits to E2.
- Q14/Q15 for transmission to DSSD (nominal elsewhere): plateau at 96.83% (183/189 default rays) along a ridge, e.g. Q14 = Q15 = -20%; the 6 lost rays are lost upstream (Q7 Exit, FP3 Slits). Results/q14_q15_*.csv.
- Field-error scans (+-1%, 0.1% steps) of each of Q1-Q15 around Q14 = Q15 = -20%: transmission, FP2/FP3 resolution |ME(1,7)|/WV, DSSD mean/range (real and a hypothetical 50x50 cm DSSD). Ranking by peak-to-peak: Q3 and Q6 dominate all metrics. Results/quad_field_error_scan_all*, quad_ranking_*.
- Monte Carlo, all 15 quads Uniform[-1%,+1%] simultaneously: 11504 trials, histograms in Results/mc_quad_mistune_histograms.png (transmission 84.4 +- 10.9%).
Assessment: FP3/DSSD numbers are on the old lattice (wrong WF2 misalignment) and the FP3 resolution uses ME(1,7) from compute_me17_cumulative.py, which is biased (simulation/agentsearch-pipeline.md). Treat as outdated; FP2 parts remain valid. Operational lesson: the MC filled the disk (cleanup deleted the wrong name) -> fixed, see simulation/agentsearch-pipeline.md.

## 2026-10-04 — Lattice correction (WF2) and pipeline changes
User found the spurious WF2 TA +-0.1 deg misalignment; AgentSearch now uses pg_June2025_corrected (cosy/fox-file-notes.md). Ported the Emulator "bug 3" fixes, added fast_tracer (validated), misalign_blocks (yaw/pitch, WF2), fp3_eval (single-pass, validated). Details: simulation/agentsearch-pipeline.md.

## 2026-10-04 — FP3 knob ranking by local derivative, 46 FP2 candidates (corrected lattice)
Dataset: E2 FP3 groups at VD1783, 3000 rays/group.
Free parameters (user's list): Q8-Q11 B_SC; B5, B6, WF2 Exit dB; X, Y, Roll, Yaw, Pitch of Q8-Q11, B5, B6, WF2 (42 knobs). Upstream knobs fixed at each FP2 solution.
Method: central differences (1% B_SC, 0.25% dB, 0.25 mm, 0.1 deg roll, 0.05 deg yaw/pitch); dLoss/dk by the chain rule on [M12, Off12, M16, Off16]. fp3_knob_ranking.py; results Results/fp3_knob_ranking_results.csv.
Baseline: FP3 loss of the 46 FP2 solutions on the corrected lattice 109-173 (2-sigma threshold 9.7); all miss the same way: M12 -0.70..-0.80 (target -0.24), Offset12 5.7-7.5 mm (4.34), Offset16 5.2-6.8 mm (3.06), M16 -1.32..-1.56 (-2.10). Candidate v1:20: loss 72 (old lattice) -> 172 (corrected): offsets moved ~+3.7 mm, slopes almost unchanged.
Result: ranking within each unit is the same in all 46 candidates (pairwise Spearman >= 0.96); derivative magnitudes vary ~8% between candidates. Top: B5/WF2/B6 Exit dB; Q9, Q11, Q8, B5 X; B6, WF2, B5 yaw.
Validation of the method: the FP2 ranking was redone through the same pipeline (fp2_ranking_validation.py): all 131 points equal the archived FP2 OAT screen to 9e-15 (also confirms FP2 unchanged by the correction); differences from the original chart come only from local slope vs full-range OLS (rolls, Q1_dX, B2_dX).
Range check (fp3_knob_range_check.py, 3 candidates, +-10% B_SC, +-1% dB, +-3 mm, +-0.5 deg roll, +-0.1 deg yaw/pitch): consistent; Y/pitch/roll respond quadratically (small locally).
Equal-unit ranking (user's units: 1% = 1 mm = 0.5 deg roll = 0.1 deg yaw/pitch): Results/fp3_equal_unit_ranking.*.

## 2026-10-04/05 — FP3 Jacobian, 29 knobs (user's set: equal-unit ranking down to Q8_B_SC)
Built from the ranking data, no new runs: Results/fp3_jacobian_29knobs.npz (46 x 4 x 29).
Result: about four independent directions with whitened singular values 51 (common offset shift), 4.0 (M16), 0.71 (M12), 0.07 (Offset12 - Offset16); ~20 position knobs (dB, X, yaw) are nearly parallel. Linear minimum-norm correction to the median residual: |dk| ~ 11 units, dominated by Q9_B_SC ~ +9.3%.
Check with real compiles (range-check data): M12 is linear in Q9 to +-10% (+0.056 per 1%); M12 reaches the target at Q9 ~ +7..+11% depending on the candidate, but this pushes both offsets up ~3.3 mm; Q11_B_SC moves M12 equally fast but also M16 by ~0.7 per 1%.
Assessment: the FP3 M12 miss requires a large Q9 change with these knobs. Do not compare its sign directly with E3 values (different sign convention; user 2026-10-07). Per-observable decomposition of dLoss: Results/fp3_dloss_by_term.png.
Open: whether ~10% on Q9 is physically plausible.

## 2026-10-07 — Archived Hall-probe settings applied at FP2
Dataset: E2 FP2 groups; Hall-probe % = (archived 2024-06-15 mean - cycled snapshot)/cycled for Q1-Q10 (user's plot), applied as B_SC changes (signs as plotted, confirmed by the user).
Result (3000 rays/group, 4-term FP2 loss): nominal M12 +0.208, M16 +0.198, loss 1153; Hall-probe set M12 +0.505, M16 +1.239, loss 4623. Both slopes keep the wrong sign. Hall-probe Q6 +2.96% vs FP2 family -0.1..-4.0%; Q1, Q5, Q7 also outside the family's ranges.
Assessment: archived field differences alone do not reproduce the E2 FP2 slopes. User says "there is more to that story" about Q6 and other magnets (to be discussed; instrument details in local notes). Results/fp2_hallprobe_check.csv.

## 2026-10-17 .. 2026-10-19 — WF1 field-model / EFL scenarios (with the efl agent)
Purpose: decide whether WF1's effective field length needs its own knob.
Method: rays at the Q7 Exit sent to efl, which tracked them through WF1 to the H3 Entrance in 6 scenarios (its "Python way", int E dz = v int B dz, no misalignments): 1 COSY E/B Enge (EFL_B 2431.7), 2 Danfysik E/B (2367), 3/4 B EFL 2337/2382 with COSY tails, 5/6 the same with Danfysik tails. ECAT apertures and maps used before and after. Full E2 beam (4978 rays that pass FP1 slits, 12 groups): analyse_wf1_efl_scenarios.py; reference reproduces the standard evaluation to 9e-16.
Result (vs ECAT nominal): FP2 M16 shifts -0.09 .. -0.38 (sigma 0.03); B EFL 2337 -> 2382 alone shifts FP2 M16 by ~-0.14 and FP3 offsets by ~+0.5 mm. FP2 M12 barely changes (+0.20..+0.25, data -0.25). Scenario 1 brings FP2 M16 from +0.20 to -0.18 with no quad change. Plots: Results/wf1_efl_scenarios_slopes_FP2.png, hex3_scenarios_*.
Assessment: WF1's field model matters for E2 at FP2 and FP3, part of it through the E-normalization convention (ECAT uses E0 = v*B0) rather than the EFL; the user decided not to have efl separate these (no "scenario 0"). Still to do: wf1_knob_jacobian.py (8 compiles, user-run) to test whether existing WF1 knobs (Exit dB, length, rhoE, rhoB, X) reproduce the EFL effect; analyse_wf1_efl_scenarios.py then reports the fit.

## 2026-10-23 — sr_test_k3: eval-queue mechanics test (one Jacobian round, FP3), NOT a fit
Dataset: E2 FP3 groups at VD1783, 3000 rays/group; lattice pg_June2025_corrected.
Free: Q9_B_SC, Q11_B_SC, B5_Exit_dB (trust ±15%, ±15%, ±1%). Fixed: everything else at NOMINAL, including the
FP2 section (no FP2 solution upstream). Method: jac_loop round 0, 5 candidates, FP3 loss (fp3_loss).
Result: base (nominal) FP3 loss 141.9; best real candidate r0c000 loss 0.73 (predicted 0.55, rho 0.999) at
Q9 +0.18%, Q11 -2.05%, B5 -0.16%: M12 -0.269, Offset12 3.69 mm, M16 -2.075, Offset16 3.75 mm. 3 of 5 candidates
have loss <= 4.72.
Assessment: a test of the queue mechanics only. With the FP2 section at nominal (FP2 loss ~1153), the FP3
optics do not describe the E2 machine, so these values say nothing about Q9/Q11/B5. It does not contradict the
2026-10-05 finding (Q9 ~+10% needed when starting from the FP2 solutions, where FP3 M12 is -0.70..-0.80; at
nominal upstream FP3 M12 is -0.05). Q11 moves FP3 M16 by ~-0.84 per 1% here (cf. ~0.7 per 1% on 2026-10-05).
Files: eval_queue/runs/sr_test_k3/rounds.json; DigitalTwin/eval_queue/outbox/sr_test_k3/res_*.csv.

## 2026-10-24 — E2 Hall-probe fields vs "leg down" (data, no simulation yet)
Dataset: archived Hall-probe readings during E2 (tables not included in this repository). E2 window defined by
Fernando; the segment after a ~1% setpoint step of Q1–Q10 (made without re-cycling) is used ("segment B"). "Leg down" = |Imax|→0 branch of Fernando's 2026-10-20 hysteresis scan, interpolated at the
E2 mean current readback (confirmed by Fernando via secar-pv-desk, 2026-10-24).
Result (segment B, % on |B|, Hall vs leg down): Q1 -0.62, Q2 -0.75, Q3 -0.69, Q4 -2.61, Q5 -0.81, Q6 -1.73,
Q7 -7.03, Q8 -1.95, Q9 -1.40, Q10 -4.78. Plot: Results/e2_hall_vs_legdown_segB.png (segA also made;
plot_e2_hall_vs_legdown.py).
Caveats: the hysteresis loops were measured ~2.3 years after E2 (possible probe drift 0.5–1.5%); the Q6
comparison uses two different probes; mT zero offsets matter at Q7/Q10 (~0.02 T).
**Correction to the 2026-10-07 entry (Archived Hall-probe settings applied at FP2):** its reference, the
"cycled" Hall-probe snapshot, is not a valid reference (2026-10-24). Its
percentages (e.g. Q1 +4.69%, Q6 +2.96%) should not be used; the leg-down comparison above replaces them.
Addendum 2026-10-24: Fernando: for E2 at FP2, Q8–Q10 don't matter; H1–H3 and O1 do. Same window/segment; loops
= 0.5 A scan of 2026-10-22, Fernando's choice. Segment B vs leg down: H1 -0.26%, H2 -8.23% (outside the loop, 4% below
leg up), O1 -6.63% (near leg up). H3: no valid Hall reading for E2; only stand-in is the current readback (-0.50 A) with Danfysik's curve, below its first point (1.5 A), so
undetermined to ±30% or worse, sign included. Caveat: fields at the probes are tiny (H1 6.6 mT, H2 9.1 mT,
O1 21.7 mT) and the 0 A down-leg reading (remanence + offset) is 15–20% of them; 1 mT offset change = 11–15% for
H1/H2, so few-% deviations are not significant. Plot now Q1–Q7, H1–H3, O1 (plot_e2_hall_vs_legdown.py).
Decisions (Fernando, 2026-10-24) for simulating E2 at FP2 with these fields: only the % vs leg down matters (absolute
Hall values irrelevant); new SC = nominal deck SC × (1 + %/100) for every magnet (e.g. Q7 0.981 × (1 − 0.0703));
use all bars as measured: the ~0.7% common offset is real, not probe drift;
Q7 −7.03% as measured (not capped at leg up); H1/H2/O1 as measured (may revisit); H3 at nominal; Q1H at nominal;
Q1's third multipole term (∝ Q1) follows Q1. Settings: Results/e2_hall_legdown_settings.csv.

## 2026-10-24 — FP2 E2 slopes with the E2 Hall-probe (vs leg down) settings
Dataset: E2 FP2 groups at VD1638, 3000 rays/group, seed 42; lattice pg_June2025_corrected (MISALIGN 1, WFs on).
Settings: Results/e2_hall_legdown_settings.csv (Q1–Q7, H1, H2, O1 = nominal SC × (1 + % vs leg down); H3, Q1H,
rest at nominal; decisions above). One compile (queue_eval private compile, run by secar-reality on Fernando's go).
Result [M12 mm/mrad, Off12 mm, M16 mm/%, Off16 mm, FP2 4-term loss]:
  data     -0.2472  +1.543  -0.3842  +0.537   0
  nominal  +0.2075  -1.663  +0.1982  -1.464   1153  (reproduces the 2026-10-07 nominal)
  Hall     +0.0550  -1.957  -0.0943  -1.585   630
Assessment: the measured E2 fields move both slopes toward the data: M12 covers ~33% of the gap (still the
wrong sign), M16 ~50% (now the right sign). Offsets move slightly away (by ~0.3 and ~0.1 mm). The fields alone do
not reproduce E2 at FP2. Cf. 2026-10-19: WF1 field model scenario 1 alone shifted FP2 M16 by ~-0.38 with M12
nearly unchanged. Files: Results/e2_hall_fp2_slopes.csv/.png, e2_hall_fp2_slopes.py.

## 2026-10-24 — FP2 knob ranking at the E2 Hall base (sr_fp2_rank69; DigitalTwin dt_eval, SPEC v0.4)
Dataset: E2 FP2 groups, 3000 rays/group; base = E2 Hall settings (base obs equal DigitalTwin's hall run).
Knobs (69): B1–B4, WF1 exit dB; WF1 rhoE, rhoB, length; H3 B_SC; X/Y/Roll/Yaw/Pitch of Q1–Q7, B1–B4, WF1.
Loss: FP2 Mahalanobis (σ 0.027, 0.36, 0.3, 1.5; fit covariances). Base loss 204.3. 139/139 points ok.
Result (per equal unit; Results/fp2_rank69.csv/.png, rank_fp2_knobs.py): offsets are dominated by the exit kicks
(B1 −15, B2 −18, B4 +14, WF1 +14 mm per 1%); M12 (needs −0.30) moves most with B2 dB −0.14, B1 dB −0.11, B3 dB
−0.08 per 1%, then Q3_X +0.05, Q4_X −0.05 per mm; M16 (needs −0.29) with WF1 dB −0.26, B1/B2 dB +0.24, WF1 rhoE +0.14
per 1%. Nonlinear over the step: B1_X (curv 3.4), B2_X (1.8), B4 dB (1.6).
Linear bounded least squares within the trust ranges: all 69 → loss 0.08 with 9 knobs at bounds; fields only
→ 2.55 with all four dipole dB at +1% and WF1 length/rhoB at bounds; misalignments only → 1.22 with 57 at bounds.
Assessment: a fit is linearly reachable only with many knobs at their bounds (degenerate, symptom of a missing
model element rather than a solution); next: ridge-regularized loop (jac_loop) with a chosen subset.

## 2026-10-24 — FP2 fit, 17 knobs, at the E2 Hall base (sr_fp2_fit17; DigitalTwin dt_eval, SPEC v0.4)
Free: B1–B4, WF1 exit dB; WF1 rhoE, rhoB, length; H3 B_SC; Q1–Q5 X; Q1, Q2, WF1 yaw (trust as sr_fp2_rank69, yaw ±0.2°).
Fixed: Q1–Q7, H1, H2, O1 at E2 Hall settings; rest nominal. Loss and sampling: FP2 Mahalanobis (σ 0.027, 0.36, 0.3, 1.5).
Result: round 0 (75 evaluations, 3 min): base loss 204.3 → best r0c037 1.79 (predicted 3.62, rho 1.01); 16/40
candidates ≤ 4.72 (1σ). Best: M12 −0.264, Off12 1.38, M16 −0.262, Off16 2.00. Stopped by the physics-stop rule:
WF1 exit dB at its +2% bound in the best candidate (2 of 16 at the bound).
Family (16) medians: all exit kicks positive — B1 +0.23%, B2 +0.35%, B3 +0.54%, B4 +0.39%, WF1 +1.6%; quad X
alternating 1–2 mm (Q1 +0.9, Q2 −0.8, Q3 −1.7, Q4 +1.9, Q5 −1.3 mm; 30–63% of ±3 mm); WF1 geometry, H3 and yaws
near zero (≤ 10% of range).
Assessment: a good FP2 fit exists, but it needs WF1 +1.6–2% and 1–2 mm alternating quad shifts, both large. The
common positive sign of all exit kicks suggests a common cause (e.g. beam rigidity / reference energy) rather than
five independent dipole errors — hypothesis, not tested. Files: eval_queue/runs/sr_fp2_fit17/, DigitalTwin outbox.

### 2026-10-24 — Viewer-plane counting convention (decision)
Fernando approved: a ray stopped AT a viewer plane (ECAT transmission index == plane index) is not counted as seen
there; the readout uses ti > idx ("survived the plane") instead of ti >= idx. Raised by DigitalTwin from the VD1515
group-10 profile (19–20 of 3000 rays stopped there, |X| > 150 mm). For group 10 at FP2 the change is 0 rays. DigitalTwin
quantifies the impact on all FP2/FP3 groups (N, H), then implements it in the dt_eval readout with new pins; fit17 is
rerun only if an observable moves by > 0.1σ of the FP2 loss.
Result (DigitalTwin, 2026-10-24, exports/secar-reality_readout-strict_2026-10-24/readout_strict_impact.csv): 0 rays
stopped at VD1638 or FP3 Slits/VD1783 in any E2 group (N and H); every FP2/FP3 observable identical, so fit17 stands.
Implemented as dt_eval dte-2 (bb66bf2d…) + dt_readout dr-1 (70c28803…); jac_loop PINNED updated (fp3_eval/fp3_loss dropped).

### 2026-10-24 — TODO (Fernando): compare candidate viewer images with the E2 viewer images
The E2 measured group positions (mean X per group) come from viewer images. Besides the slopes, the best
candidates' simulated viewer profiles/images (e.g. at VD1515, as in DigitalTwin's e2_g10_VD1515_profile.png) should be
compared with the measured E2 viewer images to see whether the candidates survive. Deferred by Fernando; do it
after the FP3 step. Note: at VD1515 the fit17 best (r0c037) moves the group-10 mean X by +13.5 mm vs the Hall base.
