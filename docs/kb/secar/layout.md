# SECAR beamline layout (as encoded in the simulation)

**[VERIFIED — but see caveat]** Element sequence and drift lengths
(meters) from Target Center to DSSD, as given by `elements_dict` in
`config.py:206-285`. This is the *simulation's* record of the layout;
it has not been cross-checked against a design drawing or survey
document, and `config.py:205` itself notes this dict is manually kept
in sync with the `.fox` file rather than derived from it.

Sequence (name: cumulative length of that segment, m):
Target Center → Q1 Entrance (0.800527) → Q1 Exit (0.2498) →
D1477 Aperture Drive (0.1060544) → Q2 Entrance (0.0844356) →
Q2 Exit (0.2979) → FC1485 (0.391283) → B1 Entrance (0.189755) →
B1 Exit (0.491123652) → B2 Entrance (0.999778) →
B2 Exit (0.491137833) → VD1515 (0.87087) → H1 Entrance (0.297381) →
H1 Exit (0.263) → Q3 Entrance (0.268763) → Q3 Exit (0.3499) →
Q4 Entrance (0.35139) → Q4 Exit (0.3467) → Q5 Entrance (0.213664) →
Q5 Exit (0.3466) → FP1 Slits (0.152895825) → VD1542 (0.080755344) →
B3 Entrance (0.267380831) → B3 Exit (0.491574165) →
B4 Entrance (0.509073) → B4 Exit (0.492634453) →
H2 Entrance (0.297393) → H2 Exit (0.264) →
AP1568 Aperture Drive (0.23027568) → Q6 Entrance (0.30808432) →
Q6 Exit (0.3395) → Q7 Entrance (0.196437) → Q7 Exit (0.3472) →
WF1 Entrance (0.496188) → WF1 Exit (2.365) → H3 Entrance (0.498516) →
H3 Exit (0.263) → O1 Entrance (0.277156) → O1 Exit (0.262) →
FP2 Slits (1.750011) → VD1638 (0.28) → Q8 Entrance (0.59233) →
Q8 Exit (0.2473) → Q9 Entrance (0.394688) → Q9 Exit (0.3037) →
B5 Entrance (0.358445) → B5 Exit (0.927206165) →
B6 Entrance (0.35042) → B6 Exit (0.927206165) →
VD1688 (0.562676442) → Q10 Entrance (0.266293558) →
Q10 Exit (0.2628) → Q11 Entrance (0.64758) → Q11 Exit (0.3424) →
WF2 Entrance (0.997726) → WF2 Exit (2.365) →
FP3 Slits & VD1783 (4.644225559) → Q12 Entrance (0.208449441) →
Q12 Exit (0.2977) → Q13 Entrance (0.350425) → Q13 Exit (0.3008) →
B7 Entrance (0.659564) → B7 Exit (1.199913861) →
B8 Entrance (0.680262) → B8 Exit (1.199913861) →
VD1836 (0.389887068) → Q14 Entrance (0.469972932) →
Q14 Exit (0.2995) → Q15 Entrance (0.449858) → Q15 Exit (0.3012) →
UMCP (0.7112) → DMCP (1.397508) → FP4 Slits (0.512209509) →
VD1879 (0.208469264) → DSSD (0.3302)

Element inventory: 2 quadrupole triplets pre-first-dipole (Q1, Q2),
8 dipoles (B1-B8) in 4 achromatic pairs (B1/B2, B3/B4, B5/B6, B7/B8),
15 quadrupoles total (Q1-Q15), 3 hexapoles (H1, H2, H3), 1 octupole
(O1), 2 Wien filters (WF1, WF2), 4 viewers with dedicated diagnostic
info (VD1515, VD1542, VD1638, VD1688, plus VD1783/VD1836/VD1879),
4 focal-plane slit sets (FP1-FP4), plus FC1485 (Faraday cup), UMCP/DMCP
(detectors), DSSD (final detector).
Source: config.py:206-285
Recorded: 2026-08-07

Focal planes, per `main_new2.py:61-62`: FP1 = VD1542, FP2 = VD1638,
FP3 = "FP3 Slits", FP4 = DSSD.
Source: main_new2.py:60-62
Recorded: 2026-08-07

**[VERIFIED, user] — true focal plane vs. viewer position, important
for any focus/spot-width calculation**: the *true* focal plane (where
beam spot width is actually minimized in nominal/design optics) is
**not** always the same z-position as the diagnostic viewer used to
observe it:
- **FP2**: the true FP2 (the "FP2 Slits" element) is 0.28 m
  *upstream* of VD1638 — this is explicit in `elements_dict`:
  `"FP2 Slits (doO1-SLH1638)": 1.750011` then `"VD1638
  (SLH1638-VD1638)": 0.28` (config.py:249-250). Confirmed by the user
  directly, and used as the basis of the E3 analysis in `runs.md`
  (translating a spot-width minimization observed at VD1638 back to
  the true FP2 location).
- **FP3**: same situation — VD1783 is not exactly at the true FP3
  position — but the offset is small enough that `elements_dict`
  combines them into one entry, `"FP3 Slits & VD1783
  (doWF2-SLH_VD1783)": 4.644225559` (config.py:266), rather than
  splitting them the way FP2/VD1638 are split. **Do not read this
  combined naming as "FP3 slits and VD1783 are at the same
  z-position"** — they aren't, the gap is just small enough not to be
  broken out as its own line in this dict.
Source: user, 2026-08-07 conversation; config.py:249-250, 266
Recorded: 2026-08-07

**Correction/clarification, 2026-08-07**: the optimization work in
`runs.md` (from `Results/OpticsOptimization.xlsx` and
`sources/OpticsComparison-MatrixElements.pdf`) consistently uses
**FP4 = VD1836/VD1879**, not DSSD as the `main_new2.py:61-62`
`transmission_histogram` labels suggest. Both usages are internally
consistent with `elements_dict`'s ordering (VD1836 and VD1879 do sit
between B8 and DSSD, in the "FP4 region"), so this isn't a
contradiction — `main_new2.py`'s labeling is just choosing the *final
detector* (DSSD) as its nominal "FP4" marker for that particular
transmission plot, while the optics-matching work treats VD1836/VD1879
(the actual viewers/slits in that region) as FP4 for matrix-element
purposes. Don't assume "FP4" means the same physical element across
both contexts without checking.

**[VERIFIED] — Q9 misalignment, independently confirmed by measurement**:
`cosy_lines['Q9']['dXY']['nom'] = [0.000834, -0.000234]`
(config.py:133, i.e. 0.834 mm in X) is not just a simulation-encoded
placeholder — `sources/OpticsComparison-MatrixElements.pdf` (slide 18)
states this 0.834 mm X-misalignment is the "current (2022) measurement
misalignment" for Q9, confirmed by an actual alignment survey, and
that removing it (setting to zero) lowers transmission by 10% in the
simulation. This is the one specific value in `elements_dict`/
`cosy_lines` promoted from "as encoded in the simulation, unverified"
to "independently confirmed by measurement" — see `runs.md` for the
full context (this is separate from an unrelated, unconfirmed Q9
field-strength correction found during FP3 tuning).
Source: sources/OpticsComparison-MatrixElements.pdf, slide 18; config.py:131-135
Recorded: 2026-08-07
