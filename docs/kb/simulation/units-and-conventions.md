# Units and conventions at the Python boundary

**[VERIFIED]** In `initialDistribution` dicts passed to
`generateInitialDistribution()` / `runECAT()`:
- Transverse positions (`x`, `y`, target `mean`/`sigma`) are in
  **meters**. Values are written as e.g. `0.75/1000` (i.e. 0.75 mm
  expressed in meters) — see `main_new2.py:35` (`'mean': (0.1304/1000,
  -0.24/1000)`) and `helper_functions.py:279` (`'x': {'min':
  -0.75/1000, ...}`).
- Angles (`ax`, `ay`) are in **radians**. Written as e.g. `25/1000`
  (25 mrad) — `helper_functions.py:280`, and `main_new2.py:36`
  (`'sigma': (4.8/1000, 5/1000)`, i.e. 4.8/5.0 mrad).
- Energy deviation `dE` is **fractional dE/E**, not absolute energy or
  percent-as-integer. Written as e.g. `3.1/100` (3.1%) —
  `helper_functions.py:283`, `main_new2.py:38`.
- `dZ` (longitudinal) convention not yet confirmed beyond that it
  exists as a distribution parameter alongside the others; only seen
  used with `{'option': 'fixed', 'param': 0}` so far.

Source: main_new2.py:33-46, helper_functions.py:275-307
Recorded: 2026-08-07

**Correction/partial resolution, 2026-08-07**: Read
`sources/COSYBeamMan100.pdf` (COSY Beam Physics Manual). COSY's `SR`
(set ray) command — `SR <x> <a> <y> <b> <T> <D> <G> <Z> <color>;` —
takes `x`,`y` in meters and `a`,`b` explicitly "the angles in radians"
(manual p.13), matching the Python-side convention above exactly. So
for the *ray/phase-space* coordinates (`x`,`y`,`ax`,`ay`,`dE`) the two
sides agree: meters, radians, fractional dE/E respectively. See
`cosy/language-reference.md` for full citation.

**Important — do not generalize this to all "angle" parameters in the
`.fox` file.** COSY's own global rule (manual p.8) is "all units are
SI except voltages (kV) and angles (**degrees**)." That degrees rule
governs a *different* class of angle: element/misalignment parameters
like `RA <angle>` (roll) and `TA <ax> <ay>` (tilt), used in
`config.py`'s `cosy_lines` `Roll`/`dXY`-adjacent entries. Those are
literally in degrees, not radians — confirmed directly against
`COSY10.2/pg_June2025.fox:294-298`. See `cosy/language-reference.md`
for the full derivation; this is flagged there as the single most
likely place for a unit-conversion mistake in this codebase, since the
word "angle" means radians at the ray/phase-space level and degrees at
the element-misalignment level, within the same file.

**[UNVERIFIED], still open**: whether `createCosyFile`/
`prepareCOSYchanges` in `helper_functions.py` do any conversion when
writing a `tuneChanges` value into the `.fox` file — e.g. if a future
`tuneChanges` entry targets `Roll`, does the Python caller need to
supply degrees (matching `RA`'s nominal value convention) or does the
code convert from some other unit? Not yet checked — needs a direct
read of those two functions.

**[VERIFIED]** Beamline z-positions (`z_coordinates` in `config.py`)
are cumulative sums of the `elements_dict` drift/element lengths,
which are themselves in **meters** (e.g. `"Q1 Entrance": 0.800527`).
Source: config.py:206-302
Recorded: 2026-08-07
