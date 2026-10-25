# config.py reference

### Base paths and active lattice file
**[VERIFIED]** `base = "pg_June2025"` (config.py:17) selects
`fox_file = matrix_file = "pg_June2025"`, i.e. the active lattice is
`COSY10.2/pg_June2025.fox`. A commented-out alternative,
`base = "Shumpei_FP4"`, sits directly above it (config.py:16) —
unclear if this is a prior/parallel lattice variant; not investigated.
Source: config.py:8-20
Recorded: 2026-08-07

### `cosy_lines` — element/parameter → .fox line map
**[VERIFIED]** `cosy_lines` (config.py:40-198) is a nested dict:
`cosy_lines[element_name][parameter_name] = {'line': ..., 'text1':
..., 'nom': ..., 'text2': ...}`. `line` is either a single line number
or a `[line_a, line_b]` pair (for parameters that appear twice in the
file, e.g. entrance/exit of a fringe-field region). `nom` is the
nominal value currently baked into the `.fox` file at that line. This
is the mechanism `tuneChanges` uses to rewrite the lattice file (see
`simulation/entry-points.md`).

Parameters that recur per-element: `Roll`, `dXY` (x,y offset), `B_SC`
(scale factor on nominal field, `nom` typically 1.0 but not always —
e.g. `Q7_SC` nom is 0.981, `Q11_SC` nom is 0.985, `Q12_SC` nom is 0.97,
`Q13_SC` nom is 0.86, `Q14_SC` nom is 1.327, `Q15_SC` nom is 1.421,
`H1_SC` nom is 1.02). Drift elements (named by 4-digit viewer/slit
IDs like `'1515'`, `'1542'`) instead have a `dist` parameter
(two-component: distance to element, distance past it). `WF1` also has
a `WFR` parameter (Wien filter radius/geometry, nom `[7.0, 7.0]`, text2
carries additional COSY keywords `NE1 NM1 4`).

Elements covered (in beamline order): Q1, Q2, B1, B2, H1, Q3, Q4, Q5,
B3, B4, H2, Q6, Q7, WF1, H3, O1, Q8, Q9, B5, B6, Q10, Q11, WF2, Q12,
Q13, B7, B8, Q14, Q15, plus drift markers 1515, 1542, 1638, 1688, 1783,
1836, 1879.
Source: config.py:40-198
Recorded: 2026-08-07

### `elements_dict` — beamline geometry
**[VERIFIED]** `elements_dict` (config.py:206-285) maps a descriptive
element-boundary name (e.g. `"Q1 Entrance (TA-upQ1)"`) to the drift
distance in meters *from the previous entry to this one*. Comment at
config.py:205 notes the element count and distances **must match**
those in the COSY `.fox` file — this is a manually-maintained
parallel structure, not derived from the `.fox` file automatically.
`element_names_full`/`element_names_stripped` (config.py:297-298)
strip the parenthetical `(...)` annotation for display;
`z_coordinates` (config.py:301-302) is the cumulative sum, i.e.
absolute z-position in meters from Target Center.
Source: config.py:206-302
Recorded: 2026-08-07

See `secar/layout.md` for the actual element sequence and lengths.

### Diagnostics: viewers, apertures, detectors, faraday cups
**[VERIFIED]** Four dicts in config.py:308-379 (`viewers`, `apertures`,
`detectors`, `faraday_cups`) each map a device ID to
`{'index': <position in element_names_stripped>, 'misalignment':
(dx_mm, dy_mm), ...}`. Viewer `size_mm` values are computed from inch
dimensions × 25.4, with a `/sqrt(2)` factor on one axis for several
devices (viewers are likely mounted at 45°, but that interpretation is
**[UNVERIFIED]** — not confirmed against a hardware drawing).
`misalignment` units are mm.
Source: config.py:308-379
Recorded: 2026-08-07
