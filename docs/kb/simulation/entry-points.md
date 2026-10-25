# Entry points and control flow

### Main script
**[VERIFIED]** `main_new2.py` is the active entry point. It imports
geometry/config from `config.py` and processing/plotting helpers from
`helper_functions.py`.
Source: main_new2.py:1-22
Recorded: 2026-08-07

The file has two active workflows, switched by which block is
commented in/out, plus dead/reference code in large commented
sections:

1. **Basic ECAT run** (lines 24-82, active by default): defines an
   `initialDistribution` dict, calls `runECAT(...)`, then produces a
   transmission histogram and colored ray plots.
2. **E3 optimization** (lines 187-234, currently active at bottom of
   file): imports `load_projection, optimizeMCMC` from
   `ViewerAnalysis.py` and defines a `viewerImages` dict pointing at
   `E3_optimization/VD1515_*` and `E3_optimization/VD1542_*` files as
   comparison data for fitting initial beam parameters. The actual
   `optimizeMCMC(...)` call is commented out (lines 206-221) — as of
   this read, running the file does not launch an optimization, it
   only sets up the `viewerImages` dict.
Source: main_new2.py:24-234
Recorded: 2026-08-07

### `runECAT()` — the core simulation function
**[VERIFIED]** Defined in helper_functions.py:496. Signature:
`runECAT(initialDistribution=None, run_cosy_flag=False, foxFile=fox_file, tuneChanges=None, save_rays_flag=False, final_index=...)`.

Control flow:
1. Loads beamline geometry pickle (`beamline_geometry`, from
   `config.py`) and computes chamber polygons at each z-position via
   `find_polygons_at_z`.
2. If `tuneChanges` is not None, calls `prepareTuneChanges(tuneChanges)`
   to get `params, values, dipoleWFchanges`.
3. If `run_cosy_flag` is True: builds a COSY changes file if there are
   tune changes (`prepareCOSYchanges`, `createCosyFile`), otherwise
   runs the base `foxFile`. Either way it invokes COSY via
   `run_cosy_and_save_fort(...)`, then `process_fort_files(...)`, then
   loads the resulting per-element transfer matrices via
   `load_optics_data(...)`. See `cosy/invocation.md` for what this
   actually shells out to.
4. If `run_cosy_flag` is False: skips COSY entirely and just loads
   previously-saved transfer matrices for `foxFile` via
   `load_optics_data`.
5. Generates initial rays via `generateInitialDistribution(initialDistribution)`.
6. Propagates rays element-by-element with `process_all_rays(...)`,
   checking transmission against the beampipe polygons, up to
   `final_index`.
7. Extracts/sorts per-ray coordinate arrays with `extract_coordinates(...)`.

Implication: `run_cosy_flag=False` reuses whatever transfer matrices
are already on disk for that `foxFile` — it does **not** re-derive them
from the current `.fox` file contents. If the `.fox` file was edited
since the last COSY run, `run_cosy_flag=False` will silently use stale
matrices.
Source: helper_functions.py:496-556
Recorded: 2026-08-07

### `tuneChanges` — the free-parameter mechanism
**[VERIFIED]** Format: list of `[element_name, parameter_name, value]`
triples, e.g. `[['Q6', 'B_SC', -0.027], ['WF1 Exit', 'dB', 0.0025]]`.
`element_name`/`parameter_name` pairs are looked up against
`cosy_lines` in `config.py` (see `simulation/config-reference.md`) to
find which `.fox` file line to rewrite. This is the mechanism by which
"free parameters" in the optimization protocol get turned into an
actual COSY run — any parameter that has an entry in `cosy_lines` is,
mechanically, tunable this way.
Source: main_new2.py:47-50, config.py:40-198
Recorded: 2026-08-07
