"""
Timing characterization for the AgentSearch copy of the ECAT pipeline.
Run from within AgentSearch/ so local config.py, helper_functions.py,
and the compiled ray_functions module are found first on sys.path.
"""
import time
from config import element_names_stripped
from helper_functions import runECAT

FINAL_INDEX = element_names_stripped.index('O1 Exit')
print(f"final_index for 'O1 Exit' = {FINAL_INDEX} (elements up to/incl. O1, i.e. up to FP2 region)")

initialDistribution = {
    'type': 'mixed_rays',
    'nRays': 100,
    'target': {'option': 'gaussian_2d', 'param': {'mean': (0.1304/1000, -0.24/1000), 'sigma': (0.6652/1000, 0.5291/1000), 'rho': 0.0}},
    'angles': {'option': 'gaussian_2d', 'param': {'mean': (-0.223/1000, 0.0/1000), 'sigma': (4.8/1000, 5/1000), 'rho': 0.0}},
    'dE': {'option': 'normal', 'param': {'mu': -0.13/100, 'sigma': 1.05/100}},
    'dZ': {'option': 'fixed', 'param': 0},
}

# Step 1: run COSY once to (re)generate the nominal transfer matrices in the isolated AgentSearch dirs.
print("\n--- Step 1: run_cosy_flag=True (nominal, no tuneChanges) ---")
t0 = time.time()
result = runECAT(initialDistribution=dict(initialDistribution, nRays=10), run_cosy_flag=True, tuneChanges=[],
                  save_rays_flag=False, final_index=FINAL_INDEX)
t1 = time.time()
print(f"COSY matrix computation + tiny ray propagation (nRays=10): {t1 - t0:.2f} s")

# Step 2: time ray propagation only (run_cosy_flag=False, reusing matrices just computed), varying nRays.
print("\n--- Step 2: ray propagation only (run_cosy_flag=False), varying nRays ---")
for n in [100, 500, 1000, 5000, 10000]:
    dist = dict(initialDistribution, nRays=n)
    t0 = time.time()
    result = runECAT(initialDistribution=dist, run_cosy_flag=False, tuneChanges=[],
                      save_rays_flag=False, final_index=FINAL_INDEX)
    t1 = time.time()
    print(f"nRays={n:6d}: {t1 - t0:.3f} s  ({(t1 - t0)/n*1000:.4f} ms/ray)")
