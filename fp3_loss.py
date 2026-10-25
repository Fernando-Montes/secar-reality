"""
E2 FP3 loss, extracted unchanged from archive/2026-09-21/fp3_family_mc.py
(2026-08-18 definition) so it can be imported:

    loss = d12^T C12^-1 d12 + ((M16 - T_M16)/0.2)^2 + ((Off16 - T_Off16)/1.5)^2

d12 = [M12 - T_M12, Offset12 - T_Offset12]; C12 = covariance of the 3-point
OLS fit to the measured FP3 dalphaX data (groups 10, 14, 15), divided by 4
per the user's 2026-08-18 tightening. Order everywhere: [M12, Off12, M16, Off16].
2-sigma acceptance for 4 dof: 9.7156; 1-sigma: 4.7196.
"""
import numpy as np
from fp3_metrics import DALPHAX_GROUPS, DALPHAX_GROUPS_MEASURED_X, DE_GROUPS


def _m12_fit_covariance():
    groups = list(DALPHAX_GROUPS.keys())
    x = np.array([DALPHAX_GROUPS[g] for g in groups])
    y = np.array([DALPHAX_GROUPS_MEASURED_X[g] for g in groups])
    X = np.column_stack([np.ones(len(x)), x])          # beta = [intercept, slope]
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    rss = float(np.sum((y - X @ beta) ** 2))
    cov_full = (rss / (len(x) - 2)) * np.linalg.inv(X.T @ X)
    order = [1, 0]                                     # -> [slope, intercept]
    return beta[1], beta[0], cov_full[np.ix_(order, order)]


M12_TARGET, OFFSET12_TARGET, _COV_M12_RAW = _m12_fit_covariance()
COV_M12 = _COV_M12_RAW / 4
COV_M12_INV = np.linalg.inv(COV_M12)
M16_TARGET, OFFSET16_TARGET = -2.0956, 3.0608
SIGMA_M16, SIGMA_OFFSET16 = 0.2, 1.5
TARGETS = np.array([M12_TARGET, OFFSET12_TARGET, M16_TARGET, OFFSET16_TARGET])
CHI2_1SIGMA_DF4 = 4.7196
CHI2_2SIGMA_DF4 = 9.715641142541578

# loss = r^T H r with r = obs - TARGETS (block-diagonal H)
H = np.zeros((4, 4))
H[0:2, 0:2] = COV_M12_INV
H[2, 2] = 1 / SIGMA_M16 ** 2
H[3, 3] = 1 / SIGMA_OFFSET16 ** 2


def observables(mean_x_by_group):
    """{group: mean X [mm] at VD1783} -> np.array([M12, Off12, M16, Off16])."""
    a = np.array([DALPHAX_GROUPS[g] for g in DALPHAX_GROUPS])
    M12, Off12 = np.polyfit(a, [mean_x_by_group[g] for g in DALPHAX_GROUPS], 1)
    e = np.array([DE_GROUPS[g] for g in DE_GROUPS])
    M16, Off16 = np.polyfit(e, [mean_x_by_group[g] for g in DE_GROUPS], 1)
    return np.array([M12, Off12, M16, Off16])


def loss(obs):
    r = np.asarray(obs) - TARGETS
    return float(r @ H @ r)


def loss_gradient(obs, dobs):
    """dLoss/dknob by the chain rule: 2 r^T H dobs, with dobs = d[obs]/dknob."""
    r = np.asarray(obs) - TARGETS
    return float(2 * r @ H @ np.asarray(dobs))
