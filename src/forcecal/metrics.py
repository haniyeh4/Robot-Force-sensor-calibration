"""Error metrics.

The manuscript's MAE/RMSE values are averaged over all six wrench components
(Fx, Fy, Fz in N and Tx, Ty, Tz in N*m) -> reported here as `all6_*`.
Force-only and per-axis values are reported alongside.
"""
import json

import numpy as np

# Full-scale output used for the %FSO numbers in the manuscript (0.54 N -> 0.108 %).
FSO_N = 500.0
AXES_F = ("Fx", "Fy", "Fz")
AXES_T = ("Tx", "Ty", "Tz")


def summarize(y_true, y_pred):
    e = np.asarray(y_pred, float) - np.asarray(y_true, float)
    out = {
        "n": int(len(e)),
        "all6_mse": float(np.mean(e**2)),
        "all6_mae": float(np.mean(np.abs(e))),
        "all6_rmse": float(np.sqrt(np.mean(e**2))),
    }
    ef = e[:, :3]
    out["force_mae"] = float(np.mean(np.abs(ef)))
    out["force_rmse"] = float(np.sqrt(np.mean(ef**2)))
    mag = np.linalg.norm(ef, axis=1)
    out["force_max_magnitude"] = float(mag.max())
    out["force_max_magnitude_pct_fso"] = float(100 * mag.max() / FSO_N)
    for k, ax in enumerate(AXES_F):
        out[f"{ax}_mae"] = float(np.mean(np.abs(ef[:, k])))
        out[f"{ax}_rmse"] = float(np.sqrt(np.mean(ef[:, k] ** 2)))
        out[f"{ax}_max_abs"] = float(np.abs(ef[:, k]).max())
        out[f"{ax}_max_abs_pct_fso"] = float(100 * np.abs(ef[:, k]).max() / FSO_N)
    if e.shape[1] == 6:
        et = e[:, 3:]
        out["torque_mae"] = float(np.mean(np.abs(et)))
        out["torque_rmse"] = float(np.sqrt(np.mean(et**2)))
        for k, ax in enumerate(AXES_T):
            out[f"{ax}_mae"] = float(np.mean(np.abs(et[:, k])))
            out[f"{ax}_rmse"] = float(np.sqrt(np.mean(et[:, k] ** 2)))
            out[f"{ax}_max_abs"] = float(np.abs(et[:, k]).max())
    return out


def force_only(y_true, y_pred):
    """Metrics on the three force components only (e.g. against the raw signal)."""
    return summarize(np.asarray(y_true)[:, :3], np.asarray(y_pred)[:, :3])


def save(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
    print(f"wrote {path}")
