"""Sensor noise vs model error (reviewer R1C4).

All model targets are FF_EE, the wrench after the Simulink low-pass filter,
so reported RMSEs are errors with respect to the filtered signal. This script
quantifies the different "noise" quantities so they are not confused:

1. raw-minus-filtered residual RMS(FRaw - FF_EE) during motion (dynamic recordings)
2. the same residual at the static QS poses
3. raw sensor std inside stationary windows of the dynamic recordings
   (commanded velocity and acceleration ~ 0 for >= 1 s); the random-motion
   recordings contain none
4. raw-minus-filtered residual on the 5 % lowest-acceleration frames
5. sample-to-sample noise std(diff(signal)) / sqrt(2) for raw and filtered signals
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from forcecal.data import QS_BENCH, QS_TRAIN, RESULTS_DIR, load_dyn, load_qs
from forcecal.metrics import AXES_F, save

FS = 250  # Hz
STILL_VEL, STILL_ACC, MIN_RUN = 1e-4, 1e-4, FS


def per_axis(v):
    return {ax: float(v[k]) for k, ax in enumerate(AXES_F)}


def stationary_runs(d):
    still = (np.abs(d["xd"]).max(1) < STILL_VEL) & (np.abs(d["xdd"]).max(1) < STILL_ACC)
    runs, start = [], None
    for k, s in enumerate(np.append(still, False)):
        if s and start is None:
            start = k
        elif not s and start is not None:
            if k - start >= MIN_RUN:
                runs.append((start, k))
            start = None
    return runs


def main():
    res = {"qs": {}, "dynamic": {}}
    for name, f, stat in (("train_poses", QS_TRAIN, False), ("benchmark_poses", QS_BENCH, True)):
        q = load_qs(f, stationary_only=stat)
        r = q["fraw"][:, :3] - q["ff_ee"][:, :3]
        res["qs"][name] = {"rms_raw_minus_filtered": per_axis(np.sqrt((r**2).mean(0))),
                           "std_raw_minus_filtered": per_axis(r.std(0))}

    for i in (1, 2, 4):
        d = load_dyn(i)
        fr, fe = d["fraw"][:, :3], d["ff_ee"][:, :3]
        out = {
            "rms_raw_minus_filtered": per_axis(np.sqrt(((fr - fe) ** 2).mean(0))),
            "sample_noise_raw": per_axis(np.diff(fr, axis=0).std(0) / np.sqrt(2)),
            "sample_noise_filtered": per_axis(np.diff(fe, axis=0).std(0) / np.sqrt(2)),
        }
        runs = stationary_runs(d)
        if runs:
            stds = np.array([fr[a:b].std(0) for a, b in runs])
            out["stationary_windows"] = {"count": len(runs), "total_s": sum(b - a for a, b in runs) / FS,
                                         "raw_std_median": per_axis(np.median(stds, 0))}
        else:
            out["stationary_windows"] = {"count": 0}
        # "Near-stationary" frames: lowest 5 % of linear acceleration magnitude
        acc = np.linalg.norm(d["xdd"][:, :3], axis=1)
        low = acc <= np.percentile(acc, 5)
        out["low_accel_5pct"] = {"n": int(low.sum()),
                                 "rms_raw_minus_filtered": per_axis(np.sqrt(((fr[low] - fe[low]) ** 2).mean(0)))}
        res["dynamic"][f"L{i}"] = out
        print(f"L{i}: RMS(raw-filt) {np.round(np.sqrt(((fr - fe) ** 2).mean(0)), 3)}  "
              f"stationary windows {out['stationary_windows']['count']}")

    save(res, RESULTS_DIR / "noise.json")


if __name__ == "__main__":
    main()
