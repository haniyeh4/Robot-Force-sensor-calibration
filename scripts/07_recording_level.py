"""Recording-level validation on the additional session (Aug 29), manuscript Section 3.5 / Table 3.

Second-degree polynomial model (same pipeline as 02_dynamic_poly.py: standardize -> degree-2
features -> least squares) evaluated on 11 random-motion recordings in two ways:
  (i)  random sample-level 80/20 split (random_state=42), as used for the main results;
  (ii) leave-one-recording-out (LORO): each recording held out once, trained on the others.
Both are done for recordings 1-10 (similar motion) and 1-11 (including recording 11, which
contains much larger accelerations). LORO errors are pooled over all held-out samples.
Also reports per-recording LORO errors and acceleration ranges (99th percentile of |a| and |alpha|).

Run from the repository root:  python scripts/07_recording_level.py
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from forcecal.data import DATA_DIR, RESULTS_DIR  # noqa: E402
from forcecal.metrics import save, summarize  # noqa: E402
from forcecal.models import dyn_poly  # noqa: E402

AUG29 = DATA_DIR / "aug29"
IDS = list(range(1, 12))


def load(i):
    X, y = [], []
    with gzip.open(AUG29 / f"2_DyF_L{i}.jsonl.gz", "rt") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                X.append(r["POS"][3:6] + r["X_DoubleDot_Cmds"][:6])  # roll, pitch, yaw + 6 commanded acc.
                y.append(r["FF_EE_Dynamic"][:6])
    return np.array(X), np.array(y)


def main():
    R = {i: load(i) for i in IDS}
    cat = lambda ids, k: np.vstack([R[i][k] for i in ids])  # noqa: E731
    res = {"n_samples": {f"L{i}": len(R[i][0]) for i in IDS},
           "acc_p99": {f"L{i}": {"linear": float(np.percentile(np.abs(R[i][0][:, 3:6]), 99)),
                                 "angular": float(np.percentile(np.abs(R[i][0][:, 6:9]), 99))} for i in IDS}}
    for name, ids in (("L1-L10", IDS[:10]), ("L1-L11", IDS)):
        X, y = cat(ids, 0), cat(ids, 1)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        rand = summarize(yte, dyn_poly(2).fit(Xtr, ytr).predict(Xte))
        P, T, per = [], [], {}
        for h in ids:
            tr = [i for i in ids if i != h]
            p = dyn_poly(2).fit(cat(tr, 0), cat(tr, 1)).predict(R[h][0])
            P.append(p); T.append(R[h][1])
            per[f"L{h}"] = summarize(R[h][1], p)
        loro = summarize(np.vstack(T), np.vstack(P))
        res[name] = {"random_split": rand, "leave_one_recording_out": loro, "loro_per_recording": per}
        print(f"{name}: RMSE (six components) random split {rand['all6_rmse']:.4f} | "
              f"leave-one-recording-out {loro['all6_rmse']:.4f}")
    print("Held-out RMSE of recording 11 (trained on 1-10):",
          f"{res['L1-L11']['loro_per_recording']['L11']['all6_rmse']:.3f}")
    save(res, RESULTS_DIR / "recording_level_aug29.json")


if __name__ == "__main__":
    main()
