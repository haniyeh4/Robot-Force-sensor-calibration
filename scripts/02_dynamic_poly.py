"""Dynamic calibration, polynomial regression (Sections 3.2.3 and 4.2.1, Figures 7-8).

Protocol reproducing the reported numbers:
- Training pool: recordings L1 + L2 (88,223 samples at 250 Hz).
- Random sample-level 80/20 split (seed 42) -> test set.
- Independent benchmark: recording L4.
- Inputs: roll, pitch, yaw + 6-DOF commanded Cartesian acceleration, standardized.
- Target: FF_EE_Dynamic (filtered wrench minus quasi-static prediction).

Extra analyses requested by reviewers (existing data only):
- Degree sweep 1-4.
- Quasi-static-only baseline, i.e. predicting zero dynamic offset (R1C8).
- Ridge regularization and design conditioning (R1C11).
- Recording-level splits to quantify optimism of the random split (R1C4, R1C15, R2C3).
- Errors against the unfiltered sensor signal (R1C4).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

from forcecal import plots
from forcecal.data import DYN_BENCH, DYN_TRAIN_POOL, RESULTS_DIR, load_dyn, stack
from forcecal.metrics import AXES_F, force_only, save, summarize
from forcecal.models import dyn_poly

SEED = 42
FIG_WINDOW = slice(5000, 10000)  # benchmark window shown in Figures 7-8


def main():
    rec = {i: load_dyn(i) for i in (1, 2, 3, 4)}
    pool = [rec[i] for i in DYN_TRAIN_POOL]
    X, y, y_raw = stack(pool, "X"), stack(pool, "y"), stack(pool, "y_raw")
    idx_tr, idx_te = train_test_split(np.arange(len(X)), test_size=0.2, random_state=SEED)
    Xtr, ytr, Xte, yte = X[idx_tr], y[idx_tr], X[idx_te], y[idx_te]
    B = rec[DYN_BENCH]
    print(f"pool {len(X)} samples (train {len(Xtr)}, test {len(Xte)}), benchmark L{DYN_BENCH} {len(B['X'])}")

    res = {
        "train_pool": [f"L{i}" for i in DYN_TRAIN_POOL], "benchmark": f"L{DYN_BENCH}",
        "n_pool": len(X), "n_train": len(Xtr), "n_test": len(Xte), "n_benchmark": len(B["X"]),
        "split": "random sample-level, sklearn train_test_split(test_size=0.2, random_state=42)",
        "samples_per_recording": {f"L{i}": len(r["X"]) for i, r in rec.items()},
    }

    # Final degree-2 model
    model = dyn_poly(2).fit(Xtr, ytr)
    res["n_features"] = int(model.named_steps["poly"].n_output_features_)
    pte, pb = model.predict(Xte), model.predict(B["X"])
    res["final"] = {
        "degree": 2,
        "test": summarize(yte, pte),
        "benchmark": summarize(B["y"], pb),
        # Same predictions scored against the unfiltered sensor signal
        "test_vs_raw": force_only(y_raw[idx_te], pte),
        "benchmark_vs_raw": force_only(B["y_raw"], pb),
    }
    (RESULTS_DIR / "models").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, RESULTS_DIR / "models" / "dyn_poly2.joblib")

    # Degree sweep
    res["degree_sweep"] = {}
    for deg in (1, 2, 3, 4):
        m = dyn_poly(deg).fit(Xtr, ytr)
        res["degree_sweep"][deg] = {"test": summarize(yte, m.predict(Xte)),
                                    "benchmark": summarize(B["y"], m.predict(B["X"]))}
        print(f"  deg {deg}: test RMSE {res['degree_sweep'][deg]['test']['all6_rmse']:.4f}"
              f"  bench RMSE {res['degree_sweep'][deg]['benchmark']['all6_rmse']:.4f}")

    # Quasi-static-only compensation (dynamic offset predicted as zero)
    res["qs_only_baseline"] = {
        "test": summarize(yte, np.zeros_like(yte)),
        "benchmark": summarize(B["y"], np.zeros_like(B["y"])),
        "benchmark_vs_raw": force_only(B["y_raw"], np.zeros_like(B["y_raw"])),
    }

    # Conditioning and ridge
    feats_scaled = model[:-1].transform(Xtr)
    feats_raw = PolynomialFeatures(2, include_bias=False).fit_transform(Xtr)
    ones = np.ones((len(Xtr), 1))
    res["conditioning"] = {
        "cond_design_standardized": float(np.linalg.cond(np.hstack([ones, feats_scaled]))),
        "cond_design_unscaled": float(np.linalg.cond(np.hstack([ones, feats_raw]))),
    }
    m_raw = dyn_poly(2, scale=False).fit(Xtr, ytr)
    res["unscaled_ols"] = {"test": summarize(yte, m_raw.predict(Xte)),
                           "benchmark": summarize(B["y"], m_raw.predict(B["X"]))}
    res["ridge"] = {}
    for a in (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0):
        m = dyn_poly(2, Ridge(alpha=a)).fit(Xtr, ytr)
        res["ridge"][a] = {"test": summarize(yte, m.predict(Xte)),
                           "benchmark": summarize(B["y"], m.predict(B["X"])),
                           "coef_norm": float(np.linalg.norm(m.named_steps["reg"].coef_))}
    res["ridge"]["ols_coef_norm"] = float(np.linalg.norm(model.named_steps["reg"].coef_))

    # Recording-level evaluation: train on one recording, test on another (no shared samples)
    res["recording_level"] = {}
    for tr, te in ((2, 1), (1, 2)):
        for deg in (1, 2):
            m = dyn_poly(deg).fit(rec[tr]["X"], rec[tr]["y"])
            res["recording_level"][f"deg{deg}_train_L{tr}_test_L{te}"] = {
                "test": summarize(rec[te]["y"], m.predict(rec[te]["X"])),
                "benchmark": summarize(B["y"], m.predict(B["X"])),
            }
    # Leave-one-recording-out over L1-L3 (as in the original notebook pool)
    for te in (1, 2, 3):
        tr = [i for i in (1, 2, 3) if i != te]
        m = dyn_poly(2).fit(stack([rec[i] for i in tr], "X"), stack([rec[i] for i in tr], "y"))
        res["recording_level"][f"deg2_train_L{'+L'.join(map(str, tr))}_test_L{te}"] = {
            "test": summarize(rec[te]["y"], m.predict(rec[te]["X"])),
            "benchmark": summarize(B["y"], m.predict(B["X"])),
        }
    # Contiguous split: last 20 % of each pool recording held out
    tr_i, te_i, off = [], [], 0
    for r in pool:
        n = len(r["X"]); cut = int(0.8 * n)
        tr_i += range(off, off + cut); te_i += range(off + cut, off + n); off += n
    m = dyn_poly(2).fit(X[tr_i], y[tr_i])
    res["recording_level"]["deg2_contiguous_last20pct"] = {"test": summarize(y[te_i], m.predict(X[te_i])),
                                                          "benchmark": summarize(B["y"], m.predict(B["X"]))}

    save(res, RESULTS_DIR / "dynamic_poly.json")

    # Figures 7-8: total force (quasi-static + dynamic prediction) vs measured filtered force, benchmark window
    w = FIG_WINDOW
    total_pred = B["ff_quasi"][w, :3] + pb[w, :3]
    plots.pred_vs_meas(B["ff_ee"][w, :3], total_pred, AXES_F,
                       f"Dynamic benchmark (L{DYN_BENCH}, samples {w.start}-{w.stop}): polynomial, degree 2",
                       "fig07_dyn_poly_force_benchmark.png", lines=True)
    plots.errors(total_pred - B["ff_ee"][w, :3], AXES_F,
                 "Dynamic benchmark: polynomial prediction error", "fig08_dyn_poly_force_errors.png", lines=True)

    f, b = res["final"]["test"], res["final"]["benchmark"]
    print(f"deg2 test MAE {f['all6_mae']:.4f} RMSE {f['all6_rmse']:.4f} | bench MAE {b['all6_mae']:.4f} "
          f"RMSE {b['all6_rmse']:.4f} | max {b['Fx_max_abs']:.3f}/{b['Fy_max_abs']:.3f}/{b['Fz_max_abs']:.3f} "
          f"mag {b['force_max_magnitude']:.3f} N")


if __name__ == "__main__":
    main()
