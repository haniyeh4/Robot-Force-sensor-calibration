"""Quasi-static calibration (Sections 3.1 and 4.1, Figures 4-6).

- Third-degree polynomial on (roll, pitch, yaw) -> 6-D wrench.
- 362 poses, random 80/20 split (seed 42) -> 289 train / 73 test.
- Independent benchmark: 56 stationary frames from manually jogged poses.
- Degree sweep 1-5 with 5-fold CV (reviewers R1C7, R2C4).
- Joint-angle input model (reviewer R1C14).
- Torque metrics (reviewer R1C13).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import numpy as np
from sklearn.model_selection import KFold, cross_val_predict, train_test_split

from forcecal import plots
from forcecal.data import QS_BENCH, QS_TRAIN, RESULTS_DIR, load_qs, load_qs_joints
from forcecal.metrics import AXES_F, AXES_T, save, summarize
from forcecal.models import qs_poly

SEED = 42


def main():
    train = load_qs(QS_TRAIN)
    bench = load_qs(QS_BENCH, stationary_only=True)
    X, y = train["rpy"], train["ff_ee"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    print(f"QS samples: {len(X)} (train {len(Xtr)}, test {len(Xte)}), benchmark {len(bench['rpy'])}")

    res = {"n_total": len(X), "n_train": len(Xtr), "n_test": len(Xte), "n_benchmark": len(bench["rpy"]),
           "split": "random sample-level, sklearn train_test_split(test_size=0.2, random_state=42)"}

    # Final model
    model = qs_poly(3).fit(Xtr, ytr)
    res["final_degree"] = 3
    res["n_terms_per_output"] = int(model.named_steps["poly"].n_output_features_)
    res["test"] = summarize(yte, model.predict(Xte))
    yb_pred = model.predict(bench["rpy"])
    res["benchmark"] = summarize(bench["ff_ee"], yb_pred)
    (RESULTS_DIR / "models").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, RESULTS_DIR / "models" / "qs_poly3.joblib")

    # Degree sweep: held-out test, benchmark, and 5-fold CV on all 362 poses
    kf = KFold(5, shuffle=True, random_state=SEED)
    res["degree_sweep"] = {}
    for deg in range(1, 6):
        m = qs_poly(deg).fit(Xtr, ytr)
        cv = cross_val_predict(qs_poly(deg), X, y, cv=kf)
        res["degree_sweep"][deg] = {
            "test": summarize(yte, m.predict(Xte)),
            "benchmark": summarize(bench["ff_ee"], m.predict(bench["rpy"])),
            "cv5": summarize(y, cv),
        }

    # Joint-angle input (commanded joint angles of the same 362 poses).
    # The benchmark log has no joint angles, so only the held-out split is available.
    q = load_qs_joints()
    assert len(q) == len(X)
    qtr, qte, _, _ = train_test_split(q, y, test_size=0.2, random_state=SEED)
    res["joint_angle_model"] = {}
    for deg in (2, 3):
        m = qs_poly(deg).fit(qtr, ytr)
        res["joint_angle_model"][deg] = {"test": summarize(yte, m.predict(qte))}

    save(res, RESULTS_DIR / "quasi_static.json")

    # Figures 4-6 (benchmark set)
    plots.pred_vs_meas(bench["ff_ee"][:, :3], yb_pred[:, :3], AXES_F,
                       "Quasi-static benchmark: force", "fig04_qs_force_benchmark.png", "Benchmark sample")
    plots.pred_vs_meas(bench["ff_ee"][:, 3:], yb_pred[:, 3:], AXES_T,
                       "Quasi-static benchmark: torque", "fig05_qs_torque_benchmark.png", "Benchmark sample", unit="N·m")
    plots.errors(yb_pred[:, :3] - bench["ff_ee"][:, :3], AXES_F,
                 "Quasi-static benchmark: force prediction error (degree 3)", "fig06_qs_force_errors.png",
                 "Benchmark sample")

    b = res["benchmark"]
    print(f"test MSE(all6) {res['test']['all6_mse']:.6f} | bench MAE {b['all6_mae']:.4f} RMSE {b['all6_rmse']:.4f} "
          f"| max |e| {b['Fx_max_abs']:.3f}/{b['Fy_max_abs']:.3f}/{b['Fz_max_abs']:.3f} N")
    for deg, r in res["degree_sweep"].items():
        print(f"  deg {deg}: test MSE {r['test']['all6_mse']:.5f}  bench RMSE {r['benchmark']['all6_rmse']:.4f}"
              f"  CV force RMSE {r['cv5']['force_rmse']:.4f}")


if __name__ == "__main__":
    main()
