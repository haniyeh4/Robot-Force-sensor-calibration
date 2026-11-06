"""Dynamic calibration, MLP (Sections 3.2.4 and 4.2.2, Figures 9-10).

Same data, inputs, target and split as scripts/02_dynamic_poly.py.
Architecture 9-64-128-64-6 (Linear-ReLU-BatchNorm-Dropout(0.2)), Adam lr 1e-3,
MSE loss, batch 64, 30 epochs, no scheduler or early stopping.

Differences from the original (unseeded) notebook run:
- fixed seeds; the model is trained for several seeds and mean +- std is reported,
- the input StandardScaler is fitted on the training split only and saved with the weights.

Also reports recording-level evaluation and CPU inference latency (reviewer R1C19).
"""
import argparse
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from forcecal import plots
from forcecal.data import DYN_BENCH, DYN_TRAIN_POOL, RESULTS_DIR, load_dyn, stack
from forcecal.metrics import AXES_F, force_only, save, summarize
from forcecal.models import force_predictor

SPLIT_SEED = 42
EPOCHS, BATCH, LR = 30, 64, 1e-3
FIG_WINDOW = slice(5000, 10000)


def train(Xtr, ytr, seed, epochs=EPOCHS):
    torch.manual_seed(seed)
    np.random.seed(seed)
    scaler = StandardScaler().fit(Xtr)
    Xt = torch.tensor(scaler.transform(Xtr), dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32)
    model = force_predictor()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()
    g = torch.Generator().manual_seed(seed)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt), generator=g)
        total = 0.0
        for k in range(0, len(Xt), BATCH):
            b = perm[k:k + BATCH]
            if len(b) < 2:  # BatchNorm needs >1 sample
                continue
            opt.zero_grad()
            loss = loss_fn(model(Xt[b]), yt[b])
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        print(f"    seed {seed} epoch {epoch + 1}/{epochs} train MSE {total / len(Xt):.6f}", end="\r")
    print()
    return model.eval(), scaler


@torch.no_grad()
def predict(model, scaler, X):
    return model(torch.tensor(scaler.transform(X), dtype=torch.float32)).numpy().astype(float)


def latency(model, scaler, X, reps=2000):
    """Median wall time per forward pass on CPU (eval mode), including input scaling."""
    out = {}
    for bs in (1, 64):
        xb = X[:bs]
        for _ in range(50):
            predict(model, scaler, xb)
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            predict(model, scaler, xb)
            ts.append(time.perf_counter() - t0)
        out[f"batch{bs}_median_us"] = float(np.median(ts) * 1e6)
        out[f"batch{bs}_p99_us"] = float(np.percentile(ts, 99) * 1e6)
    out.update(torch_threads=torch.get_num_threads(), torch=torch.__version__,
               cpu=platform.processor() or platform.machine(), platform=platform.platform())
    return out


def agg(runs, key):
    """mean/std over seeds for each scalar metric."""
    keys = runs[0][key].keys()
    return {k: {"mean": float(np.mean([r[key][k] for r in runs])),
                "std": float(np.std([r[key][k] for r in runs]))} for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()
    torch.use_deterministic_algorithms(True)

    rec = {i: load_dyn(i) for i in (1, 2, 4)}
    pool = [rec[i] for i in DYN_TRAIN_POOL]
    X, y, y_raw = stack(pool, "X"), stack(pool, "y"), stack(pool, "y_raw")
    idx_tr, idx_te = train_test_split(np.arange(len(X)), test_size=0.2, random_state=SPLIT_SEED)
    B = rec[DYN_BENCH]

    res = {"epochs": args.epochs, "batch_size": BATCH, "lr": LR, "seeds": args.seeds,
           "split": "random sample-level, sklearn train_test_split(test_size=0.2, random_state=42)",
           "n_train": len(idx_tr), "n_test": len(idx_te), "n_benchmark": len(B["X"])}
    n_params = sum(p.numel() for p in force_predictor().parameters() if p.requires_grad)
    res["n_parameters"] = int(n_params)

    runs, main_model = [], None
    for seed in args.seeds:
        model, scaler = train(X[idx_tr], y[idx_tr], seed, args.epochs)
        pte, pb = predict(model, scaler, X[idx_te]), predict(model, scaler, B["X"])
        r = {"seed": seed, "test": summarize(y[idx_te], pte), "benchmark": summarize(B["y"], pb),
             "test_vs_raw": force_only(y_raw[idx_te], pte), "benchmark_vs_raw": force_only(B["y_raw"], pb)}
        runs.append(r)
        print(f"  seed {seed}: test MAE {r['test']['all6_mae']:.4f} RMSE {r['test']['all6_rmse']:.4f} | "
              f"bench MAE {r['benchmark']['all6_mae']:.4f} RMSE {r['benchmark']['all6_rmse']:.4f} | "
              f"max mag {r['benchmark']['force_max_magnitude']:.3f} N")
        if main_model is None:
            main_model, main_scaler, main_pb = model, scaler, pb
    res["runs"] = runs
    res["main_seed"] = args.seeds[0]
    res["summary_over_seeds"] = {k: agg(runs, k) for k in ("test", "benchmark", "test_vs_raw", "benchmark_vs_raw")}

    out = RESULTS_DIR / "models"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(main_model.state_dict(), out / f"mlp_seed{args.seeds[0]}.pt")
    np.savez(out / f"mlp_seed{args.seeds[0]}_scaler.npz", mean=main_scaler.mean_, scale=main_scaler.scale_)

    # Recording-level evaluation (main seed)
    res["recording_level"] = {}
    for tr, te in ((2, 1), (1, 2)):
        model, scaler = train(rec[tr]["X"], rec[tr]["y"], args.seeds[0], args.epochs)
        res["recording_level"][f"train_L{tr}_test_L{te}"] = {
            "test": summarize(rec[te]["y"], predict(model, scaler, rec[te]["X"])),
            "benchmark": summarize(B["y"], predict(model, scaler, B["X"])),
        }

    res["latency"] = latency(main_model, main_scaler, B["X"])
    save(res, RESULTS_DIR / "dynamic_mlp.json")

    w = FIG_WINDOW
    total_pred = B["ff_quasi"][w, :3] + main_pb[w, :3]
    plots.pred_vs_meas(B["ff_ee"][w, :3], total_pred, AXES_F,
                       f"Dynamic benchmark (L{DYN_BENCH}, samples {w.start}-{w.stop}): MLP, seed {args.seeds[0]}",
                       "fig09_dyn_mlp_force_benchmark.png", lines=True)
    plots.errors(total_pred - B["ff_ee"][w, :3], AXES_F,
                 "Dynamic benchmark: MLP prediction error", "fig10_dyn_mlp_force_errors.png", lines=True)


if __name__ == "__main__":
    main()
