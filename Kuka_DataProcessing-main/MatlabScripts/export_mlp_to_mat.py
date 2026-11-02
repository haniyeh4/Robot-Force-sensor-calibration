#!/usr/bin/env python3
"""
Export sklearn MLPRegressor weights and StandardScaler stats to MATLAB .mat

This script loads the trained model produced by dataCollection/4_Dy_MLP_Train.py
and exports:
  W1, b1, W2, b2, W3, b3, mu, invsigma

Shapes (example):
  W1: (H1, F)   b1: (H1, 1)
  W2: (H2, H1)  b2: (H2, 1)
  W3: (O,  H2)  b3: (O,  1)
  mu: (F,  1)   invsigma: (F, 1)

Usage (defaults target the latest Aug29 experiment layout):
  python MatlabScripts/export_mlp_to_mat.py \
      --exp-name Aug29 \
      --model-dir models/Aug29/mlp \
      --output force_mlp_params.mat

If --model-dir is not provided, the script will use:
  models/<exp-name>/mlp
"""

import argparse
import os
from pathlib import Path
import numpy as np
from scipy.io import savemat
import joblib

# Hardcoded experiment name (edit here if needed)
EXP_NAME = 'Aug29'


def resolve_model_dir(model_dir: str | None) -> Path:
    if model_dir:
        return Path(model_dir)
    # Only consider the experiment folder under models/<EXP_NAME>/mlp
    return Path("models") / EXP_NAME / "mlp"


def export_mlp_to_mat(mlp, scaler, mat_path: Path) -> None:
    # sklearn stores weights as (n_in, n_out); transpose to (n_out, n_in)
    coefs = mlp.coefs_
    intercepts = mlp.intercepts_

    if len(coefs) != len(intercepts) or len(coefs) < 1:
        raise ValueError(
            f"Unexpected MLP structure: coefs={len(coefs)}, intercepts={len(intercepts)}"
        )

    export_dict: dict[str, np.ndarray] = {}

    print("Export shapes:")
    for idx, (w, b) in enumerate(zip(coefs, intercepts), start=1):
        W = w.T.astype(np.float32)
        B = b.astype(np.float32).reshape(-1, 1)
        export_dict[f"W{idx}"] = W
        export_dict[f"b{idx}"] = B
        print(f"  W{idx}", W.shape, f"b{idx}", B.shape)

    mu = scaler.mean_.astype(np.float32).reshape(-1, 1)
    invsigma = (1.0 / scaler.scale_).astype(np.float32).reshape(-1, 1)
    export_dict["mu"] = mu
    export_dict["invsigma"] = invsigma
    print("  mu", mu.shape, "invsigma", invsigma.shape)

    mat_path.parent.mkdir(parents=True, exist_ok=True)
    savemat(str(mat_path), export_dict)
    print(f"Saved {mat_path}")


def main():
    parser = argparse.ArgumentParser(description="Export sklearn MLP to MATLAB .mat")
    parser.add_argument("--model-dir", type=str, default=str(Path("models") / EXP_NAME / "mlp"),
                        help="Directory containing mlp_model.pkl and mlp_scaler.pkl (default: models/EXP_NAME/mlp)")
    parser.add_argument("--model-filename", type=str, default="mlp_model.pkl",
                        help="Filename of the saved sklearn MLP model")
    parser.add_argument("--scaler-filename", type=str, default="mlp_scaler.pkl",
                        help="Filename of the saved StandardScaler")
    parser.add_argument("--output", type=str, default="force_mlp_params.mat",
                        help="Output .mat file path (default inside model dir)")

    args = parser.parse_args()

    model_dir = resolve_model_dir(args.model_dir)
    model_path = model_dir / args.model_filename
    scaler_path = model_dir / args.scaler_filename

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

    print(f"Loading model from: {model_path}")
    print(f"Loading scaler from: {scaler_path}")
    mlp = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Determine output path
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = model_dir / out_path

    export_mlp_to_mat(mlp, scaler, out_path)


if __name__ == "__main__":
    main()


