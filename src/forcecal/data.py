"""Loaders for the logged JSONL data (see data/README.md for the schema)."""
import gzip
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

QS_TRAIN = "1_Quasi_Static_Log2.jsonl.gz"
QS_BENCH = "manual_fresh_log.jsonl.gz"
QS_JOINTS = "1_TheRightWay.csv"
DYN_FILES = {i: f"2_DyF_L{i}.jsonl.gz" for i in (1, 2, 3, 4)}
DYN_TRAIN_POOL = (1, 2)  # recordings used for the reported dynamic models
DYN_BENCH = 4            # independent benchmark recording

# Threshold on |commanded Cartesian velocity| used to keep only stationary
# frames from the manually jogged QS benchmark log.
STATIONARY_VEL = 1e-4


def _read_jsonl(name):
    path = DATA_DIR / name
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_qs(name, stationary_only=False):
    """Quasi-static log -> dict of arrays.

    rpy: end-effector roll/pitch/yaw [deg] (model input)
    ff_ee: Simulink-filtered wrench in the EE frame [N, N*m] (model target)
    fraw: unfiltered sensor wrench
    """
    rows = _read_jsonl(name)
    if stationary_only:
        rows = [r for r in rows if np.all(np.abs(r["X_Dot_Cmds"]) < STATIONARY_VEL)]
    return {
        "rpy": np.array([r["X_Pos"][3:6] for r in rows]),
        "pos": np.array([r["X_Pos"] for r in rows]),
        "ff_ee": np.array([r["FF_EE"][:6] for r in rows]),
        "fraw": np.array([r["FRaw"][:6] for r in rows]),
    }


def load_qs_joints():
    """Commanded joint angles [deg] for the 362 QS poses (same order as the QS log)."""
    return np.loadtxt(DATA_DIR / QS_JOINTS, delimiter=",", skiprows=1)


def load_dyn(i):
    """Dynamic recording L<i> -> dict of arrays.

    X: model input = [roll, pitch, yaw, 6-DOF commanded Cartesian acceleration]
       (POS[3:6] + X_DoubleDot_Cmds, exactly as in the original notebooks)
    y: dynamic target FF_EE_Dynamic = FF_EE - FF_EE_Quasi (computed online)
    """
    rows = _read_jsonl(DYN_FILES[i])
    d = {
        "t": np.array([r["timestamp"] for r in rows]),
        "pos": np.array([r["POS"] for r in rows]),
        "xdd": np.array([r["X_DoubleDot_Cmds"] for r in rows]),
        "xd": np.array([r["X_Dot_Cmds"] for r in rows]),
        "fraw": np.array([r["FRaw"][:6] for r in rows]),
        "ff_ee": np.array([r["FF_EE"][:6] for r in rows]),
        "ff_quasi": np.array([r["FF_EE_Quasi"][:6] for r in rows]),
        "y": np.array([r["FF_EE_Dynamic"][:6] for r in rows]),
    }
    d["X"] = np.hstack([d["pos"][:, 3:6], d["xdd"]])
    # Same target but built from the unfiltered sensor signal.
    d["y_raw"] = d["fraw"] - d["ff_quasi"]
    return d


def stack(dicts, key):
    return np.vstack([d[key] for d in dicts])
