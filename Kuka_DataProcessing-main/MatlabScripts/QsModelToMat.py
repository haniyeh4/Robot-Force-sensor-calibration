# scripts/QsModelToMat.py
import joblib, numpy as np
from scipy.io import savemat

# scikit imports for type checks (safe even if some aren’t used)
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

EXP_NAME = 'Oct30'
PKL = "models/" + EXP_NAME + "/1_Quasi_Static_Poly.pkl"
MAT = "models/" + EXP_NAME + "/qs_poly_model_params.mat"

def _unwrap_any(obj):
    """
    Recursively search for (poly, scaler, reg) inside common wrappers.
    Returns (poly, scaler, reg). Any of them may be None if not present.
    """
    # dict-like bundle?
    if isinstance(obj, dict):
        # Try common keys first
        for k in ["model", "pipeline", "estimator", "best_estimator_", "regressor"]:
            if k in obj:
                poly, sc, reg = _unwrap_any(obj[k])
                if reg is not None:
                    return poly, sc, reg
        # Try all values
        for v in obj.values():
            poly, sc, reg = _unwrap_any(v)
            if reg is not None:
                return poly, sc, reg
        return None, None, None

    # Grid/Randomized search: go to best_estimator_
    if isinstance(obj, (GridSearchCV, RandomizedSearchCV)):
        return _unwrap_any(obj.best_estimator_)

    # TransformedTargetRegressor: regressor_ holds the trained regressor
    if isinstance(obj, TransformedTargetRegressor):
        return _unwrap_any(obj.regressor_)

    # Pipeline: look through steps
    if isinstance(obj, Pipeline):
        poly = None
        scaler = None
        reg = None
        for name, step in obj.named_steps.items():
            if isinstance(step, PolynomialFeatures):
                poly = step
            elif isinstance(step, StandardScaler):
                scaler = step
        # final step (or any step) that has coef_/intercept_ OR MOR
        # Prefer the last step
        last = list(obj.named_steps.values())[-1]
        # MultiOutputRegressor?
        if isinstance(last, MultiOutputRegressor):
            reg = last
        elif hasattr(last, "coef_") and hasattr(last, "intercept_"):
            reg = last
        else:
            # search other steps just in case
            for step in obj.named_steps.values():
                if isinstance(step, MultiOutputRegressor):
                    reg = step; break
                if hasattr(step, "coef_") and hasattr(step, "intercept_"):
                    reg = step; break
        return poly, scaler, reg

    # MultiOutputRegressor directly
    if isinstance(obj, MultiOutputRegressor):
        return None, None, obj

    # Bare estimator with coef_/intercept_
    if hasattr(obj, "coef_") and hasattr(obj, "intercept_"):
        return None, None, obj

    # Nothing matched
    return None, None, None

def _collect_coef_intercept(reg):
    """
    Return (COEF, INTERCEPT) as float32 with shape:
      COEF: (n_out, n_phi), INTERCEPT: (n_out, 1)
    Works for single estimator or MultiOutputRegressor of linear estimators.
    """
    # MultiOutputRegressor -> stack rows
    if isinstance(reg, MultiOutputRegressor):
        rows = []
        biases = []
        for est in reg.estimators_:
            if not (hasattr(est, "coef_") and hasattr(est, "intercept_")):
                raise ValueError("An inner estimator in MultiOutputRegressor lacks coef_/intercept_.")
            rows.append(np.asarray(est.coef_, dtype=np.float32).reshape(1, -1))
            biases.append(np.asarray(est.intercept_, dtype=np.float32).reshape(1, 1))
        coef = np.vstack(rows).astype(np.float32)            # (n_out, n_phi)
        intercept = np.vstack(biases).astype(np.float32)     # (n_out, 1)
        return coef, intercept

    # Single-output or multi-output regressor with 2D coef_
    coef = np.asarray(reg.coef_, dtype=np.float32)
    if coef.ndim == 1:
        coef = coef.reshape(1, -1)
    intercept = np.asarray(reg.intercept_, dtype=np.float32).reshape(-1, 1)
    return coef, intercept

def export_to_mat(pkl_path, mat_path):
    model = joblib.load(pkl_path)
    poly, scaler, reg = _unwrap_any(model)
    if reg is None:
        raise ValueError("Could not find a fitted regressor with coef_/intercept_ inside the saved object.")

    # COEF / INTERCEPT
    COEF, INTERCEPT = _collect_coef_intercept(reg)   # (n_out, n_phi), (n_out, 1)

    # POWERS from PolynomialFeatures
    if poly is None:
        raise ValueError(
            "PolynomialFeatures step not found in the saved object. "
            "Please save a Pipeline that includes PolynomialFeatures so we can export POWERS."
        )
    if not hasattr(poly, "powers_"):
        raise ValueError("PolynomialFeatures does not have powers_. Ensure it was fitted before saving.")
    POWERS = poly.powers_.astype(np.int32)           # (n_phi, n_feat)
    n_phi, n_feat = POWERS.shape

    # Optional StandardScaler on inputs (if present in the same pipeline)
    if (scaler is not None) and hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
        MU = scaler.mean_.astype(np.float32).reshape(-1, 1)          # (n_feat,1)
        INVSIGMA = (1.0 / scaler.scale_).astype(np.float32).reshape(-1, 1)
    else:
        MU = np.zeros((n_feat, 1), dtype=np.float32)
        INVSIGMA = np.ones((n_feat, 1), dtype=np.float32)

    print("Export shapes -> COEF:", COEF.shape, "INTERCEPT:", INTERCEPT.shape,
          "POWERS:", POWERS.shape, "MU:", MU.shape, "INVSIGMA:", INVSIGMA.shape)

    savemat(mat_path, {
        "COEF": COEF,               # float32 (n_out, n_phi)
        "INTERCEPT": INTERCEPT,     # float32 (n_out, 1)
        "POWERS": POWERS,           # int32   (n_phi, n_feat)
        "MU": MU,                   # float32 (n_feat, 1)
        "INVSIGMA": INVSIGMA,       # float32 (n_feat, 1)
    })
    print(f"Saved {mat_path}")

if __name__ == "__main__":
    export_to_mat(PKL, MAT)
