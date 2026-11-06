"""Model definitions, matching the original implementation."""
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def qs_poly(degree=3):
    """Quasi-static model: full polynomial in (roll, pitch, yaw), one OLS per output.

    Degree 3 -> 20 terms per output (incl. constant and all interaction terms).
    """
    return Pipeline([
        ("poly", PolynomialFeatures(degree)),
        ("regressor", MultiOutputRegressor(LinearRegression())),
    ])


def dyn_poly(degree=2, regressor=None, scale=True):
    """Dynamic polynomial model on the 9 standardized inputs.

    Degree 2 -> 54 features (9 linear + 9 squares + 36 interactions) + intercept,
    with separate coefficients for each of the 6 outputs.
    """
    steps = [("scaler", StandardScaler())] if scale else []
    steps += [
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("reg", regressor if regressor is not None else LinearRegression()),
    ]
    return Pipeline(steps)


def force_predictor(input_dim=9, hidden_dims=(64, 128, 64), output_dim=6, dropout=0.2):
    """MLP used for dynamic calibration (18,118 trainable parameters)."""
    import torch.nn as nn

    layers, dims = [], [input_dim, *hidden_dims]
    for a, b in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(a, b), nn.ReLU(), nn.BatchNorm1d(b), nn.Dropout(dropout)]
    layers.append(nn.Linear(dims[-1], output_dim))
    return nn.Sequential(*layers)
