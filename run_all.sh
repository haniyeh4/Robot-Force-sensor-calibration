#!/usr/bin/env bash
# Reproduce every result and figure. Takes about 15-25 min on a laptop CPU (mostly MLP training).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
$PY scripts/01_quasi_static.py
$PY scripts/02_dynamic_poly.py
$PY scripts/03_dynamic_mlp.py
$PY scripts/04_correlation.py
$PY scripts/05_noise.py
$PY scripts/06_summary.py
