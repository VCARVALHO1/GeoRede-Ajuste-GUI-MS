# GeoRede Ajuste

Python application for three-dimensional coordinate registration, network adjustment and residual diagnostics.

## Installation and execution

Use Python 3.10 or later. From this folder:

```bash
python -m venv .venv
```

Activate the environment on Windows with `.venv\Scripts\activate`, then install and launch:

```bash
python -m pip install -e .
python run_gui.py
```

Select your station-data folder in the interface and choose the processing settings and output location. Survey data and generated results are not included in this source package.

## Structure

- `run_gui.py`: desktop application entry point.
- `src/freesurvey/`: interface, registration methods and supporting modules.
- `analyze_methods_data.py` and `outlier_statistics.py`: processing and diagnostics used by the interface; required runtime files.
- `pyproject.toml` and `requirements.txt`: package metadata and dependencies.

Keep the folder structure intact and use an editable installation from this source checkout: the interface invokes the processing scripts at the repository root.

## Methodological scope

This package is a snapshot of the supplied working code. Processing settings, retained observations and convergence status must be checked for each execution. Covariance and statistical diagnostics depend on the adopted stochastic model. This snapshot has not been certified as numerically identical to the published article or dissertation execution. See the source for the actual implementation and parameters.

## Related publication

Ferreira, I. O.; Carvalho, V. A.; Ayres Neto, A. (2026). Network-Adjusted Three-Dimensional Coordinate Transformation with Integrated Quality Control for Dimensional and Hydrographic Survey Applications. https://doi.org/10.18540/jcecvl12iss1pp24874