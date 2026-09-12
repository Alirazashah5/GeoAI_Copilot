# GeoAI Copilot development instructions

## Scope
This repository is a geoscience decision-support application. Deterministic numerical
calculations are the source of truth. LLM output must never silently alter numerical
results.

## Stack
- Python 3.11+
- FastAPI
- Streamlit
- pandas / NumPy
- scikit-learn
- optional XGBoost
- pytest / Ruff

## Geoscience rules
- Preserve original uploaded data.
- Make preprocessing explicit and reproducible.
- Document units and assumptions for petrophysical equations.
- Do not invent geological values when data are missing.
- Separate data QC, calculations, ML, visualization and AI explanation.
- Never commit proprietary well data or credentials.

## ML rules
- Record random seeds.
- Report MAE, RMSE and R² together.
- Keep train/test methodology explicit.
- Do not present model output as a geological conclusion without uncertainty/context.
- Future well-based validation should be preferred over random row splitting when the
  objective is assessing generalization to unseen wells.

## Code quality
- Add tests for new deterministic functions.
- Run `ruff check .` and `pytest` before committing.
