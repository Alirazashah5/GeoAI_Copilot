# GeoAI Copilot

An open-source AI-assisted workspace for geologists, geoscientists, and petrophysicists.

GeoAI Copilot is designed to combine natural-language assistance with deterministic geoscience workflows: well-log QC, petrophysics, machine learning, geospatial analysis, lithofacies interpretation, and technical reporting.

> **Project status:** Initial MVP scaffold. Core deterministic modules are intentionally usable without an LLM or external API key.

## Vision

Turn repetitive geoscience data preparation and interpretation into reproducible, explainable workflows while keeping the geoscientist in control.

## Initial capabilities

- CSV well-log loading and validation
- Well-log quality-control summary
- Basic feature engineering
- Petrophysical calculations
- ML regression benchmarking:
  - Random Forest
  - Gradient Boosting
  - XGBoost (optional)
  - MLP/ANN
- Feature importance and evaluation metrics
- Natural-language copilot interface through Streamlit
- FastAPI REST endpoints
- Optional OpenAI-compatible LLM adapter
- Docker deployment
- Tests and CI
- Clear separation between deterministic geoscience calculations and AI-generated explanations

## Architecture

```text
                    ┌──────────────────────────┐
                    │      Geoscientist        │
                    └────────────┬─────────────┘
                                 │
                         Natural language
                                 │
                    ┌────────────▼─────────────┐
                    │      Streamlit UI        │
                    └────────────┬─────────────┘
                                 │ HTTP
                    ┌────────────▼─────────────┐
                    │       FastAPI API        │
                    └────────────┬─────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                  │
      ┌───────▼───────┐  ┌───────▼────────┐  ┌──────▼──────┐
      │ Data / QC     │  │ Petrophysics   │  │ ML Engine   │
      │ CSV / LAS*    │  │ calculations   │  │ RF/GBDT/MLP │
      └───────┬───────┘  └────────────────┘  └──────┬──────┘
              │                                      │
              └──────────────────┬───────────────────┘
                                 ▼
                    ┌──────────────────────────┐
                    │  AI Explanation Layer    │
                    │  optional LLM provider   │
                    └──────────────────────────┘

* LAS support is planned; CSV is supported in this initial scaffold.
```

## Quick start

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
Copy-Item .env.example .env
streamlit run app/streamlit_app.py
```

API:

```powershell
uvicorn geoai_copilot.api:app --reload --port 8000
```

Open:
- UI: `http://localhost:8501`
- API docs: `http://localhost:8000/docs`

### Docker

```bash
docker compose up --build
```

## Sample workflow

```python
from geoai_copilot.data import load_well_csv
from geoai_copilot.ml import benchmark_regressors

df = load_well_csv("data/sample/sample_well_logs.csv")
result = benchmark_regressors(
    df,
    target="Phi",
    features=["RHOB", "GR", "NPHI", "PEFZ"],
)
print(result.metrics)
```

## Data policy

Do not commit proprietary well logs, confidential reports, credentials, API keys, or licensed datasets. Put private data under a local ignored directory such as `data/private/`.

## Geoscience scope

The project is intended to grow toward:

1. Well-log QC and normalization
2. Petrophysical parameter calculation
3. Lithofacies classification
4. Tight-gas reservoir characterization
5. Well-to-well transfer learning
6. GIS and spatial analysis
7. Cuttings/XRD/thin-section knowledge integration
8. AI-assisted interpretation and report generation

The repository is deliberately designed so that AI explanations do not silently replace numerical calculations.

## Disclaimer

GeoAI Copilot is a research and decision-support tool. Geological, drilling, completion, reserve, and production decisions must be independently reviewed by qualified professionals.
