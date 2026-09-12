# Architecture

## Design principle

GeoAI Copilot separates three layers:

### 1. Deterministic geoscience layer
Contains data loading, QC, equations and ML evaluation. It should work without an LLM.

### 2. Application layer
FastAPI exposes programmatic endpoints. Streamlit provides the scientist-facing UI.

### 3. AI explanation layer
An optional LLM can explain outputs, draft reports, suggest workflows and answer domain
questions. It is not the numerical source of truth.

## Planned modules

```text
geoai_copilot/
├── data.py
├── petrophysics.py
├── ml.py
├── llm.py
└── api.py

future:
├── las.py
├── lithofacies.py
├── cuttings.py
├── xrd.py
├── gis.py
├── reports.py
└── agents/
```

## Future validation

For reservoir characterization, random row-level train/test splits can leak information
between adjacent depths. A future release should support:
- leave-one-well-out validation
- grouped cross-validation by WELL
- blind-well testing
- depth-aware QC
- formation/zone-aware evaluation
