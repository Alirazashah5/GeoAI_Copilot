# GeoAI Copilot — Initial Product Requirements Document

## Problem

Geoscientists routinely move between spreadsheets, well-log software, Python notebooks,
GIS platforms and technical reports. The workflow is powerful but fragmented and often
requires repetitive data preparation.

## Product goal

Provide one workspace where a geoscientist can load data, perform reproducible
calculations, benchmark ML models and ask an AI assistant to explain the results.

## Users

- Petroleum geologists
- Exploration geologists
- Petrophysicists
- Reservoir engineers
- Geoscience students/researchers
- GIS/geospatial analysts

## MVP user journey

1. Upload a CSV well-log dataset.
2. Inspect columns and QC statistics.
3. Select target and input curves.
4. Run baseline ML models.
5. Compare MAE/RMSE/R².
6. Export metrics.
7. Ask the optional AI layer for a plain-language interpretation.

## Non-goals for MVP

- Automated reserves certification
- Automated drilling decisions
- Unsupervised publication of geological conclusions
- Replacement of commercial interpretation suites
- Automatic use of confidential data in external AI services

## Success criteria

- Installation works from a clean Python environment.
- Sample dataset runs without credentials.
- Numerical functions have automated tests.
- ML results are reproducible.
- UI exposes assumptions clearly.
- Private data are excluded from Git.
