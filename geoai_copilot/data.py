from pathlib import Path
import pandas as pd


def load_well_csv(path: str | Path) -> pd.DataFrame:
    """Load a well-log CSV and normalize column names without changing values."""
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def qc_summary(df: pd.DataFrame) -> dict:
    """Return deterministic data-quality statistics."""
    numeric = df.select_dtypes(include="number")
    missing = df.isna().sum()
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "numeric_columns": list(numeric.columns),
        "missing_values": {k: int(v) for k, v in missing.items() if int(v) > 0},
        "duplicate_rows": int(df.duplicated().sum()),
    }


def prepare_features(
    df: pd.DataFrame,
    features: list[str],
    target: str,
) -> tuple[pd.DataFrame, pd.Series]:
    missing = [c for c in [*features, target] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work = df[[*features, target]].copy()
    work = work.apply(pd.to_numeric, errors="coerce").dropna()
    return work[features], work[target]
