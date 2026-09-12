import numpy as np
import pandas as pd


def shale_volume_linear(gr: pd.Series, gr_clean: float | None = None,
                        gr_shale: float | None = None) -> pd.Series:
    """Linear GR shale-volume index, clipped to 0-1."""
    if gr_clean is None:
        gr_clean = float(gr.quantile(0.05))
    if gr_shale is None:
        gr_shale = float(gr.quantile(0.95))
    if gr_shale <= gr_clean:
        raise ValueError("GR shale endpoint must be greater than clean endpoint.")
    return ((gr - gr_clean) / (gr_shale - gr_clean)).clip(0, 1)


def density_porosity(rhob: pd.Series, matrix_density: float = 2.65,
                     fluid_density: float = 1.0) -> pd.Series:
    """Density porosity using the standard two-density relation."""
    if matrix_density <= fluid_density:
        raise ValueError("Matrix density must exceed fluid density.")
    return ((matrix_density - rhob) / (matrix_density - fluid_density)).clip(0, 1)


def add_basic_petrophysics(
    df: pd.DataFrame,
    gr_col: str = "GR",
    rhob_col: str = "RHOB",
) -> pd.DataFrame:
    """Add deterministic Vsh and density-porosity curves."""
    out = df.copy()
    if gr_col in out.columns:
        out["Vsh_linear"] = shale_volume_linear(out[gr_col])
    if rhob_col in out.columns:
        out["Phi_density"] = density_porosity(out[rhob_col])
    return out
