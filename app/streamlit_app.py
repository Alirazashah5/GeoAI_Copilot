import os
import tempfile

import streamlit as st
import pandas as pd
import lasio
import plotly.graph_objects as go


def load_uploaded_file(uploaded_file):
    """Load CSV or LAS well-log file into a pandas DataFrame."""

    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    if filename.endswith(".las"):
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".las"
        ) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name

        try:
            las = lasio.read(temp_path)
            df = las.df().reset_index()

            if "DEPT" in df.columns:
                df = df.rename(columns={"DEPT": "DEPTH"})

            return df

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    raise ValueError(
        "Unsupported file format. Please upload a CSV or LAS file."
    )


st.set_page_config(
    page_title="GeoAI Copilot",
    layout="wide",
)
from plotly.subplots import make_subplots

from geoai_copilot.data import load_well_csv, qc_summary
from geoai_copilot.ml import benchmark_regressors
from geoai_copilot.petrophysics import add_basic_petrophysics


st.set_page_config(
    page_title="GeoAI Copilot",
    page_icon="🌍",
    layout="wide",
)

st.title("🌍 GeoAI Copilot")
st.caption("AI-assisted geoscience, well-log and petrophysics workspace")


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.header("Workflow")

    uploaded = st.file_uploader(
        "Upload Well-Log Data",
        type=["csv", "las"],
        help="Upload a CSV or LAS well-log file.",
)


# ---------------------------------------------------------
# NO FILE UPLOADED
# ---------------------------------------------------------
if uploaded is None:
    st.info(
        "Upload a CSV well-log file to begin. "
        "A sample dataset is included in data/sample/."
    )

    st.markdown(
        """
### What this MVP can do

- QC your tabular well-log data
- Visualize well-log curves
- Calculate basic Vsh and density porosity
- Benchmark Random Forest, GBDT, MLP/ANN and optional XGBoost
- Export model metrics as CSV

**Recommended columns:**

`DEPTH`, `GR`, `RHOB`, `NPHI`, `PEFZ`, and a target such as `Phi`.
"""
    )

    st.stop()


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
df = load_uploaded_file(uploaded)


# ---------------------------------------------------------
# 1. DATA PREVIEW
# ---------------------------------------------------------
st.subheader("1. Data Preview")

st.dataframe(
    df.head(20),
    width="stretch",
)


# ---------------------------------------------------------
# 2. WELL LOG VISUALIZATION
# ---------------------------------------------------------
st.subheader("2. Well Log Visualization")

if "DEPTH" not in df.columns:
    st.warning(
        "DEPTH column was not found. A well-log plot requires a DEPTH column."
    )
else:

    # Convert depth to numeric
    plot_df = df.copy()
    plot_df["DEPTH"] = pd.to_numeric(
        plot_df["DEPTH"],
        errors="coerce",
    )

    plot_df = plot_df.dropna(subset=["DEPTH"])

    # -----------------------------------------------------
    # Well selection
    # -----------------------------------------------------
    if "WELL" in plot_df.columns:

        wells = (
            plot_df["WELL"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        if len(wells) > 0:

            selected_well = st.selectbox(
                "Select Well",
                wells,
            )

            plot_df = plot_df[
                plot_df["WELL"].astype(str) == selected_well
            ].copy()

    # Sort by depth
    plot_df = plot_df.sort_values("DEPTH")

    # -----------------------------------------------------
    # Depth range
    # -----------------------------------------------------
    min_depth = float(plot_df["DEPTH"].min())
    max_depth = float(plot_df["DEPTH"].max())

    depth_range = st.slider(
        "Depth Range",
        min_value=min_depth,
        max_value=max_depth,
        value=(min_depth, max_depth),
    )

    plot_df = plot_df[
        (plot_df["DEPTH"] >= depth_range[0])
        & (plot_df["DEPTH"] <= depth_range[1])
    ].copy()

    # -----------------------------------------------------
    # Curve availability
    # -----------------------------------------------------
    curve_groups = {
        "Gamma Ray": ["GR", "CGR", "SGR"],
        "Density / Porosity": ["RHOB", "NPHI", "DRHO"],
        "Resistivity": ["HLLD", "HLLS", "LLD", "LLS", "RXOZ"],
        "Sonic": ["DT", "LSDT"],
        "Other": ["PEFZ", "PEF", "SP", "CALI", "HCAL"],
        "Petrophysics": ["Vsh", "Phi", "Phi_d", "Sw", "Rw"],
    }

    available_groups = {}

    for group_name, curves in curve_groups.items():

        available = [
            curve for curve in curves
            if curve in plot_df.columns
        ]

        if available:
            available_groups[group_name] = available

    if not available_groups:

        st.warning(
            "No recognized well-log curves were found."
        )

    else:

        # -------------------------------------------------
        # Curve selection
        # -------------------------------------------------
        st.markdown("#### Select Curves")

        selected_curves = []

        for group_name, curves in available_groups.items():

            selected = st.multiselect(
                group_name,
                curves,
                default=curves,
                key=f"curve_selection_{group_name}",
            )

            selected_curves.extend(selected)

        if selected_curves:

            # ---------------------------------------------
            # Number of tracks
            # ---------------------------------------------
            track_definitions = []

            for group_name, curves in available_groups.items():

                selected_in_group = [
                    c for c in selected_curves
                    if c in curves
                ]

                if selected_in_group:
                    track_definitions.append(
                        (group_name, selected_in_group)
                    )

            n_tracks = len(track_definitions)

            # ---------------------------------------------
            # Create Plotly figure
            # ---------------------------------------------
            fig = make_subplots(
                rows=1,
                cols=n_tracks,
                shared_yaxes=True,
                horizontal_spacing=0.025,
                subplot_titles=[
                    group_name
                    for group_name, _ in track_definitions
                ],
            )

            # ---------------------------------------------
            # Add curves
            # ---------------------------------------------
            for track_number, (
                group_name,
                curves,
            ) in enumerate(track_definitions, start=1):

                for curve in curves:

                    values = pd.to_numeric(
                        plot_df[curve],
                        errors="coerce",
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=values,
                            y=plot_df["DEPTH"],
                            mode="lines",
                            name=curve,
                            connectgaps=False,
                            hovertemplate=(
                                f"{curve}: %{{x}}"
                                "<br>Depth: %{y}"
                                "<extra></extra>"
                            ),
                        ),
                        row=1,
                        col=track_number,
                    )

                # -----------------------------------------
                # Track labels
                # -----------------------------------------
                fig.update_xaxes(
                    title_text=group_name,
                    row=1,
                    col=track_number,
                    showgrid=True,
                    zeroline=False,
                )

            # ---------------------------------------------
            # Depth axis
            # ---------------------------------------------
            fig.update_yaxes(
                title_text="Depth",
                autorange="reversed",
                showgrid=True,
            )

            # ---------------------------------------------
            # Layout
            # ---------------------------------------------
            fig.update_layout(
                height=900,
                hovermode="closest",
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="left",
                    x=0,
                ),
                margin=dict(
                    l=60,
                    r=30,
                    t=100,
                    b=50,
                ),
            )

            st.plotly_chart(
                fig,
                width="stretch",
                config={
                    "scrollZoom": True,
                    "displaylogo": False,
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "GeoAI_Copilot_Well_Log",
                        "height": 1600,
                        "width": 2200,
                        "scale": 2,
                    },
                },
            )

            st.success(
                f"Showing {len(selected_curves)} curves "
                f"from {depth_range[0]:.2f} to "
                f"{depth_range[1]:.2f} depth."
            )

        else:
            st.info("Select at least one curve to display.")


# ---------------------------------------------------------
# 3. QC SUMMARY
# ---------------------------------------------------------
st.subheader("3. QC Summary")

qc = qc_summary(df)

c1, c2, c3 = st.columns(3)

c1.metric(
    "Rows",
    qc["rows"],
)

c2.metric(
    "Columns",
    qc["columns"],
)

c3.metric(
    "Duplicate rows",
    qc["duplicate_rows"],
)

if qc["missing_values"]:

    st.warning(
        f"Missing values: {qc['missing_values']}"
    )

else:

    st.success(
        "No missing values detected."
    )


# ---------------------------------------------------------
# 4. BASIC PETROPHYSICS
# ---------------------------------------------------------
st.subheader("4. Basic Petrophysics")

if "GR" in df.columns or "RHOB" in df.columns:

    pet = add_basic_petrophysics(df)

    st.dataframe(
        pet.head(20),
        width="stretch",
    )

else:

    st.info(
        "GR and/or RHOB were not found, so the "
        "basic petrophysics step was skipped."
    )


# ---------------------------------------------------------
# 5. ML BENCHMARK
# ---------------------------------------------------------
st.subheader("5. ML Benchmark")

numeric = df.select_dtypes(
    include="number"
).columns.tolist()

target_default = (
    "Phi"
    if "Phi" in df.columns
    else (numeric[-1] if numeric else "")
)

feature_defaults = [
    c
    for c in ["RHOB", "GR", "NPHI", "PEFZ"]
    if c in df.columns
]

if numeric:

    target = st.selectbox(
        "Target",
        numeric,
        index=(
            numeric.index(target_default)
            if target_default in numeric
            else 0
        ),
    )

    features = st.multiselect(
        "Features",
        [c for c in numeric if c != target],
        default=[
            c for c in feature_defaults
            if c != target
        ],
    )

    if st.button(
        "Run benchmark",
        type="primary",
    ):

        if not features:

            st.error(
                "Select at least one feature."
            )

        else:

            with st.spinner(
                "Training models..."
            ):

                result = benchmark_regressors(
                    df,
                    target=target,
                    features=features,
                )

            st.dataframe(
                result.metrics,
                width="stretch",
            )

            csv = (
                result.metrics
                .to_csv(index=False)
                .encode("utf-8")
            )

            st.download_button(
                "Download metrics CSV",
                data=csv,
                file_name="geoai_model_metrics.csv",
                mime="text/csv",
            )

else:

    st.info(
        "No numeric columns are available for ML benchmarking."
    )