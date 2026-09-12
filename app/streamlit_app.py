import io
import streamlit as st
import pandas as pd

from geoai_copilot.data import load_well_csv, qc_summary
from geoai_copilot.ml import benchmark_regressors
from geoai_copilot.petrophysics import add_basic_petrophysics

st.set_page_config(page_title="GeoAI Copilot", page_icon="🌍", layout="wide")

st.title("🌍 GeoAI Copilot")
st.caption("AI-assisted geoscience, well-log and petrophysics workspace")

with st.sidebar:
    st.header("Workflow")
    uploaded = st.file_uploader("Upload well-log CSV", type=["csv"])

if uploaded is None:
    st.info("Upload a CSV well-log file to begin. A sample dataset is included in data/sample/.")
    st.markdown(
        """
### What this MVP can do
- QC your tabular well-log data
- Calculate basic Vsh and density porosity
- Benchmark Random Forest, GBDT, MLP/ANN and optional XGBoost
- Export model metrics as CSV

**Recommended columns:** `DEPTH`, `GR`, `RHOB`, `NPHI`, `PEFZ`, and a target such as `Phi`.
"""
    )
    st.stop()

df = pd.read_csv(uploaded)
st.subheader("1. Data preview")
st.dataframe(df.head(20), use_container_width=True)

st.subheader("2. QC summary")
qc = qc_summary(df)
c1, c2, c3 = st.columns(3)
c1.metric("Rows", qc["rows"])
c2.metric("Columns", qc["columns"])
c3.metric("Duplicate rows", qc["duplicate_rows"])
if qc["missing_values"]:
    st.warning(f"Missing values: {qc['missing_values']}")
else:
    st.success("No missing values detected.")

st.subheader("3. Basic petrophysics")
if "GR" in df.columns or "RHOB" in df.columns:
    pet = add_basic_petrophysics(df)
    st.dataframe(pet.head(20), use_container_width=True)
else:
    st.info("GR and/or RHOB were not found, so the basic petrophysics step was skipped.")

st.subheader("4. ML benchmark")
numeric = df.select_dtypes(include="number").columns.tolist()
target_default = "Phi" if "Phi" in df.columns else (numeric[-1] if numeric else "")
feature_defaults = [c for c in ["RHOB", "GR", "NPHI", "PEFZ"] if c in df.columns]

target = st.selectbox("Target", numeric, index=numeric.index(target_default) if target_default in numeric else 0)
features = st.multiselect("Features", [c for c in numeric if c != target], default=feature_defaults)

if st.button("Run benchmark", type="primary"):
    if not features:
        st.error("Select at least one feature.")
    else:
        with st.spinner("Training models..."):
            result = benchmark_regressors(df, target=target, features=features)
        st.dataframe(result.metrics, use_container_width=True)
        csv = result.metrics.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download metrics CSV",
            data=csv,
            file_name="geoai_model_metrics.csv",
            mime="text/csv",
        )
