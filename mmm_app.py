"""
Marketing Mix Modelling (MMM) Frontend

Streamlit app for MMM with:
- Dataset upload
- Adstock decay, saturation curvature
- Model selection: Linear, Ridge, Lasso, Bayesian, Hierarchical
- Constraints: positive coefficients, lag sum range
"""

import os
import sys

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mmm.config import MMMConfig, infer_target_column, infer_channel_columns
from mmm.pipeline import MMMPipeline

st.set_page_config(
    page_title="Marketing Mix Modelling",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Marketing Mix Modelling")
st.markdown(
    "Configure transforms, model type, and constraints to fit your MMM and attribute sales to marketing channels."
)

# Sidebar: Data & Params
with st.sidebar:
    st.header("📁 Data")
    use_sample = st.checkbox("Use sample dataset", value=True)
    if use_sample:
        sample_path = os.path.join(
            os.path.dirname(__file__),
            "data",
            "marketing_mix_weekly.csv",
        )
        if os.path.exists(sample_path):
            df = pd.read_csv(sample_path)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            st.success(f"Loaded: {len(df)} rows")
        else:
            st.error("Sample dataset not found. Run create_mmm_dataset.py first.")
            df = None
    else:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            st.success(f"Loaded: {len(df)} rows")
        else:
            df = None

    st.header("🔄 Transforms")
    decay = st.slider(
        "Adstock decay (carryover)",
        min_value=0.0,
        max_value=0.95,
        value=0.5,
        step=0.05,
        help="Higher = longer carryover effect from past spend",
    )
    max_lag = st.slider(
        "Adstock max lag (weeks)",
        min_value=1,
        max_value=8,
        value=4,
    )
    alpha = st.slider(
        "Saturation curvature (Hill alpha)",
        min_value=0.3,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help=">1 = S-curve, 1 = concave, <1 = convex",
    )
    half_sat = st.number_input(
        "Half-saturation (optional, 0 = auto)",
        min_value=0.0,
        value=0.0,
        step=10.0,
        help="Spend level at 50% saturation. 0 = auto from data",
    )

    st.header("🧮 Model")
    model_type = st.selectbox(
        "Model type",
        ["linear", "ridge", "lasso", "bayesian", "hierarchical"],
        format_func=lambda x: {
            "linear": "Linear",
            "ridge": "Ridge",
            "lasso": "Lasso",
            "bayesian": "Bayesian",
            "hierarchical": "Hierarchical",
        }[x],
    )
    ridge_alpha = 1.0
    lasso_alpha = 0.1
    if model_type == "ridge":
        ridge_alpha = st.number_input("Ridge alpha", min_value=0.01, value=1.0, step=0.1)
    if model_type == "lasso":
        lasso_alpha = st.number_input("Lasso alpha", min_value=0.01, value=0.1, step=0.01)

    bayesian_samples = 500
    bayesian_tune = 300
    if model_type in ["bayesian", "hierarchical"]:
        bayesian_samples = st.number_input(
            "Bayesian samples", min_value=100, value=500, step=100
        )
        bayesian_tune = st.number_input(
            "Bayesian tune", min_value=100, value=300, step=50
        )

    st.header("⚙️ Constraints")
    positive_constraints = st.checkbox(
        "Ensure positive coefficients",
        value=True,
        help="Channel effects must be >= 0",
    )
    use_lag_sum = st.checkbox(
        "Constrain lag sum (channel coefficients)",
        value=False,
    )
    if use_lag_sum:
        lag_lower = st.number_input(
            "Lag sum lower bound",
            min_value=0.0,
            value=0.8,
            step=0.1,
        )
        lag_upper = st.number_input(
            "Lag sum upper bound",
            min_value=0.0,
            value=1.0,
            step=0.1,
        )
    else:
        lag_lower = None
        lag_upper = None

# Column mapping - always visible, works with any dataset
if df is not None:
    st.subheader("📋 Column Mapping")
    st.caption("Map your dataset columns to target, marketing channels, and controls.")

    cols = df.columns.tolist()
    date_cols = [c for c in cols if "date" in c.lower() or "time" in c.lower() or "week" in c.lower()]
    inferred_target = infer_target_column(cols)
    target_idx = cols.index(inferred_target) if inferred_target in cols else 0

    date_col = None
    if any("date" in c.lower() or "time" in c.lower() or "week" in c.lower() for c in cols):
        date_candidates = [c for c in cols if "date" in c.lower() or "time" in c.lower() or "week" in c.lower()]
        date_col = date_candidates[0] if date_candidates else None

    col1, col2, col3 = st.columns(3)
    with col1:
        target_col = st.selectbox(
            "Target (sales/revenue)",
            cols,
            index=target_idx,
            help="The metric you want to attribute (e.g. sales, revenue, conversions)",
        )
    with col2:
        exclude_cols = [target_col] + (["date"] if "date" in df.columns else [])
        numeric_cols = [
            c for c in df.columns
            if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])
        ]
        default_channels = infer_channel_columns(df, target_col, ["date"])
        channel_cols = st.multiselect(
            "Channel columns (spend)",
            numeric_cols,
            default=default_channels,
            help="Marketing spend columns (TV, digital, etc.)",
        )
    with col3:
        control_options = [
            c for c in df.columns
            if c not in [target_col, "date"] and c not in channel_cols
        ]
        control_cols = st.multiselect(
            "Control columns",
            control_options,
            default=[],
            help="Covariates (seasonality, holidays, promotions, etc.)",
        )

# Main content
if df is not None and channel_cols:

    config = MMMConfig(
        target_col=target_col,
        channel_cols=channel_cols,
        control_cols=control_cols,
        adstock_decay=decay,
        adstock_max_lag=max_lag,
        saturation_alpha=alpha,
        saturation_half_sat=half_sat if half_sat > 0 else None,
        positive_constraints=positive_constraints,
        lag_sum_lower=lag_lower if use_lag_sum else None,
        lag_sum_upper=lag_upper if use_lag_sum else None,
        model_type=model_type,
        ridge_alpha=ridge_alpha if model_type == "ridge" else 1.0,
        lasso_alpha=lasso_alpha if model_type == "lasso" else 0.1,
        bayesian_samples=bayesian_samples if model_type in ["bayesian", "hierarchical"] else 1000,
        bayesian_tune=bayesian_tune if model_type in ["bayesian", "hierarchical"] else 500,
    )

    if st.button("🚀 Fit Model", type="primary"):
        with st.spinner("Fitting model..."):
            try:
                pipeline = MMMPipeline(config)
                pipeline.fit(df)
                st.session_state["pipeline"] = pipeline
                st.session_state["fitted"] = True
            except Exception as e:
                st.error(str(e))
                st.session_state["fitted"] = False

    if st.session_state.get("fitted"):
        pipeline = st.session_state["pipeline"]

        col1, col2, col3 = st.columns(3)
        with col1:
            r2 = pipeline.score(df)
            st.metric("R² Score", f"{r2:.4f}")
        with col2:
            pred = pipeline.predict(df)
            mape = np.mean(np.abs((df[target_col] - pred) / (df[target_col] + 1e-6))) * 100
            st.metric("MAPE (%)", f"{mape:.2f}")
        with col3:
            st.metric("Channels", len(channel_cols))

        st.subheader("Channel Attribution (Coefficients)")
        coef = pipeline.get_coefficients()
        coef_df = pd.DataFrame(
            [{"Variable": k, "Coefficient": v} for k, v in coef.items()]
        )
        st.dataframe(coef_df, use_container_width=True)

        fig_bar = px.bar(
            coef_df[coef_df["Variable"] != "intercept"],
            x="Variable",
            y="Coefficient",
            color="Coefficient",
            color_continuous_scale="Blues",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("Actual vs Predicted")
        plot_df = df.copy()
        plot_df["predicted"] = pipeline.predict(df)
        fig = go.Figure()
        x_col = date_col if date_col and date_col in plot_df.columns else ("date" if "date" in plot_df.columns else None)
        x_axis = plot_df[x_col] if x_col else plot_df.index
        fig.add_trace(go.Scatter(x=x_axis, y=plot_df[target_col], name="Actual", line=dict(color="#1f77b4")))
        fig.add_trace(go.Scatter(x=x_axis, y=plot_df["predicted"], name="Predicted", line=dict(color="#ff7f0e", dash="dash")))
        fig.update_layout(xaxis_title="Date", yaxis_title=target_col)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Residuals")
        plot_df["residual"] = plot_df[target_col] - plot_df["predicted"]
        fig_res = px.scatter(
            plot_df,
            x="predicted",
            y="residual",
            trendline="ols",
        )
        fig_res.add_hline(y=0, line_dash="dash")
        st.plotly_chart(fig_res, use_container_width=True)

else:
    if df is None:
        st.info("Upload a CSV or use the sample dataset to get started.")
    else:
        st.warning("Select at least one channel column.")
