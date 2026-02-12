"""
Marketing Mix Modelling (MMM) - Step-by-Step Wizard

Step 1: Brand, Model, Data, Channels, Date, Target, Segments
Step 2: Per-channel transform, curvature, adstock, Fit model
"""

import os
import sys

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mmm.config import MMMConfig, infer_target_column, infer_channel_columns

st.set_page_config(
    page_title="MMM - Marketing Mix Modelling",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Initialize session state
if "step" not in st.session_state:
    st.session_state.step = 1
if "df" not in st.session_state:
    st.session_state.df = None
if "brand_name" not in st.session_state:
    st.session_state.brand_name = ""
if "channel_cols" not in st.session_state:
    st.session_state.channel_cols = []
if "date_col" not in st.session_state:
    st.session_state.date_col = None
if "target_col" not in st.session_state:
    st.session_state.target_col = None
if "segment_cols" not in st.session_state:
    st.session_state.segment_cols = []
if "control_cols" not in st.session_state:
    st.session_state.control_cols = []
if "model_type" not in st.session_state:
    st.session_state.model_type = "linear"


def load_sample_data():
    path = os.path.join(os.path.dirname(__file__), "data", "marketing_mix_weekly.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df
    return None


# ============ STEP 1 ============
def render_step1():
    st.header("Step 1: Setup & Column Mapping")
    st.caption("Brand, model type, data source, and column mapping")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.session_state.brand_name = st.text_input(
            "Brand name",
            value=st.session_state.brand_name or "My Brand",
            placeholder="Enter brand name",
        )
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
            key="step1_model",
        )
        st.session_state.model_type = model_type

        st.subheader("Data source")
        data_source = st.radio(
            "Source",
            ["Sample dataset", "Upload CSV", "Generate dataset"],
            key="step1_source",
        )

        if data_source == "Sample dataset":
            df = load_sample_data()
            if df is not None:
                st.session_state.df = df
                st.success(f"Loaded: {len(df)} rows")
            else:
                st.error("Sample not found. Use Generate dataset.")
                st.session_state.df = None

        elif data_source == "Upload CSV":
            uploaded = st.file_uploader("Upload CSV", type=["csv"], key="step1_upload")
            if uploaded:
                df = pd.read_csv(uploaded)
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                st.session_state.df = df
                st.success(f"Loaded: {len(df)} rows")
            else:
                st.session_state.df = None

        else:  # Generate
            from create_mmm_dataset import generate_mmm_dataset, FREQ_MAP

            gen_start = st.text_input("Start date", "2020-01-01", key="gen_start")
            gen_end = st.text_input("End date", "2023-12-31", key="gen_end")
            gen_freq = st.selectbox("Frequency", ["daily", "weekly", "monthly", "yearly"], index=1, key="gen_freq")
            gen_channels = st.text_input(
                "Channel names (comma-separated)",
                "tv_spend, digital_spend, radio_spend, print_spend, social_spend",
                key="gen_ch",
            )
            if st.button("Generate", key="gen_btn"):
                ch_list = [c.strip() for c in gen_channels.split(",") if c.strip()] or None
                try:
                    df = generate_mmm_dataset(
                        start_date=gen_start, end_date=gen_end,
                        freq=FREQ_MAP[gen_freq], channel_names=ch_list, seed=42,
                    )
                    df["date"] = pd.to_datetime(df["date"])
                    st.session_state.df = df
                    st.success(f"Generated: {len(df)} rows")
                except Exception as e:
                    st.error(str(e))

    with col2:
        df = st.session_state.df
        if df is not None:
            st.subheader("Column mapping")
            st.caption("List all columns below. Select channels by moving them to the right.")

            cols = df.columns.tolist()
            inferred_target = infer_target_column(cols)
            target_idx = cols.index(inferred_target) if inferred_target in cols else 0

            # Date column
            date_candidates = [c for c in cols if "date" in c.lower() or "time" in c.lower() or "week" in c.lower()]
            date_idx = cols.index(date_candidates[0]) if date_candidates else 0
            date_col = st.selectbox(
                "Date column",
                cols,
                index=date_idx,
                key="step1_date",
            )
            st.session_state.date_col = date_col

            # Target column
            target_col = st.selectbox(
                "Target column (sales/revenue)",
                cols,
                index=target_idx,
                key="step1_target",
            )
            st.session_state.target_col = target_col

            # Channel selector - dual list style
            exclude = [date_col, target_col]
            available = [c for c in cols if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
            default_ch = infer_channel_columns(df, target_col, ["date"])
            st.caption("Select columns for marketing channels. In multiselect: click to add, order = priority.")
            channel_cols = st.multiselect(
                "Marketing channels",
                available,
                default=default_ch,
                key="step1_channels",
                help="Pick columns to use as marketing spend channels (e.g. TV, Digital, Radio)",
            )
            st.session_state.channel_cols = channel_cols

            # Segment columns
            segment_options = [c for c in cols if c not in [date_col, target_col] and c not in channel_cols]
            segment_cols = st.multiselect(
                "Segment columns (optional - for segment-level modelling)",
                segment_options,
                default=[],
                key="step1_segments",
                help="e.g. Region, Product category - for modelling by segment",
            )
            st.session_state.segment_cols = segment_cols

            # Control columns (remaining)
            used = [date_col, target_col] + channel_cols + segment_cols
            control_options = [c for c in cols if c not in used]
            control_cols = st.multiselect(
                "Control columns (optional)",
                control_options,
                default=[c for c in ["week_of_year", "month", "holiday_period", "promotion"] if c in control_options],
                key="step1_controls",
            )
            st.session_state.control_cols = control_cols

            st.info(f"**Summary:** {len(channel_cols)} channels, target=`{target_col}`")

        else:
            st.info("Load or generate data first.")

    if st.session_state.df is not None and st.session_state.channel_cols:
        st.divider()
        if st.button("Next: Step 2 →", type="primary", key="step1_next"):
            st.session_state.step = 2
            st.rerun()


# ============ STEP 2 ============
def render_step2():
    if st.session_state.df is None or not st.session_state.channel_cols:
        st.warning("Missing data or channels. Going back to Step 1.")
        st.session_state.step = 1
        st.rerun()
    st.header("Step 2: Transform & Model Parameters")
    st.caption("Per-channel transformation, curvature, adstock, and fit")

    df = st.session_state.df
    channel_cols = st.session_state.channel_cols
    target_col = st.session_state.target_col
    date_col = st.session_state.date_col or "date"

    TRANSFORM_LABELS = {
        "hill": "Hill",
        "negative_exponential": "Negative exponential",
        "log": "Log",
        "linear": "Linear",
        "power": "Power",
    }

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Per-channel transformation")
        st.caption("Select transform type for each channel (all follow diminishing returns)")
        channel_transform_types = {}
        for ch in channel_cols:
            channel_transform_types[ch] = st.selectbox(
                f"{ch}",
                list(TRANSFORM_LABELS.keys()),
                index=1,  # default: negative_exponential
                format_func=lambda x: TRANSFORM_LABELS[x],
                key=f"transform_{ch}",
            )
        st.session_state.channel_transform_types = channel_transform_types

        st.subheader("Curvature & Adstock")
        alpha = st.slider(
            "Curvature value (α)",
            min_value=0.3,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="Shape for Hill/Power transforms",
            key="step2_alpha",
        )
        decay = st.slider(
            "Adstock decay coefficient",
            min_value=0.0,
            max_value=0.95,
            value=0.5,
            step=0.05,
            help="Higher = longer carryover effect",
            key="step2_decay",
        )
        max_lag = st.number_input(
            "Number of adstock periods (months for monthly, weeks for weekly)",
            min_value=1,
            max_value=24,
            value=4,
            step=1,
            key="step2_maxlag",
        )
        half_sat = st.number_input(
            "Half-saturation (0 = auto)",
            min_value=0.0,
            value=0.0,
            step=10.0,
            key="step2_halfsat",
        )

    with col2:
        st.subheader("Model fit")
        model_type = st.selectbox(
            "Model type",
            ["linear", "ridge", "lasso", "bayesian", "hierarchical"],
            index=["linear", "ridge", "lasso", "bayesian", "hierarchical"].index(st.session_state.get("model_type", "linear")),
            format_func=lambda x: {"linear": "Linear", "ridge": "Ridge", "lasso": "Lasso", "bayesian": "Bayesian", "hierarchical": "Hierarchical"}[x],
            key="step2_model",
        )
        ridge_alpha = 1.0
        lasso_alpha = 0.1
        bayesian_samples = 500
        bayesian_tune = 300
        if model_type == "ridge":
            ridge_alpha = st.number_input("Ridge alpha", 0.01, 10.0, 1.0, 0.1, key="ridge_a")
        elif model_type == "lasso":
            lasso_alpha = st.number_input("Lasso alpha", 0.01, 10.0, 0.1, 0.01, key="lasso_a")
        elif model_type in ["bayesian", "hierarchical"]:
            bayesian_samples = st.number_input("Samples", 100, 2000, 500, 100, key="bay_samp")
            bayesian_tune = st.number_input("Tune", 100, 1000, 300, 50, key="bay_tune")

        config = MMMConfig(
            target_col=target_col,
            channel_cols=channel_cols,
            control_cols=st.session_state.control_cols + st.session_state.segment_cols,
            date_col=date_col,
            adstock_decay=decay,
            adstock_max_lag=max_lag,
            saturation_alpha=alpha,
            saturation_half_sat=half_sat if half_sat > 0 else None,
            saturation_transform_type="negative_exponential",
            channel_transform_types=channel_transform_types,
            model_type=model_type,
            ridge_alpha=ridge_alpha,
            lasso_alpha=lasso_alpha,
            bayesian_samples=bayesian_samples,
            bayesian_tune=bayesian_tune,
        )

        if st.button("Fit model", type="primary", key="step2_fit"):
            from mmm.pipeline import MMMPipeline

            with st.spinner("Fitting..."):
                try:
                    pipeline = MMMPipeline(config)
                    pipeline.fit(df)
                    st.session_state.pipeline = pipeline
                    st.session_state.fitted = True
                except Exception as e:
                    st.error(str(e))
                    st.session_state.fitted = False

        if st.session_state.get("fitted"):
            pipeline = st.session_state.pipeline
            r2 = pipeline.score(df)
            st.metric("R² Score", f"{r2:.4f}")
            coef = pipeline.get_coefficients()
            st.dataframe(pd.DataFrame([{"Variable": k, "Coefficient": v} for k, v in coef.items()]))

    if st.session_state.get("fitted"):
        st.divider()
        st.subheader("Results")
        pipeline = st.session_state.pipeline
        plot_df = df.copy()
        plot_df["predicted"] = pipeline.predict(df)
        x_ax = plot_df[date_col] if date_col in plot_df.columns else plot_df.index
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_ax, y=plot_df[target_col], name="Actual", line=dict(color="#1f77b4")))
        fig.add_trace(go.Scatter(x=x_ax, y=plot_df["predicted"], name="Predicted", line=dict(color="#ff7f0e", dash="dash")))
        fig.update_layout(title=f"{st.session_state.brand_name} - Actual vs Predicted")
        st.plotly_chart(fig, use_container_width=True)

        coef_df = pd.DataFrame(
            [{"Variable": k, "Coefficient": v} for k, v in pipeline.get_coefficients().items()]
        )
        fig_bar = px.bar(coef_df[coef_df["Variable"] != "intercept"], x="Variable", y="Coefficient", color="Coefficient")
        st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()
    if st.button("← Back to Step 1", key="step2_back"):
        st.session_state.step = 1
        st.rerun()


# ============ MAIN ============
st.title("📊 Marketing Mix Modelling")
st.caption(f"Brand: {st.session_state.brand_name or '(not set)'}  |  Step {st.session_state.step} of 2")

step_indicator = st.columns(2)
with step_indicator[0]:
    st.markdown("**Step 1**" + (" ✓" if st.session_state.step > 1 else " ←"))
with step_indicator[1]:
    st.markdown("**Step 2**" + (" ←" if st.session_state.step == 2 else ""))

if st.session_state.step == 1:
    render_step1()
else:
    render_step2()
