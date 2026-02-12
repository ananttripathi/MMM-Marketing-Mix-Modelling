"""
Marketing Mix Modelling (MMM) - Step-by-Step Wizard

Step 1: Brand, Model, Data, Channels, Date, Target, Segments
Step 2: Per-channel transform, curvature, adstock, Fit model
Step 3: ROI inputs - brand unit price, promotional unit cost per channel
Step 4: Results - ROI, mROI, channel & segment graphs
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
        if st.button("Next: Step 3 →", type="primary", key="step2_next"):
            st.session_state.step = 3
            st.rerun()

    st.divider()
    if st.button("← Back to Step 1", key="step2_back"):
        st.session_state.step = 1
        st.rerun()


def _safe_float(val, default=1.0):
    """Safely convert to float for Streamlit number_input."""
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# ============ STEP 3 ============
def render_step3():
    if st.session_state.df is None or not st.session_state.channel_cols or not st.session_state.get("fitted"):
        st.warning("Complete Step 1 and Step 2 (fit model) first.")
        st.session_state.step = 1
        st.rerun()
    st.header("Step 3: ROI Inputs")
    st.caption("Enter brand unit price and promotional unit cost per channel for ROI calculation")

    channel_cols = st.session_state.channel_cols

    col1, col2 = st.columns(2)
    with col1:
        roi_default = _safe_float(st.session_state.get("roi_unit_price"), 1.0)
        brand_unit_price = st.number_input(
            "Brand unit price ($ per unit of target)",
            min_value=0.01,
            value=roi_default,
            step=0.1,
            format="%.2f",
            help="Price per unit of your target variable (e.g. $ per sale). Use 1.0 if target is already revenue.",
            key="roi_unit_price",
        )
    with col2:
        st.info("Cost per $1 of spend: 1.0 = spend equals cost; 1.1 = 10% overhead")

    st.subheader("Promotional unit cost per channel")
    st.caption("Cost per $1 of spend (e.g. 1.0 = media spend only)")
    channel_costs = {}
    n_cols = min(len(channel_cols), 4)
    for i in range(0, len(channel_cols), n_cols):
        cost_cols = st.columns(n_cols)
        for j in range(n_cols):
            if i + j < len(channel_cols):
                ch = channel_cols[i + j]
                cost_key = f"roi_cost_{ch}"
                cost_default = _safe_float(st.session_state.get(cost_key), 1.0)
                with cost_cols[j]:
                    channel_costs[ch] = st.number_input(
                        ch,
                        min_value=0.01,
                        value=cost_default,
                        step=0.05,
                        format="%.2f",
                        key=cost_key,
                    )
    st.session_state.channel_costs = channel_costs

    st.divider()
    if st.button("Next: Step 4 → Results", type="primary", key="step3_next"):
        st.session_state.step = 4
        st.rerun()
    if st.button("← Back to Step 2", key="step3_back"):
        st.session_state.step = 2
        st.rerun()


# ============ STEP 4 ============
def render_step4():
    if st.session_state.df is None or not st.session_state.channel_cols or not st.session_state.get("fitted"):
        st.warning("Complete Steps 1–3 first.")
        st.session_state.step = 1
        st.rerun()
    st.header("Step 4: Results & Insights")
    st.caption(f"Brand: {st.session_state.brand_name} - ROI, mROI, channel & segment analysis")

    df = st.session_state.df
    channel_cols = st.session_state.channel_cols
    target_col = st.session_state.target_col
    date_col = st.session_state.date_col or "date"
    segment_cols = st.session_state.segment_cols
    pipeline = st.session_state.pipeline
    brand_unit_price = st.session_state.get("roi_unit_price", 1.0)
    channel_costs = st.session_state.get("channel_costs", {ch: 1.0 for ch in channel_cols})

    contributions = pipeline.get_channel_contributions(df)
    plot_df = df.copy()
    plot_df["predicted"] = pipeline.predict(df)
    for ch in channel_cols:
        plot_df[f"contrib_{ch}"] = contributions.get(ch, np.zeros(len(df)))

    # ROI computation
    roi_data = []
    for ch in channel_cols:
        total_contrib = float(np.sum(contributions.get(ch, np.zeros(len(df)))))
        total_spend = float(df[ch].sum())
        cost_per = channel_costs.get(ch, 1.0)
        total_cost = total_spend * cost_per
        revenue = total_contrib * brand_unit_price
        roi_pct = ((revenue - total_cost) / total_cost * 100) if total_cost > 0 else 0.0
        mroi = pipeline.get_marginal_roi(df, ch)
        mroi_pct = ((mroi * brand_unit_price) / cost_per - 1) * 100 if cost_per > 0 else 0.0
        roi_data.append({
            "Channel": ch,
            "Spend ($)": round(total_spend, 2),
            "Cost ($)": round(total_cost, 2),
            "Contribution": round(total_contrib, 2),
            "Revenue ($)": round(revenue, 2),
            "ROI (%)": round(roi_pct, 1),
            "mROI (%)": round(mroi_pct, 1),
        })
    roi_df = pd.DataFrame(roi_data)

    # --- Row 1: Actual vs Predicted, Model fit ---
    st.subheader("Model fit")
    r2 = pipeline.score(df)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("R² Score", f"{r2:.4f}")
    with col2:
        st.metric("Observations", len(df))
    with col3:
        st.metric("Channels", len(channel_cols))

    x_ax = plot_df[date_col] if date_col in plot_df.columns else plot_df.index
    fig_fit = go.Figure()
    fig_fit.add_trace(go.Scatter(x=x_ax, y=plot_df[target_col], name="Actual", line=dict(color="#1f77b4")))
    fig_fit.add_trace(go.Scatter(x=x_ax, y=plot_df["predicted"], name="Predicted", line=dict(color="#ff7f0e", dash="dash")))
    fig_fit.update_layout(title=f"{st.session_state.brand_name} - Actual vs Predicted")
    st.plotly_chart(fig_fit, use_container_width=True)

    # --- Row 2: Coefficient bar, ROI bar, mROI bar ---
    st.subheader("Channel coefficients & ROI")
    coef_df = pd.DataFrame([{"Variable": k, "Coefficient": v} for k, v in pipeline.get_coefficients().items() if k != "intercept"])
    c1, c2, c3 = st.columns(3)
    with c1:
        fig_coef = px.bar(coef_df, x="Variable", y="Coefficient", color="Coefficient", title="Coefficients")
        st.plotly_chart(fig_coef, use_container_width=True)
    with c2:
        fig_roi = px.bar(roi_df, x="Channel", y="ROI (%)", color="ROI (%)", color_continuous_scale="RdYlGn", title="ROI by Channel")
        st.plotly_chart(fig_roi, use_container_width=True)
    with c3:
        fig_mroi = px.bar(roi_df, x="Channel", y="mROI (%)", color="mROI (%)", color_continuous_scale="RdYlGn", title="Marginal ROI by Channel")
        st.plotly_chart(fig_mroi, use_container_width=True)

    st.dataframe(roi_df, use_container_width=True, hide_index=True)

    # --- ROI vs mROI comparison ---
    st.subheader("ROI vs Marginal ROI")
    roi_melt = roi_df.melt(id_vars=["Channel"], value_vars=["ROI (%)", "mROI (%)"], var_name="Metric", value_name="Value")
    fig_roi_mroi = px.bar(roi_melt, x="Channel", y="Value", color="Metric", barmode="group", title="ROI vs mROI by channel")
    st.plotly_chart(fig_roi_mroi, use_container_width=True)

    # --- Row 3: Spend over time, Contribution over time ---
    st.subheader("Spend & contribution over time")
    spend_cols = [c for c in channel_cols if c in plot_df.columns]
    if spend_cols:
        fig_spend = go.Figure()
        for ch in spend_cols:
            fig_spend.add_trace(go.Scatter(x=x_ax, y=plot_df[ch], name=ch, stackgroup="one"))
        fig_spend.update_layout(title="Spend by channel over time", barmode="stack")
        st.plotly_chart(fig_spend, use_container_width=True)

    contrib_cols = [f"contrib_{ch}" for ch in channel_cols if f"contrib_{ch}" in plot_df.columns]
    if contrib_cols:
        fig_contrib = go.Figure()
        for ch in channel_cols:
            if f"contrib_{ch}" in plot_df.columns:
                fig_contrib.add_trace(go.Scatter(x=x_ax, y=plot_df[f"contrib_{ch}"], name=ch, stackgroup="one"))
        fig_contrib.update_layout(title="Contribution by channel over time")
        st.plotly_chart(fig_contrib, use_container_width=True)

    # --- Row 4: ROI vs Spend scatter, Pie charts ---
    st.subheader("Spend & contribution mix")
    p1, p2, p3 = st.columns(3)
    with p1:
        fig_pie_spend = px.pie(roi_df, values="Spend ($)", names="Channel", title="Spend share by channel")
        st.plotly_chart(fig_pie_spend, use_container_width=True)
    with p2:
        fig_pie_rev = px.pie(roi_df, values="Revenue ($)", names="Channel", title="Revenue share by channel")
        st.plotly_chart(fig_pie_rev, use_container_width=True)
    with p3:
        fig_scatter = px.scatter(roi_df, x="Spend ($)", y="ROI (%)", color="Channel", size="Revenue ($)", title="ROI vs Spend")
        st.plotly_chart(fig_scatter, use_container_width=True)

    # --- Row 5: Contribution heatmap ---
    st.subheader("Contribution heatmap")
    contrib_matrix = np.column_stack([contributions.get(ch, np.zeros(len(df))) for ch in channel_cols])
    fig_heat = px.imshow(
        contrib_matrix.T,
        labels=dict(x="Period", y="Channel", color="Contribution"),
        x=[str(i) for i in range(len(df))],
        y=channel_cols,
        aspect="auto",
        color_continuous_scale="Blues",
    )
    fig_heat.update_layout(title="Channel contribution over time (heatmap)")
    st.plotly_chart(fig_heat, use_container_width=True)

    # --- Cumulative contribution ---
    st.subheader("Cumulative contribution by channel")
    cum_contrib = {ch: np.cumsum(contributions.get(ch, np.zeros(len(df)))) for ch in channel_cols}
    fig_cum = go.Figure()
    for ch in channel_cols:
        fig_cum.add_trace(go.Scatter(x=x_ax, y=cum_contrib[ch], name=ch, mode="lines"))
    fig_cum.update_layout(title="Cumulative contribution over time")
    st.plotly_chart(fig_cum, use_container_width=True)

    # --- Revenue per $ spend (efficiency) ---
    roi_df["Rev per $"] = np.where(roi_df["Spend ($)"] > 0, roi_df["Revenue ($)"] / roi_df["Spend ($)"], 0)
    fig_eff = px.bar(roi_df, x="Channel", y="Rev per $", color="Rev per $", title="Revenue per $1 spend (efficiency)")
    st.plotly_chart(fig_eff, use_container_width=True)

    # --- Row 6: Segment-level ROI (if segments exist) ---
    if segment_cols:
        st.subheader("Segment-level ROI")
        for seg_col in segment_cols:
            if seg_col not in df.columns:
                continue
            seg_vals = df[seg_col].dropna().unique()
            seg_roi = []
            for seg_val in seg_vals[:10]:
                mask = df[seg_col] == seg_val
                seg_df = df[mask]
                total_spend_seg = sum(seg_df[ch].sum() for ch in channel_cols)
                total_target_seg = seg_df[target_col].sum()
                revenue_seg = total_target_seg * brand_unit_price
                cost_seg = 0
                for ch in channel_cols:
                    cost_seg += seg_df[ch].sum() * channel_costs.get(ch, 1.0)
                roi_seg = ((revenue_seg - cost_seg) / cost_seg * 100) if cost_seg > 0 else 0.0
                seg_roi.append({"Segment": str(seg_val), "Spend ($)": round(total_spend_seg, 2), "Revenue ($)": round(revenue_seg, 2), "ROI (%)": round(roi_seg, 1)})
            seg_roi_df = pd.DataFrame(seg_roi)
            st.caption(f"By {seg_col}")
            st.dataframe(seg_roi_df, use_container_width=True, hide_index=True)
            fig_seg = px.bar(seg_roi_df, x="Segment", y="ROI (%)", color="ROI (%)", color_continuous_scale="RdYlGn")
            st.plotly_chart(fig_seg, use_container_width=True)

    st.divider()
    if st.button("← Back to Step 3", key="step4_back"):
        st.session_state.step = 3
        st.rerun()


# ============ MAIN ============
st.title("📊 Marketing Mix Modelling")
st.caption(f"Brand: {st.session_state.brand_name or '(not set)'}  |  Step {st.session_state.step} of 4")

step_indicator = st.columns(4)
with step_indicator[0]:
    st.markdown("**Step 1**" + (" ✓" if st.session_state.step > 1 else " ←"))
with step_indicator[1]:
    st.markdown("**Step 2**" + (" ✓" if st.session_state.step > 2 else (" ←" if st.session_state.step == 2 else "")))
with step_indicator[2]:
    st.markdown("**Step 3**" + (" ✓" if st.session_state.step > 3 else (" ←" if st.session_state.step == 3 else "")))
with step_indicator[3]:
    st.markdown("**Step 4**" + (" ←" if st.session_state.step == 4 else ""))

if st.session_state.step == 1:
    render_step1()
elif st.session_state.step == 2:
    render_step2()
elif st.session_state.step == 3:
    render_step3()
else:
    render_step4()
