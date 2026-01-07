from __future__ import annotations

import math
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from abtest import (
    export_to_csv,
    generate_experiment_data,
    generate_recommendation,
    simulate_power_analysis,
    z_test_proportions,
)
from abtest.storage import (
    get_experiment_data,
    get_experiments,
    save_experiment,
    save_experiment_data,
)

st.set_page_config(page_title="A/B Test Simulator (Python)", layout="wide")

st.title("A/B Test Simulator")
st.caption("Python rewrite of the original TypeScript/React project — built with Streamlit.")


def _compute_segment_table(df: pd.DataFrame) -> pd.DataFrame:
    if "segment" not in df.columns or df["segment"].isna().all():
        return pd.DataFrame()

    out_rows = []
    for seg in sorted(df["segment"].dropna().unique()):
        seg_df = df[df["segment"] == seg]
        a = seg_df[seg_df["group"] == "A"]
        b = seg_df[seg_df["group"] == "B"]

        a_n = len(a)
        b_n = len(b)
        a_conv = int(a["converted"].sum())
        b_conv = int(b["converted"].sum())
        a_rate = a_conv / a_n if a_n else 0.0
        b_rate = b_conv / b_n if b_n else 0.0
        lift = ((b_rate - a_rate) / a_rate * 100) if a_rate else 0.0

        out_rows.append(
            {
                "segment": str(seg),
                "control_rate": a_rate,
                "treatment_rate": b_rate,
                "relative_lift_%": lift,
                "sample_size": a_n + b_n,
                "control_conversions": f"{a_conv}/{a_n}",
                "treatment_conversions": f"{b_conv}/{b_n}",
            }
        )

    out = pd.DataFrame(out_rows)
    return out.sort_values("relative_lift_%", ascending=False).reset_index(drop=True)


with st.sidebar:
    st.header("Experiment setup")

    mode = st.radio("Mode", ["Run new simulation", "View history"], index=0)

    if mode == "Run new simulation":
        name = st.text_input("Experiment name", value=f"Experiment {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        baseline = st.number_input("Baseline conversion (A)", min_value=0.0, max_value=1.0, value=0.10, step=0.01, format="%.4f")
        uplift = st.number_input("Expected uplift (absolute)", min_value=-1.0, max_value=1.0, value=0.01, step=0.005, format="%.4f")
        sample_size = st.number_input("Sample size per group", min_value=50, max_value=5_000_000, value=2000, step=50)
        confidence_level = st.slider("Confidence level", min_value=0.80, max_value=0.99, value=0.95, step=0.01)

        include_segments = st.checkbox("Include segments", value=True)
        seed = st.text_input("Seed (optional)", value="")
        sims = st.slider("Power simulations", min_value=200, max_value=5000, value=1000, step=200)

        save_history = st.checkbox("Save to history (SQLite)", value=True)
        run = st.button("Run simulation", type="primary")
    else:
        st.info("History is stored locally in abtest.db (SQLite).")
        history_limit = st.slider("How many experiments", 5, 50, 25, 5)
        experiments = get_experiments(limit=history_limit)
        if experiments:
            options = {f"#{e['id']} • {e['name']} • {e['created_at']}": e for e in experiments}
            selected_label = st.selectbox("Select an experiment", list(options.keys()), index=0)
            selected = options[selected_label]
        else:
            selected = None


def render_results(df: pd.DataFrame, baseline: float, uplift: float, sample_size: int, confidence_level: float, title_prefix: str = ""):
    a = df[df["group"] == "A"]
    b = df[df["group"] == "B"]

    a_n = len(a)
    b_n = len(b)
    a_conv = int(a["converted"].sum())
    b_conv = int(b["converted"].sum())

    res = z_test_proportions(a_conv, a_n, b_conv, b_n, confidence_level=confidence_level)
    rec = generate_recommendation(res)

    power_res = simulate_power_analysis(
        baseline_conversion=baseline,
        expected_uplift=uplift,
        sample_size=sample_size,
        simulations=st.session_state.get("power_sims", 1000),
        alpha=1 - confidence_level,
    )

    # --- top metrics ---
    st.subheader(f"{title_prefix}Results")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Control (A) conversion", f"{res.control_rate*100:.2f}%", f"{a_conv}/{a_n}")
    c2.metric("Treatment (B) conversion", f"{res.treatment_rate*100:.2f}%", f"{b_conv}/{b_n}")
    c3.metric("Absolute lift (B − A)", f"{res.absolute_lift*100:.2f} pp")
    c4.metric("Relative lift", f"{res.relative_lift:+.2f}%")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("p-value", f"{res.p_value:.4f}")
    c6.metric("z-score", f"{res.z_score:.3f}")
    c7.metric("Significant?", "Yes" if res.is_significant else "No")
    c8.metric("Estimated power", f"{power_res.power*100:.1f}%")

    st.write("**Confidence interval (B − A):**", f"[{res.confidence_interval.lower*100:.2f}, {res.confidence_interval.upper*100:.2f}] pp")

    st.success(rec) if (res.is_significant and res.relative_lift >= 0) else st.warning(rec)

    # --- charts ---
    left, right = st.columns(2)

    with left:
        st.markdown("#### Conversion rates")
        conv_df = pd.DataFrame(
            {
                "group": ["A", "B"],
                "conversion_rate": [res.control_rate, res.treatment_rate],
            }
        )
        fig = px.bar(conv_df, x="group", y="conversion_rate", text=conv_df["conversion_rate"].map(lambda v: f"{v*100:.2f}%"))
        fig.update_layout(yaxis_tickformat=",.0%", height=360)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("#### Confidence interval (difference)")
        diff = res.absolute_lift
        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=[res.confidence_interval.lower, res.confidence_interval.upper],
                y=[0, 0],
                mode="lines",
                line=dict(width=8),
                name="CI",
            )
        )
        fig2.add_trace(
            go.Scatter(
                x=[diff],
                y=[0],
                mode="markers",
                marker=dict(size=14),
                name="Observed lift",
            )
        )
        fig2.update_layout(
            xaxis_title="Lift (B − A)",
            yaxis_visible=False,
            yaxis_showticklabels=False,
            height=360,
        )
        fig2.update_xaxes(tickformat=",.2%")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Cumulative conversions over time")
    tmp = df.copy()
    tmp["date"] = pd.to_datetime(tmp["timestamp"]).dt.date
    tmp = tmp.groupby(["date", "group"], as_index=False)["converted"].sum()
    tmp = tmp.sort_values("date")
    tmp["cum_converted"] = tmp.groupby("group")["converted"].cumsum()

    fig3 = px.line(tmp, x="date", y="cum_converted", color="group", markers=True)
    fig3.update_layout(height=340)
    st.plotly_chart(fig3, use_container_width=True)

    seg_tbl = _compute_segment_table(df)
    if len(seg_tbl):
        st.markdown("#### Segment analysis")
        show = seg_tbl.copy()
        show["control_rate"] = (show["control_rate"] * 100).map(lambda v: f"{v:.2f}%")
        show["treatment_rate"] = (show["treatment_rate"] * 100).map(lambda v: f"{v:.2f}%")
        show["relative_lift_%"] = show["relative_lift_%"].map(lambda v: f"{v:+.2f}%")
        st.dataframe(show, use_container_width=True, hide_index=True)

    st.markdown("#### Download data")
    csv_str = export_to_csv(df)
    st.download_button(
        label="Download CSV",
        data=csv_str.encode("utf-8"),
        file_name="experiment_data.csv",
        mime="text/csv",
    )

    return res, rec, power_res


if mode == "Run new simulation":
    if run:
        st.session_state["power_sims"] = int(sims)

        df = generate_experiment_data(
            baseline_conversion=float(baseline),
            expected_uplift=float(uplift),
            sample_size=int(sample_size),
            include_segments=bool(include_segments),
            seed=seed.strip() or None,
        )

        res, rec, power_res = render_results(df, float(baseline), float(uplift), int(sample_size), float(confidence_level))

        if save_history:
            exp_id = save_experiment(
                name=name.strip() or "Experiment",
                baseline_conversion=float(baseline),
                expected_uplift=float(uplift),
                sample_size=int(sample_size),
                confidence_level=float(confidence_level),
                result=res,
                recommendation=rec,
            )
            save_experiment_data(exp_id, df)
            st.info(f"Saved to history as experiment #{exp_id} (abtest.db).")

else:
    if not selected:
        st.warning("No experiments found yet. Run a simulation first.")
    else:
        st.session_state["power_sims"] = 1000
        df = get_experiment_data(int(selected["id"]))

        baseline = float(selected["baseline_conversion"])
        uplift = float(selected["expected_uplift"])
        sample_size = int(selected["sample_size"])
        confidence_level = float(selected["confidence_level"])

        st.subheader("Experiment details")
        meta = {
            "name": selected["name"],
            "created_at": selected["created_at"],
            "baseline_conversion": baseline,
            "expected_uplift": uplift,
            "sample_size_per_group": sample_size,
            "confidence_level": confidence_level,
        }
        st.json(meta)

        render_results(df, baseline, uplift, sample_size, confidence_level, title_prefix="Historical ")
