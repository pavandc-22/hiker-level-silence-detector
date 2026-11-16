"""
app.py
======
Streamlit Demonstration App for Capstone Phase-1 Review:
"Event-Driven SNN-GNN Framework for Terrain-Aware Silence Attribution and Shared-Fate Detection in Offline Hiking Groups"

DISCLAIMER & INTEGRITY NOTICE:
- PROTOTYPE DEMO — SIMULATED DATA, RULE-BASED ANALYSIS; NOT A VALIDATED SAFETY SYSTEM.
- This application uses synthetic scenario datasets and transparent heuristics.
- It does NOT employ trained neural network weights (SNN or GNN), live hardware sensors, or field-tested emergency classifiers.
"""

import streamlit as st
import pandas as pd
import numpy as np

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from typing import Dict, Any

import data_generator
from pathlib import Path
from gps_processor import load_gps_dataset, get_track_summary
from analysis import (
    PrototypeConfig,
    run_pipeline,
    AttributionResult,
    IndividualHikerAnalysis,
    GroupPatternAnalysis,
)

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Silence Attribution & Shared-Fate Prototype",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Presentation-friendly CSS styling
st.markdown("""
<style>
    /* Metric Card Styling */
    .metric-box {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 14px 18px;
        border-left: 4px solid #4a6fa5;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        margin-bottom: 10px;
    }
    .metric-box-title {
        font-size: 0.85rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .metric-box-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #212529;
    }
    .metric-box-sub {
        font-size: 0.8rem;
        color: #888;
        margin-top: 2px;
    }

    /* Result Banner */
    .status-card {
        border-radius: 10px;
        padding: 20px 24px;
        margin-bottom: 20px;
        color: white;
    }
    .status-card-title {
        font-size: 0.9rem;
        letter-spacing: 1px;
        text-transform: uppercase;
        font-weight: 600;
        opacity: 0.9;
    }
    .status-card-heading {
        font-size: 1.85rem;
        font-weight: 800;
        margin: 6px 0 10px 0;
    }
    .status-card-summary {
        font-size: 1.05rem;
        line-height: 1.45;
        opacity: 0.95;
    }

    /* Disclaimer Banner */
    .prototype-banner {
        background: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
        padding: 12px 18px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 18px;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* Stage Header */
    .stage-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 700;
        background-color: #e9ecef;
        color: #495057;
        margin-bottom: 6px;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS & THRESHOLDS
# -----------------------------------------------------------------------------
st.sidebar.title("⚙️ Controls & Parameters")

# Reset button helper
if "reset_trigger" not in st.session_state:
    st.session_state["reset_trigger"] = 0

def reset_to_defaults():
    st.session_state["selected_scenario"] = "scenario_1"
    st.session_state["low_link_threshold"] = 0.35
    st.session_state["consecutive_miss_threshold"] = 2
    st.session_state["group_silence_min_hikers"] = 2
    st.session_state["movement_halt_threshold"] = 0.8
    st.session_state["movement_drop_pct_threshold"] = 60.0
    st.session_state["reset_trigger"] += 1

# Scenario selector
scenario_options = {
    "scenario_1": "1. Shared Coverage Gap",
    "scenario_2": "2. Individual Anomaly",
    "scenario_3": "3. Possible Group Risk",
}

if "selected_scenario" not in st.session_state:
    st.session_state["selected_scenario"] = "scenario_1"

selected_scenario_key = st.sidebar.selectbox(
    "Select Demo Scenario:",
    options=list(scenario_options.keys()),
    format_func=lambda k: scenario_options[k],
    key="selected_scenario",
)

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Heuristic Thresholds")
st.sidebar.caption("Configurable rule thresholds (illustrative, not validated safety boundaries):")

low_link_thresh = st.sidebar.slider(
    "Low Link Quality Threshold",
    min_value=0.10,
    max_value=0.60,
    value=st.session_state.get("low_link_threshold", 0.35),
    step=0.05,
    help="RF link quality below this value indicates terrain/environmental attenuation.",
    key="low_link_threshold",
)

consec_miss_thresh = st.sidebar.slider(
    "Consecutive Miss Threshold",
    min_value=1,
    max_value=4,
    value=st.session_state.get("consecutive_miss_threshold", 2),
    step=1,
    help="Number of sequential missed check-ins required to trigger persistent silence.",
    key="consecutive_miss_threshold",
)

group_silence_min = st.sidebar.slider(
    "Min Hikers for Group Silence",
    min_value=2,
    max_value=4,
    value=st.session_state.get("group_silence_min_hikers", 2),
    step=1,
    help="Minimum number of concurrently silent hikers to evaluate shared-fate patterns.",
    key="group_silence_min_hikers",
)

movement_halt_thresh = st.sidebar.slider(
    "Movement Halt Threshold (km/h)",
    min_value=0.0,
    max_value=2.0,
    value=st.session_state.get("movement_halt_threshold", 0.8),
    step=0.1,
    help="Hiker speed at or below this value is treated as stationary/arrested.",
    key="movement_halt_threshold",
)

movement_drop_pct_thresh = st.sidebar.slider(
    "Speed Drop Anomaly (%)",
    min_value=30.0,
    max_value=90.0,
    value=st.session_state.get("movement_drop_pct_threshold", 60.0),
    step=5.0,
    help="Percentage drop from baseline speed considered an unexpected deceleration.",
    key="movement_drop_pct_threshold",
)

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reset Demo to Defaults", use_container_width=True, on_click=reset_to_defaults):
    st.sidebar.success("Reset applied!")

st.sidebar.markdown("""
<div style="font-size:0.75rem; color:#6c757d; line-height:1.4; padding-top:10px;">
<b>Research Framing Note:</b><br>
In the full Capstone framework, these thresholds represent hyper-parameters and feature boundaries that will be replaced by event-driven Spiking Neural Networks (SNN, on-node) and Graph Neural Networks (GNN, group topology).
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# MAIN HEADER & MANDATORY PROTOTYPE BANNER
# -----------------------------------------------------------------------------
st.title("Event-Driven SNN-GNN Framework for Terrain-Aware Silence Attribution and Shared-Fate Detection in Offline Hiking Groups")
st.caption("Capstone Phase-1 Interactive Demonstration Prototype | Simulated Scenarios & Rule-Based Heuristic Attribution")

# Mandatory prominent prototype banner
st.markdown("""
<div class="prototype-banner">
    ⚠️ <span><b>PROTOTYPE DEMO — SIMULATED DATA, RULE-BASED ANALYSIS; NOT A VALIDATED SAFETY SYSTEM</b><br>
    <small>All sensor metrics, RF qualities, and trajectories are synthetic. Heuristic logic is illustrative for conceptual validation and does not offer certified emergency detection.</small></span>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# DATA LOADING & VALIDATION
# -----------------------------------------------------------------------------
try:
    df_raw = data_generator.load_scenario(selected_scenario_key)
except Exception as e:
    st.error(f"Error loading scenario data: {e}")
    st.stop()

# Scenario context card
scenario_meta = data_generator.SCENARIO_METADATA[selected_scenario_key]
st.info(f"**Current Scenario:** {scenario_meta['name']} — {scenario_meta['description']}")

# Timeline scrubber (scrub through observations to show deterministic evolution)
all_timestamps = sorted(df_raw["timestamp"].unique().tolist())
selected_timestamp = st.select_slider(
    "⏱️ **Observation Timeline Scrubber** (Simulated Check-in Cycle):",
    options=all_timestamps,
    value=all_timestamps[-1],
    help="Slide along the timeline to observe how individual silence and group status evolve as hikers advance through route segments.",
)

# Build config object
config = PrototypeConfig(
    low_link_threshold=low_link_thresh,
    consecutive_miss_threshold=consec_miss_thresh,
    group_silence_min_hikers=group_silence_min,
    movement_halt_threshold=movement_halt_thresh,
    movement_drop_pct_threshold=movement_drop_pct_thresh,
)

# Run modular analysis pipeline
try:
    analysis_result: AttributionResult = run_pipeline(
        df_raw,
        config=config,
        at_timestamp=selected_timestamp
    )
except Exception as e:
    st.error(f"Pipeline execution failed: {e}")
    st.stop()


# -----------------------------------------------------------------------------
# TOP-LEVEL STATUS BANNER & HIGH-LEVEL METRICS
# -----------------------------------------------------------------------------
grp = analysis_result.group_summary

# Map status card background colors
status_bg_colors = {
    "Group risk — follow-up required": "#dc3545",          # Red
    "Likely shared connectivity issue": "#e67e22",         # Amber/Orange
    "Individual anomaly — check required": "#d35400",      # Deep Orange
    "Insufficient evidence": "#5a6268",                    # Slate Grey
}
bg_color = status_bg_colors.get(analysis_result.classification, "#4a6fa5")

st.markdown(f"""
<div class="status-card" style="background-color: {bg_color};">
    <div class="status-card-title">Attribution Pipeline Synthesis (Stage 4 Output)</div>
    <div class="status-card-heading">🏷️ {analysis_result.classification}</div>
    <div class="status-card-summary">{analysis_result.summary}</div>
</div>
""", unsafe_allow_html=True)

# 4 Key Summary Metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    silent_txt = f"{grp.silent_hikers_count} / {grp.total_hikers}"
    silent_sub = f"Silent: {', '.join(grp.silent_hiker_ids) if grp.silent_hiker_ids else 'None'}"
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-box-title">Group Check-In Status</div>
        <div class="metric-box-value">{silent_txt} Silent</div>
        <div class="metric-box-sub">{silent_sub}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    link_status = "⚠️ Degraded" if grp.is_group_link_degraded else "✅ Adequate"
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-box-title">Mean Link Quality (RF)</div>
        <div class="metric-box-value">{grp.mean_group_link_quality:.2f} <span style="font-size:0.9rem;">(0–1)</span></div>
        <div class="metric-box-sub">{link_status} (Threshold: &le; {config.low_link_threshold:.2f})</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    halt_status = "🛑 Stationary / Halted" if grp.is_group_halted else "🥾 Active Traversal"
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-box-title">Mean Group Speed</div>
        <div class="metric-box-value">{grp.mean_group_speed:.1f} <span style="font-size:0.9rem;">km/h</span></div>
        <div class="metric-box-sub">{halt_status} (Threshold: &le; {config.movement_halt_threshold:.1f})</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-box-title">Active Terrain Context</div>
        <div class="metric-box-value" style="font-size:1.15rem;">{grp.dominant_terrain}</div>
        <div class="metric-box-sub">Segment: {grp.active_segment}</div>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# TABBED PRESENTATION SECTIONS
# -----------------------------------------------------------------------------
tab_overview, tab_pipeline, tab_charts, tab_data, tab_guide, tab_gps, tab_hybrid = st.tabs([
    "🔍 Analysis & Reasoning",
    "🧠 Rule-Based Pipeline Stages (SNN/GNN Proxy)",
    "📈 Temporal & Kinematic Charts",
    "📋 Observation Data Table",
    "⏱️ 2-Minute Review Guide",
    "Real Hiking GPS Explorer",
    "Hybrid GPS + Communication Analysis",
])


# =============================================================================
# TAB 1: ANALYSIS & REASONING ("Why this result?")
# =============================================================================
with tab_overview:
    st.subheader("💡 Why This Result? (Transparent Evidence Breakdown)")

    col_reason, col_rules = st.columns([1.2, 1.0])

    with col_reason:
        st.markdown("#### Primary Evidence Factors:")
        for idx, reason in enumerate(analysis_result.reasoning_points, start=1):
            st.markdown(f"**{idx}.** {reason}")

        st.markdown("---")
        st.markdown("#### Physical Attribution Interpretation:")
        if analysis_result.classification == "Likely shared connectivity issue":
            st.markdown("""
            > **Shared Fate Interpretation:** When several nodes enter an RF-shadowed topographical zone simultaneously,
            > wireless packet loss is correlated. Because their kinematics remain normal (hiking speed unchanged),
            > this absence of signal is attributed to environmental RF blockage rather than physical distress.
            """)
        elif analysis_result.classification == "Individual anomaly — check required":
            st.markdown("""
            > **Individual Divergence Interpretation:** In high link-quality conditions where peer hikers continue checking in,
            > an isolated node experiencing repeated silence AND cessation of movement represents an anomalous deviation.
            > This points toward individual separation, injury, or hardware loss.
            """)
        elif analysis_result.classification == "Group risk — follow-up required":
            st.markdown("""
            > **Collective Risk Interpretation:** Simultaneous cessation of check-ins combined with abrupt group standstill
            > in terrain with adequate RF coverage rules out normal RF shadowing. This indicates a potential shared-hazard event
            > (e.g. avalanche, impassable trail, rockfall, or severe weather incident) requiring follow-up.
            """)
        else:
            st.markdown("""
            > **Indeterminate Interpretation:** The current observation window does not provide enough correlated signals
            > to satisfy the prototype attribution thresholds. Data collection continues.
            """)

    with col_rules:
        st.markdown("#### Transparent Heuristic Rules Trace:")
        st.caption("Inspect exact Boolean conditions evaluated at this timestamp:")

        for rule_item in analysis_result.rule_criteria_met:
            rule_name = rule_item["rule"]
            matched = rule_item["matched"]
            status_symbol = "✅ MATCHED" if matched else "❌ NOT MET"
            border_color = "#28a745" if matched else "#ced4da"

            with st.expander(f"{rule_name} — {status_symbol}", expanded=matched):
                for cond_name, cond_val in rule_item["conditions"].items():
                    c_icon = "🟢" if cond_val else "⚪"
                    st.write(f"{c_icon} **{cond_name}**: `{cond_val}`")
                if matched:
                    st.success(f"Rule triggered outcome: '{analysis_result.classification}'")


# =============================================================================
# TAB 2: PIPELINE STAGES (Modular SNN / GNN Representation)
# =============================================================================
with tab_pipeline:
    st.subheader("🔬 4-Stage Architectural Organization")
    st.markdown("""
    The prototype decouples data ingestion, individual node analysis, group graph aggregation, and decision synthesis.
    This architecture enables directly replacing rule-based heuristics with trained neural network modules in Phase-2.
    """)

    st.markdown("---")
    st.markdown('<span class="stage-pill">Stage 1</span> **Feature Preparation & Temporal Sequence Tracking**', unsafe_allow_html=True)
    st.write("Aggregates cumulative silence runs, baseline traversal velocity, and instant speed drops up to the active timestamp.")

    st.markdown("---")
    st.markdown('<span class="stage-pill">Stage 2</span> **Individual Silence Analysis (Proposed SNN Stage Proxy)**', unsafe_allow_html=True)
    st.caption("In Phase-2: An on-node neuromorphic Spiking Neural Network (SNN) will process asynchronous event trains. *Current implementation: Rule-based local heuristic.*")

    ind_cols = st.columns(4)
    for idx, (hid, h_info) in enumerate(analysis_result.individual_summary.items()):
        with ind_cols[idx]:
            card_border = "#dc3545" if h_info.is_currently_silent else "#28a745"
            st.markdown(f"""
            <div style="border: 2px solid {card_border}; border-radius: 8px; padding: 12px; background: white; min-height: 250px;">
                <div style="font-weight: 700; font-size: 1.1rem;">👤 Hiker {h_info.hiker_id}</div>
                <div style="font-size: 0.85rem; color: #555; margin-bottom: 8px;">Role: {data_generator.HIKERS[idx]['role']}</div>
                <hr style="margin: 6px 0;">
                <div style="font-size: 0.85rem;"><b>Status:</b> {'🔴 Silent' if h_info.is_currently_silent else '🟢 Reporting'}</div>
                <div style="font-size: 0.85rem;"><b>Consecutive Misses:</b> {h_info.consecutive_misses}</div>
                <div style="font-size: 0.85rem;"><b>Latest Speed:</b> {h_info.latest_speed:.1f} km/h</div>
                <div style="font-size: 0.85rem;"><b>Speed Drop:</b> {h_info.speed_drop_pct:.0f}%</div>
                <div style="font-size: 0.85rem;"><b>Link Quality:</b> {h_info.latest_link_quality:.2f}</div>
                <div style="font-size: 0.85rem; margin-top: 6px; color: #666;"><em>{h_info.notes[-1] if h_info.notes else ''}</em></div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<span class="stage-pill">Stage 3</span> **Group-Level Pattern Analysis (Proposed GNN Stage Proxy)**', unsafe_allow_html=True)
    st.caption("In Phase-2: A Graph Neural Network (GNN) will aggregate spatial mesh topology and terrain-aware edge attributes to infer shared fate. *Current implementation: Group-level relational co-occurrence rules.*")

    g_col1, g_col2 = st.columns(2)
    with g_col1:
        st.markdown(f"""
        - **Simultaneous Silence Detected:** `{grp.is_simultaneous_silence}` ({grp.silent_hikers_count} silent hikers)
        - **Peer Isolation Detected:** `{grp.isolation_detected}` (Isolated hiker: `{grp.isolated_hiker_id or 'None'}`)
        - **Group Mobility State:** `{'Halted' if grp.is_group_halted else 'Moving'}` (Mean velocity: `{grp.mean_group_speed:.2f} km/h`)
        """)
    with g_col2:
        st.markdown(f"""
        - **Terrain Obstruction Index:** `{grp.dominant_terrain}`
        - **Group Mean RF Link Quality:** `{grp.mean_group_link_quality:.2f}` (Degraded: `{grp.is_group_link_degraded}`)
        - **Active Route Segment:** `{grp.active_segment}`
        """)

    st.markdown("---")
    st.markdown('<span class="stage-pill">Stage 4</span> **Status & Explanation Generation**', unsafe_allow_html=True)
    st.write(f"Synthesizes outputs from Stages 2 and 3 into high-level human-readable attribution: **'{analysis_result.classification}'** with detailed justification.")


# =============================================================================
# TAB 3: CHARTS (Temporal & Kinematic Visualizations)
# =============================================================================
with tab_charts:
    st.subheader("📊 Temporal Dynamics & Kinematic Analysis")
    st.caption("Visualizing the relationship between check-in reception, terrain link quality, and movement speeds over time.")

    df_plot = df_raw.copy()
    df_plot["Check-in State"] = df_plot["checkin_received"].apply(lambda x: "Received (1)" if x else "Missed (0)")
    df_plot["Checkin_Num"] = df_plot["checkin_received"].astype(int)

    col_chart1, col_chart2 = st.columns(2)

    if HAS_PLOTLY:
        with col_chart1:
            st.markdown("##### 📡 Check-In Reception History per Hiker")
            fig_checkin = px.scatter(
                df_plot,
                x="timestamp",
                y="hiker_id",
                color="Check-in State",
                color_discrete_map={"Received (1)": "#28a745", "Missed (0)": "#dc3545"},
                size=[14] * len(df_plot),
                hover_data=["route_segment", "terrain_type", "link_quality", "movement_speed"],
                labels={"timestamp": "Simulated Timestamp", "hiker_id": "Hiker"},
                title="Check-in Status over Time (Green=Received, Red=Missed)",
            )
            fig_checkin.add_vline(x=selected_timestamp, line_width=2, line_dash="dash", line_color="#4a6fa5")
            fig_checkin.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_checkin, use_container_width=True)

        with col_chart2:
            st.markdown("##### 🥾 Movement Speed over Time (km/h)")
            fig_speed = px.line(
                df_plot,
                x="timestamp",
                y="movement_speed",
                color="hiker_id",
                markers=True,
                labels={"movement_speed": "Speed (km/h)", "timestamp": "Timestamp", "hiker_id": "Hiker"},
                title="Estimated Hiker Movement Speed over Time",
            )
            fig_speed.add_hline(
                y=config.movement_halt_threshold,
                line_dash="dot",
                line_color="red",
                annotation_text=f"Halt Thresh ({config.movement_halt_threshold} km/h)",
            )
            fig_speed.add_vline(x=selected_timestamp, line_width=2, line_dash="dash", line_color="#4a6fa5")
            fig_speed.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_speed, use_container_width=True)

        # Chart 3: Link quality profile across segments
        st.markdown("##### 📶 Link Quality & Terrain Profile along Route")
        seg_summary = df_plot.groupby(["timestamp", "route_segment", "terrain_type"])["link_quality"].mean().reset_index()

        fig_link = go.Figure()
        fig_link.add_trace(go.Bar(
            x=seg_summary["timestamp"],
            y=seg_summary["link_quality"],
            text=[f"{s}<br>({t})" for s, t in zip(seg_summary["route_segment"], seg_summary["terrain_type"])],
            textposition="inside",
            marker_color=["#dc3545" if l <= config.low_link_threshold else "#28a745" for l in seg_summary["link_quality"]],
            name="Mean Link Quality",
        ))
        fig_link.add_hline(
            y=config.low_link_threshold,
            line_dash="dash",
            line_color="orange",
            annotation_text=f"Low Link Thresh ({config.low_link_threshold:.2f})",
        )
        fig_link.add_vline(x=selected_timestamp, line_width=2, line_dash="dash", line_color="#4a6fa5")
        fig_link.update_layout(
            title="Simulated RF Link Quality along Route Segments (Red = Degraded, Green = Adequate)",
            yaxis_title="Link Quality (0.0 to 1.0)",
            xaxis_title="Timestamp",
            height=320,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_link, use_container_width=True)
    else:
        # Graceful fallback to Streamlit native charts if plotly is not installed
        with col_chart1:
            st.markdown("##### 📡 Check-In Received per Hiker (1=Yes, 0=No)")
            piv_check = df_plot.pivot(index="timestamp", columns="hiker_id", values="Checkin_Num")
            st.line_chart(piv_check)

        with col_chart2:
            st.markdown("##### 🥾 Movement Speed per Hiker (km/h)")
            piv_speed = df_plot.pivot(index="timestamp", columns="hiker_id", values="movement_speed")
            st.line_chart(piv_speed)

        st.markdown("##### 📶 Link Quality per Segment")
        seg_summary = df_plot.groupby("timestamp")["link_quality"].mean()
        st.bar_chart(seg_summary)


# =============================================================================
# TAB 4: DATA TABLE (Inspect Raw / Filtered Records)
# =============================================================================
with tab_data:
    st.subheader("📋 Scenario Observation Records (Simulated)")
    st.caption("Inspect the underlying synthetic dataset. Values are generated deterministically.")

    # Filter options
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        filter_hiker = st.multiselect("Filter by Hiker:", options=df_raw["hiker_id"].unique(), default=df_raw["hiker_id"].unique())
    with filter_col2:
        show_up_to_scrubber = st.checkbox("Show records only up to selected scrubber timestamp", value=True)

    df_display = df_raw[df_raw["hiker_id"].isin(filter_hiker)].copy()
    if show_up_to_scrubber:
        df_display = df_display[df_display["timestamp"] <= selected_timestamp]

    # Format for readability
    display_table = df_display.copy()
    display_table["checkin_received"] = display_table["checkin_received"].apply(lambda x: "🟢 True" if x else "🔴 False")
    display_table["link_quality"] = display_table["link_quality"].apply(lambda x: f"{x:.2f}")
    display_table["movement_speed"] = display_table["movement_speed"].apply(lambda x: f"{x:.1f} km/h")

    st.dataframe(
        display_table,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(f"Displaying {len(display_table)} of {len(df_raw)} records.")


# =============================================================================
# TAB 5: 2-MINUTE REVIEW GUIDE (Capstone Presentation Script)
# =============================================================================
with tab_guide:
    st.subheader("⏱️ Two-Minute Capstone Review Presentation Script")
    st.markdown("""
    Use this concise structured walkthrough to guide faculty evaluators through the 3 deterministic scenarios:

    ---

    ### 0:00 – 0:30 | Introduction & Problem Context
    - **Pitch:** *"In offline group hiking, missed check-ins often cause unnecessary panic (when caused by simple canyon shadows) or delayed rescue (when caused by an injury or group trap). Our proposed Phase-1 framework explores how terrain, connectivity, and movement can attribute silence accurately."*
    - **Disclaimer:** Point out the top banner: *"This Phase-1 prototype runs on simulated data and transparent rule-based stages that will serve as the architectural blueprint for our future SNN-GNN implementation."*

    ---

    ### 0:30 – 1:00 | Scenario 1: Shared Coverage Gap
    1. Select **"1. Shared Coverage Gap"** from the sidebar.
    2. Slide the scrubber to `09:25`.
    3. **Observe:** The banner shows **“Likely shared connectivity issue”**.
    4. **Explain:** *"Notice all 4 hikers enter Shadow Gorge where link quality drops to 0.16. All check-ins drop simultaneously, but their movement speed continues at 3.8 km/h. Because movement continues through poor RF terrain, the framework attributes this to a shared coverage gap rather than an emergency."*

    ---

    ### 1:00 – 1:30 | Scenario 2: Individual Anomaly
    1. Select **"2. Individual Anomaly"** from the sidebar.
    2. Slide the scrubber to `09:25`.
    3. **Observe:** The banner updates to **“Individual anomaly — check required”**.
    4. **Explain:** *"Notice link quality is 0.86 (strong). Hikers H1, H2, and H3 are actively reporting and moving. However, Hiker H4 (Sweep) has missed 3 check-ins and their speed has dropped to 0.0 km/h. The peer graph isolates H4 as an anomalous node requiring a check."*

    ---

    ### 1:30 – 2:00 | Scenario 3: Possible Group Risk & Conclusion
    1. Select **"3. Possible Group Risk"** from the sidebar.
    2. Slide the scrubber to `09:25`.
    3. **Observe:** The banner updates to **“Group risk — follow-up required”**.
    4. **Explain:** *"Here, link quality is moderate-to-good (0.69). Radio shadow cannot explain why all check-ins ceased. Simultaneously, movement for the entire group abruptly dropped to 0.0 km/h in a boulder field. This pattern indicates a potential shared-fate hazard requiring immediate follow-up."*
    5. **Conclude:** *"In Phase-2, we replace these Stage 2 & 3 prototype rules with neuromorphic SNN spike processing and GNN graph convolution."*
    """)



# =============================================================================
# TAB 6: REAL HIKING GPS DATA EXPLORER
# =============================================================================
with tab_gps:
    st.subheader("Real Hiking GPS Dataset Explorer")

    st.success(
        "REAL DATA: Actual GPS trajectories from the Balearic Islands "
        "hiking dataset. Communication events are not included in these files."
    )

    st.markdown(
        "Explore real recorded hiking routes, movement speed, distance "
        "and elevation. GPS intervals with gaps over 10 minutes or "
        "calculated speeds above 12 km/h are excluded from movement metrics."
    )

    gps_folder = Path(__file__).resolve().parent / "data" / "real_gps"

    @st.cache_data
    def get_real_tracks(folder):
        return load_gps_dataset(folder)

    try:
        real_tracks = get_real_tracks(str(gps_folder))

        if not real_tracks:
            st.warning("No valid GPX files found in data/real_gps.")
        else:
            track_names = sorted(real_tracks.keys())

            selected_track = st.selectbox(
                "Select a real hiking GPS track:",
                track_names
            )

            observations = real_tracks[selected_track]
            summary = get_track_summary(observations)
            gps_df = pd.DataFrame(observations)

            st.caption(
                f"Selected track: {selected_track} | "
                f"{summary['gps_points']} recorded GPS points"
            )

            # Summary metrics
            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Distance", f"{summary['distance_km']} km")
            c2.metric(
                "Active Duration",
                f"{summary['active_duration_hours']} hrs"
            )
            c3.metric(
                "Average Speed",
                f"{summary['average_speed_kmh']} km/h"
            )
            c4.metric(
                "Maximum Elevation",
                f"{summary['elevation_max_m']} m"
                if summary["elevation_max_m"] is not None
                else "Unavailable"
            )

            c5, c6, c7 = st.columns(3)
            c5.metric("GPS Points", summary["gps_points"])
            c6.metric("Valid Intervals", summary["valid_intervals"])
            c7.metric(
                "Minimum Elevation",
                f"{summary['elevation_min_m']} m"
                if summary["elevation_min_m"] is not None
                else "Unavailable"
            )

            st.markdown("---")
            st.markdown("### Real GPS Route")

            if HAS_PLOTLY:
                route_fig = go.Figure()

                gps_df = gps_df.reset_index(drop=True)

                # Break the route wherever an interval is invalid or contains a long gap.
                route_lat = []
                route_lon = []

                for i, row in gps_df.iterrows():
                    if i > 0 and (
                        not bool(row["valid_interval"])
                        or bool(row["time_gap"])
                    ):
                        route_lat.append(None)
                        route_lon.append(None)

                    route_lat.append(row["latitude"])
                    route_lon.append(row["longitude"])

                lat_min = float(gps_df["latitude"].min())
                lat_max = float(gps_df["latitude"].max())
                lon_min = float(gps_df["longitude"].min())
                lon_max = float(gps_df["longitude"].max())

                lat_pad = max((lat_max - lat_min) * 0.15, 0.003)
                lon_pad = max((lon_max - lon_min) * 0.15, 0.003)

                route_fig.add_trace(go.Scattergeo(
                    lat=route_lat,
                    lon=route_lon,
                    mode="lines",
                    line=dict(color="#38bdf8", width=3),
                    name="Recorded route",
                    connectgaps=False
                ))

                route_fig.add_trace(go.Scattergeo(
                    lat=gps_df["latitude"],
                    lon=gps_df["longitude"],
                    mode="markers",
                    marker=dict(
                        size=5,
                        color=gps_df["speed_kmh"],
                        colorscale="Viridis",
                        cmin=0,
                        colorbar=dict(
                            title="Speed<br>(km/h)",
                            x=1.02,
                            len=0.75
                        )
                    ),
                    name="GPS observations",
                    showlegend=False,
                    text=[
                        f"Speed: {s:.2f} km/h<br>Elevation: {e} m"
                        for s, e in zip(
                            gps_df["speed_kmh"],
                            gps_df["elevation"]
                        )
                    ],
                    hoverinfo="text"
                ))

                route_fig.add_trace(go.Scattergeo(
                    lat=[gps_df.iloc[0]["latitude"]],
                    lon=[gps_df.iloc[0]["longitude"]],
                    mode="markers",
                    marker=dict(size=12, color="#22c55e", symbol="circle"),
                    name="Start"
                ))

                route_fig.add_trace(go.Scattergeo(
                    lat=[gps_df.iloc[-1]["latitude"]],
                    lon=[gps_df.iloc[-1]["longitude"]],
                    mode="markers",
                    marker=dict(size=12, color="#ef4444", symbol="diamond"),
                    name="End"
                ))

                route_fig.update_layout(
                    title=f"GPS Trajectory: {selected_track}",
                    geo=dict(
                        projection_type="mercator",
                        showland=True,
                        landcolor="#f1e8c8",
                        showcountries=True,
                        lataxis=dict(
                            range=[lat_min - lat_pad, lat_max + lat_pad]
                        ),
                        lonaxis=dict(
                            range=[lon_min - lon_pad, lon_max + lon_pad]
                        ),
                        domain=dict(x=[0, 0.88], y=[0, 1])
                    ),
                    legend=dict(
                        x=0.01,
                        y=0.99,
                        bgcolor="rgba(255,255,255,0.75)"
                    ),
                    height=550,
                    margin=dict(l=10, r=110, t=60, b=10)
                )
                st.plotly_chart(
                    route_fig,
                    use_container_width=True
                )

                st.markdown("---")
                st.markdown("### Movement Speed Over Time")

                # Prepare chart data while preserving missing GPS intervals.
                chart_rows = []
                gps_df = gps_df.reset_index(drop=True)

                for i, row in gps_df.iterrows():
                    interval_valid = bool(row["valid_interval"])
                    is_time_gap = bool(row["time_gap"])

                    # Insert an explicit break before disconnected observations.
                    if i > 0 and (not interval_valid or is_time_gap):
                        chart_rows.append({
                            "timestamp": row["timestamp"],
                            "speed_kmh": None,
                            "elevation": None
                        })

                    chart_rows.append({
                        "timestamp": row["timestamp"],
                        "speed_kmh": (
                            row["speed_kmh"] if interval_valid and not is_time_gap
                            else None
                        ),
                        "elevation": row["elevation"]
                    })

                chart_df = pd.DataFrame(chart_rows)

                speed_df = chart_df.dropna(
                    subset=["speed_kmh"]
                )

                if not speed_df.empty:
                    speed_fig = px.line(
                        chart_df,
                        x="timestamp",
                        y="speed_kmh",
                        title="Filtered GPS Movement Speed",
                        labels={
                            "timestamp": "GPS Timestamp",
                            "speed_kmh": "Speed (km/h)"
                        }
                    )
                    speed_fig.update_traces(connectgaps=False)
                    speed_fig.update_layout(height=350)
                    st.plotly_chart(
                        speed_fig,
                        use_container_width=True
                    )
                else:
                    st.info("No valid GPS movement intervals are available for this track.")

                st.markdown("### Elevation Profile")

                elevation_df = chart_df.dropna(
                    subset=["elevation"]
                )

                if not elevation_df.empty:
                    elevation_fig = px.line(
                        chart_df,
                        x="timestamp",
                        y="elevation",
                        title="Recorded Elevation Over Time",
                        labels={
                            "timestamp": "GPS Timestamp",
                            "elevation": "Elevation (m)"
                        }
                    )
                    elevation_fig.update_traces(connectgaps=False)
                    elevation_fig.update_layout(height=350)
                    st.plotly_chart(
                        elevation_fig,
                        use_container_width=True
                    )
                else:
                    st.info("No elevation observations are available for this track.")

                gap_count = int(gps_df["time_gap"].fillna(False).astype(bool).sum())
                invalid_count = int(
                    (~gps_df["valid_interval"].fillna(False).astype(bool)).sum()
                )

                st.caption(
                    f"Chart handling: {gap_count} long GPS gaps and "
                    f"{invalid_count} invalid intervals are represented as "
                    "discontinuities rather than continuous movement."
                )
            else:
                st.warning(
                    "Plotly is unavailable. Install the project "
                    "requirements to enable route visualisation."
                )

            st.markdown("---")
            st.markdown("### GPS Observation Records")

            st.dataframe(
                gps_df,
                use_container_width=True,
                hide_index=True
            )

            st.markdown("### Data Quality Information")
            st.write(
                f"- Long time gaps excluded: {summary['time_gaps']}"
            )
            st.write(
                f"- Other rejected intervals: "
                f"{summary['rejected_intervals']}"
            )
            st.write(
                "- Filtering assumptions: maximum 10-minute interval "
                "and maximum 12 km/h calculated speed."
            )

            st.info(
                "Research limitation: These are real individual hiking "
                "trajectories. They do not contain actual RF link quality, "
                "synchronised group check-ins, or labelled emergencies. "
                "Do not interpret rejected GPS intervals as emergencies."
            )

    except Exception as error:
        st.error(f"Could not load real GPS data: {error}")


# -----------------------------------------------------------------------------

# =============================================================================
# TAB 7: HYBRID REAL GPS + SIMULATED COMMUNICATION ANALYSIS
# =============================================================================
with tab_hybrid:
    st.subheader("Hybrid GPS + Communication Analysis")

    st.info(
        "DATA PROVENANCE: GPS coordinates, timestamps and elevation come "
        "from recorded GPX tracks. Check-in events and RF link quality are "
        "simulated experimental inputs. The four hikers share one recorded "
        "route context; they do not represent four independently tracked people."
    )

    hybrid_folder = Path(__file__).resolve().parent / "data" / "hybrid"

    hybrid_files = sorted(
        hybrid_folder.glob("10799_*.csv")
    ) if hybrid_folder.exists() else []

    if not hybrid_files:
        st.warning(
            "No hybrid datasets found. Run hybrid_data_generator.py "
            "to generate the experimental CSV files."
        )
    else:
        selected_hybrid_file = st.selectbox(
            "Select communication scenario",
            hybrid_files,
            format_func=lambda file: file.stem.replace(
                "10799_", ""
            ).replace("_", " ").title()
        )

        hybrid_df = pd.read_csv(selected_hybrid_file)

        required_hybrid_columns = [
            "scheduled_timestamp",
            "hiker_id",
            "latitude",
            "longitude",
            "elevation_m",
            "simulated_checkin_received",
            "simulated_link_quality",
            "gps_match_available",
            "gps_source_type",
            "communication_source_type",
        ]

        missing_hybrid_columns = [
            column for column in required_hybrid_columns
            if column not in hybrid_df.columns
        ]

        if missing_hybrid_columns:
            st.error(
                f"Hybrid dataset is missing columns: {missing_hybrid_columns}"
            )
        else:
            hybrid_df["scheduled_timestamp"] = pd.to_datetime(
                hybrid_df["scheduled_timestamp"],
                errors="coerce"
            )

            hybrid_df["simulated_checkin_received"] = (
                hybrid_df["simulated_checkin_received"]
                .astype(str)
                .str.lower()
                .map({"true": True, "false": False, "1": True, "0": False})
            )

            hybrid_df["simulated_link_quality"] = pd.to_numeric(
                hybrid_df["simulated_link_quality"],
                errors="coerce"
            )

            matched_gps = hybrid_df[
                hybrid_df["gps_match_available"].astype(str).str.lower() == "true"
            ].copy()

            unmatched_gps = len(hybrid_df) - len(matched_gps)

            missed_checkins = int(
                (hybrid_df["simulated_checkin_received"] == False).sum()
            )

            unique_gps_times = (
                matched_gps["gps_timestamp"].nunique()
                if "gps_timestamp" in matched_gps.columns
                else 0
            )

            metric1, metric2, metric3, metric4 = st.columns(4)

            metric1.metric("Event records", len(hybrid_df))
            metric2.metric("Matched GPS time points", unique_gps_times)
            metric3.metric("Unmatched records", unmatched_gps)
            metric4.metric("Simulated missed check-ins", missed_checkins)

            st.markdown("---")
            st.markdown("### Simulated Check-in Timeline")

            # Arrange check-in events as a hiker-by-time matrix.
            checkin_matrix = hybrid_df.pivot(
                index="hiker_id",
                columns="scheduled_timestamp",
                values="simulated_checkin_received"
            ).sort_index()

            checkin_matrix = checkin_matrix.reindex(
                ["H1", "H2", "H3", "H4"]
            )

            checkin_values = checkin_matrix.astype(float).values

            checkin_text = checkin_matrix.map(
                lambda value: (
                    "Received" if pd.notna(value) and bool(value)
                    else "Missed" if pd.notna(value)
                    else "No data"
                )
            ).values

            checkin_times = [
                pd.Timestamp(timestamp).strftime("%H:%M")
                for timestamp in checkin_matrix.columns
            ]

            checkin_fig = go.Figure(
                data=go.Heatmap(
                    z=checkin_values,
                    x=checkin_times,
                    y=checkin_matrix.index.tolist(),
                    text=checkin_text,
                    texttemplate="%{text}",
                    colorscale=[
                        [0.0, "#dc2626"],
                        [0.499, "#dc2626"],
                        [0.5, "#16a34a"],
                        [1.0, "#16a34a"]
                    ],
                    zmin=0,
                    zmax=1,
                    showscale=False,
                    xgap=3,
                    ygap=5,
                    hovertemplate=(
                        "Hiker: %{y}<br>"
                        "Scheduled time: %{x}<br>"
                        "Status: %{text}"
                        "<extra></extra>"
                    )
                )
            )

            checkin_fig.update_layout(
                title="Simulated Check-in Status — Hiker vs Time",
                xaxis_title="Scheduled Check-in Time",
                yaxis_title="Simulated Hiker",
                height=380,
                xaxis=dict(tickangle=-45),
                yaxis=dict(autorange="reversed")
            )

            st.plotly_chart(
                checkin_fig,
                width="stretch"
            )

            st.caption(
                "All check-in events shown above are simulated. "
                "They are not measurements from the GPX dataset."
            )

            st.markdown("---")
            st.markdown("### Simulated RF Link Quality")

            link_fig = px.line(
                hybrid_df,
                x="scheduled_timestamp",
                y="simulated_link_quality",
                color="hiker_id",
                markers=True,
                title="Simulated Link Quality by Hiker",
                labels={
                    "scheduled_timestamp": "Scheduled Time",
                    "simulated_link_quality": "Simulated Link Quality",
                    "hiker_id": "Simulated Hiker"
                }
            )

            link_fig.update_yaxes(range=[0, 1])
            link_fig.update_layout(height=400)
            st.plotly_chart(link_fig, use_container_width=True)

            st.caption(
                "RF link quality is synthetically generated for controlled "
                "experiments. It is not measured radio signal strength."
            )

            st.markdown("---")
            st.markdown("### Recorded GPS Route Context")

            if not matched_gps.empty:
                gps_unique = (
                    matched_gps.sort_values("gps_timestamp")
                    .drop_duplicates(subset=["gps_timestamp"])
                    .reset_index(drop=True)
                )

                gps_unique["gps_timestamp"] = pd.to_datetime(
                    gps_unique["gps_timestamp"],
                    utc=True,
                    errors="coerce"
                )

                gps_unique["gps_break_before"] = False

                if len(gps_unique) > 1:
                    elapsed_seconds = (
                        gps_unique["gps_timestamp"]
                        .diff()
                        .dt.total_seconds()
                    )

                    gps_unique["gps_break_before"] = (
                        elapsed_seconds > 600
                    )

                if "gps_valid_interval" in gps_unique.columns:
                    gps_unique["gps_break_before"] |= (
                        ~gps_unique["gps_valid_interval"].astype(bool)
                    )

                if "gps_time_gap" in gps_unique.columns:
                    gps_unique["gps_break_before"] |= (
                        gps_unique["gps_time_gap"].astype(bool)
                    )

                route_fig = go.Figure()

                # Preserve discontinuities in the recorded GPS trajectory.
                route_lat = []
                route_lon = []

                for i, row in gps_unique.iterrows():
                    if i > 0 and bool(row["gps_break_before"]):
                        route_lat.append(None)
                        route_lon.append(None)

                    route_lat.append(row["latitude"])
                    route_lon.append(row["longitude"])

                route_fig.add_trace(go.Scattergeo(
                    lat=route_lat,
                    lon=route_lon,
                    mode="lines+markers",
                    line=dict(color="#0ea5e9", width=3),
                    marker=dict(size=5, color="#0f766e"),
                    name="Recorded GPS route",
                    connectgaps=False
                ))

                route_fig.update_layout(
                    title="Real Recorded Route — Shared Group Context",
                    geo=dict(
                        projection_type="mercator",
                        showland=True,
                        showcountries=True,
                        lataxis=dict(
                            range=[
                                gps_unique["latitude"].min() - 0.005,
                                gps_unique["latitude"].max() + 0.005
                            ]
                        ),
                        lonaxis=dict(
                            range=[
                                gps_unique["longitude"].min() - 0.005,
                                gps_unique["longitude"].max() + 0.005
                            ]
                        )
                    ),
                    height=450
                )

                st.plotly_chart(route_fig, use_container_width=True)

                st.markdown("### Recorded Elevation Profile")

                elevation_data = gps_unique.dropna(
                    subset=["elevation_m"]
                )

                if not elevation_data.empty:
                    # Insert missing values at GPS discontinuities.
                    elevation_rows = []

                    for i, row in elevation_data.iterrows():
                        if i > 0 and bool(row["gps_break_before"]):
                            elevation_rows.append({
                                "gps_timestamp": row["gps_timestamp"],
                                "elevation_m": None
                            })

                        elevation_rows.append({
                            "gps_timestamp": row["gps_timestamp"],
                            "elevation_m": row["elevation_m"]
                        })

                    elevation_chart_df = pd.DataFrame(elevation_rows)

                    elevation_fig = px.line(
                        elevation_chart_df,
                        x="gps_timestamp",
                        y="elevation_m",
                        title="Recorded GPS Elevation",
                        labels={
                            "gps_timestamp": "GPS Timestamp",
                            "elevation_m": "Elevation (m)"
                        }
                    )
                    elevation_fig.update_traces(connectgaps=False)
                    elevation_fig.update_layout(height=350)
                    st.plotly_chart(
                        elevation_fig,
                        use_container_width=True
                    )
            else:
                st.warning(
                    "No matched GPS observations are available for this scenario."
                )

            st.markdown("---")
            st.markdown("### Data Provenance")

            provenance_df = pd.DataFrame([
                {
                    "Variable": "Latitude / Longitude",
                    "Source": "Real recorded GPX",
                    "Interpretation": "Recorded route coordinates"
                },
                {
                    "Variable": "Elevation",
                    "Source": "Real recorded GPX",
                    "Interpretation": "Recorded route elevation"
                },
                {
                    "Variable": "Movement speed",
                    "Source": "Real recorded GPX",
                    "Interpretation": "Speed between valid GPS observations"
                },
                {
                    "Variable": "Check-in received",
                    "Source": "Simulated",
                    "Interpretation": "Synthetic communication event"
                },
                {
                    "Variable": "RF link quality",
                    "Source": "Simulated",
                    "Interpretation": "Synthetic communication quality"
                },
                {
                    "Variable": "Individual hiker location",
                    "Source": "Unavailable",
                    "Interpretation": "Not measured by this dataset"
                },
                {
                    "Variable": "Emergency outcome",
                    "Source": "Unavailable",
                    "Interpretation": "No real emergency labels"
                }
            ])

            st.dataframe(
                provenance_df,
                use_container_width=True,
                hide_index=True
            )

            st.markdown("### Hybrid Event Records")

            st.dataframe(
                hybrid_df,
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "This dashboard is an experimental data exploration tool. "
                "It does not provide validated emergency detection or "
                "real-world hiker safety predictions."
            )

# FOOTER
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; font-size: 0.8rem; color: #888;">
    Capstone Phase-1 Review Demonstration Prototype | Event-Driven SNN-GNN Framework<br>
    Simulated Scenario Dataset • Transparent Deterministic Rule Engine • Python / Streamlit / Plotly
</div>
""", unsafe_allow_html=True)
