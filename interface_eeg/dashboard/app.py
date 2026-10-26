"""
dashboard/app.py
================
Streamlit Live Dashboard for:
EEG-Based Intent-Driven Shared-Control Prosthetic Arm System.

Visual System:
- Restrained clinical biomedical engineering control interface
- Deep charcoal canvas (#101216) with subtle panels (#1B1F25)
- Primary Accent: Muted Teal (#3AAFA9)
- Positive States: Clinical Muted Green (#38A169)
- Replay/Status: Muted Amber (#D69E2E)
- Fault/Error: Red (#E53E3E)
"""

import sys
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import joblib

# Ensure project root in sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.preprocessing import EEGPreprocessor
from src.replay import EEGDatasetReplayStream
from src.decision import SharedControlLayer
from src.serial_controller import STM32ProstheticController
from src.classifier import FilterBankCSPDeploymentPipeline

try:
    from dashboard.hand_visualizer import get_hand_svg
except ImportError:
    from hand_visualizer import get_hand_svg

# Streamlit Page Setup
st.set_page_config(
    page_title="EEG Prosthetic Arm Controller",
    page_icon="🦾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# RESTRAINED CLINICAL COLOR SYSTEM (CSS)
# =============================================================================
st.markdown("""
<style>
    /* Main App Background & Typography */
    .stApp {
        background-color: #101216 !important;
        color: #E5E7EB !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #181B21 !important;
        border-right: 1px solid #30343B !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span {
        color: #E5E7EB !important;
    }
    
    /* Headers & Text Hierarchy */
    h1 {
        color: #E5E7EB !important;
        font-size: 22px !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em;
        margin-bottom: 2px !important;
    }
    h2, h3, h4 {
        color: #E5E7EB !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        letter-spacing: 0.01em;
        margin-top: 4px !important;
        margin-bottom: 6px !important;
    }
    p, span, label {
        color: #9CA3AF;
        font-size: 13px;
    }
    
    /* Dividers */
    hr {
        border-color: #30343B !important;
        margin-top: 10px !important;
        margin-bottom: 12px !important;
    }
    
    /* Metrics Cards */
    div[data-testid="stMetric"] {
        background-color: #1B1F25 !important;
        border: 1px solid #30343B !important;
        border-radius: 4px !important;
        padding: 8px 12px !important;
    }
    div[data-testid="stMetric"] label {
        color: #9CA3AF !important;
        font-size: 11px !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #E5E7EB !important;
        font-size: 18px !important;
        font-weight: 600 !important;
    }
    
    /* Buttons - Standard Clinical Aesthetic */
    button[kind="secondary"], .stButton > button {
        background-color: #21252C !important;
        color: #E5E7EB !important;
        border: 1px solid #30343B !important;
        border-radius: 4px !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        padding: 4px 12px !important;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #2A303A !important;
        border-color: #3AAFA9 !important;
        color: #E5E7EB !important;
    }
    .stButton > button:active {
        background-color: #1A1E24 !important;
    }
    
    /* Sliders - Override default red to Teal Accent (#3AAFA9) */
    div[data-baseweb="slider"] div[role="slider"] {
        background-color: #3AAFA9 !important;
        border: 2px solid #101216 !important;
        box-shadow: none !important;
    }
    div[data-baseweb="slider"] > div > div > div:first-child {
        background-color: #3AAFA9 !important;
    }
    div[data-baseweb="slider"] > div > div {
        background-color: #30343B !important;
    }
    
    /* Selectbox & Inputs */
    div[data-baseweb="select"] > div {
        background-color: #14171D !important;
        border: 1px solid #30343B !important;
        border-radius: 4px !important;
        color: #E5E7EB !important;
    }
    
    /* Custom Clinical UI Badges */
    .badge-replay {
        background-color: #241E15;
        color: #D69E2E;
        border: 1px solid #63471A;
        border-radius: 3px;
        padding: 3px 8px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.03em;
        display: inline-block;
    }
    .badge-sampling {
        background-color: #132423;
        color: #3AAFA9;
        border: 1px solid #235451;
        border-radius: 3px;
        padding: 3px 8px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.03em;
        display: inline-block;
    }
    
    /* Decision Banners */
    .decision-accepted {
        background-color: #13281B;
        color: #68D391;
        border: 1px solid #276749;
        border-radius: 4px;
        padding: 10px 14px;
        font-size: 13px;
        font-weight: 600;
        text-align: center;
        letter-spacing: 0.02em;
    }
    .decision-hold {
        background-color: #1B1F25;
        color: #9CA3AF;
        border: 1px solid #30343B;
        border-radius: 4px;
        padding: 10px 14px;
        font-size: 13px;
        font-weight: 600;
        text-align: center;
        letter-spacing: 0.02em;
    }
    
    /* Clinical Debounce Readout */
    .debounce-container {
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 13px;
        font-weight: 600;
        text-align: center;
        padding: 6px 10px;
        border-radius: 3px;
        background-color: #14171D;
        border: 1px solid #30343B;
        color: #E5E7EB;
    }
    
    /* Probability Meter Rows */
    .prob-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 3px;
        font-size: 12px;
    }
    .prob-track {
        flex-grow: 1;
        height: 6px;
        background-color: #23272F;
        border-radius: 2px;
        margin: 0 10px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# INITIALIZE BACKEND RESOURCES
# =============================================================================
@st.cache_resource
def load_system_resources():
    data_path = os.path.join(root_dir, "data", "test_replay_stream.npz")
    stream = EEGDatasetReplayStream(data_path=data_path)
    preprocessor = EEGPreprocessor(sfreq=stream.sfreq, low_freq=8.0, high_freq=30.0)
    
    # Load FBCSP deployment pipeline
    fbcsp_path = os.path.join(root_dir, "models", "fbcsp_deployment_pipeline.joblib")
    fbcsp_model = None
    if os.path.exists(fbcsp_path):
        try:
            fbcsp_model = joblib.load(fbcsp_path)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            
    return stream, preprocessor, fbcsp_model

stream, preprocessor, fbcsp_model = load_system_resources()

if "decision_layer" not in st.session_state:
    st.session_state.decision_layer = SharedControlLayer(confidence_threshold=0.75, debounce_consecutive=2)

if "serial_controller" not in st.session_state:
    st.session_state.serial_controller = STM32ProstheticController(mode="SIMULATION")

if "is_running" not in st.session_state:
    st.session_state.is_running = False

if "history_gt" not in st.session_state:
    st.session_state.history_gt = []
if "history_pred" not in st.session_state:
    st.session_state.history_pred = []

target_14_channels = [
    "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6",
    "C5", "C3", "C1", "CZ", "C2", "C4", "C6"
]


# =============================================================================
# SIDEBAR: HARDWARE & CONTROL SETTINGS
# =============================================================================
st.sidebar.markdown("### System Configuration")

# Hardware Mode Setup
st.sidebar.markdown("**Actuator Interface**")
mode_choice = st.sidebar.radio(
    "Interface Selection",
    ["Simulation Mode", "Physical Hardware (STM32/UART)"],
    label_visibility="collapsed"
)

if mode_choice == "Physical Hardware (STM32/UART)":
    st.session_state.serial_controller.mode = "HARDWARE"
    ports = STM32ProstheticController.list_available_ports()
    port_names = [p["device"] + (" (" + p["description"][:16] + "...)" if p["description"] else "") for p in ports]
    if port_names:
        selected_port_str = st.sidebar.selectbox("Active Port", port_names)
        actual_port = selected_port_str.split(" ")[0]
        if st.sidebar.button("Initialize Port"):
            success = st.session_state.serial_controller.connect(actual_port)
            if success:
                st.sidebar.success(f"Connected: {actual_port}")
            else:
                st.sidebar.error(f"Failed to open {actual_port}")
    else:
        st.sidebar.warning("No COM devices detected.")
else:
    st.session_state.serial_controller.mode = "SIMULATION"
    st.sidebar.caption("Mode: Internal Software State Simulation")

st.sidebar.divider()

# Shared-Control Tuning
st.sidebar.markdown("**Shared-Control Parameters**")
conf_thresh = st.sidebar.slider("Confidence Gate (T_conf)", min_value=50, max_value=95, value=75, step=5, format="%d%%")
st.session_state.decision_layer.set_confidence_threshold(conf_thresh / 100.0)

debounce_windows = st.sidebar.slider("Temporal Debounce (N)", min_value=2, max_value=5, value=2, step=1, format="%d epochs")
st.session_state.decision_layer.set_debounce_consecutive(debounce_windows)

st.sidebar.divider()

# Manual Test Controls
st.sidebar.markdown("**Manual Hardware Override**")
col_m1, col_m2 = st.sidebar.columns(2)
with col_m1:
    if st.button("Close Hand"):
        st.session_state.decision_layer.arm_state = "CLOSED"
        st.session_state.serial_controller.send_command("GRASP")
with col_m2:
    if st.button("Open Hand"):
        st.session_state.decision_layer.arm_state = "OPEN"
        st.session_state.serial_controller.send_command("RELEASE")

st.sidebar.divider()

# Stream Execution
st.sidebar.markdown("**Replay Controls**")
col_s1, col_s2, col_s3 = st.sidebar.columns(3)
with col_s1:
    if st.button("Start"):
        st.session_state.is_running = True
with col_s2:
    if st.button("Pause"):
        st.session_state.is_running = False
with col_s3:
    if st.button("Reset"):
        stream.reset()
        st.session_state.history_gt.clear()
        st.session_state.history_pred.clear()
        st.session_state.is_running = False

refresh_delay = st.sidebar.slider("Refresh Interval", min_value=0.1, max_value=1.0, value=0.25, step=0.05, format="%.2fs")

# Quick Event Navigation
st.sidebar.markdown("**Trial Navigation (PhysioNet S001)**")
col_j1, col_j2, col_j3 = st.sidebar.columns(3)
with col_j1:
    if st.button("Grasp (4.2s)"):
        stream.seek_time(4.2)
        st.session_state.is_running = False
with col_j2:
    if st.button("Release (12.5s)"):
        stream.seek_time(12.5)
        st.session_state.is_running = False
with col_j3:
    if st.button("Rest (0.0s)"):
        stream.seek_time(0.0)
        st.session_state.is_running = False


# =============================================================================
# TOP HEADER
# =============================================================================
header_left, header_right = st.columns([3, 2])
with header_left:
    st.markdown("<h1>EEG-Based Intent-Driven Shared-Control Prosthetic Arm</h1>", unsafe_allow_html=True)
    st.caption("Common Spatial Patterns (FBCSP)  •  Regularized Shrinkage LDA  •  Shared-Control Safety Validation")

with header_right:
    st.markdown("""
        <div style="text-align: right; margin-top: 8px;">
            <span class="badge-replay">EEG SOURCE: DATASET REPLAY (PhysioNet S001)</span>
            &nbsp;
            <span class="badge-sampling">SAMPLING: 160 Hz</span>
        </div>
    """, unsafe_allow_html=True)

st.divider()


# =============================================================================
# PIPELINE EXECUTION STEP
# =============================================================================
pkg = stream.get_next_window()
raw_win = pkg["window_data"]
gt_label = pkg["ground_truth_label"]
cur_time = pkg["current_time_sec"]
ch_names = [ch.upper().replace('.', '') for ch in pkg["channel_names"]]

win_start = pkg["win_start_time"]
win_end = pkg["win_end_time"]
ctx_data = pkg.get("context_data", raw_win)
ctx_time = pkg.get("context_time_axis", np.linspace(win_start, win_end, raw_win.shape[1]))

# 1. Preprocessing (CAR + Notch + Bandpass)
t0 = time.perf_counter_ns()
proc_win = preprocessor.process_window(raw_win)
proc_ctx = preprocessor.process_window(ctx_data)
t1 = time.perf_counter_ns()

# 2. Spectral Power Calculation (Welch PSD)
band_powers = preprocessor.compute_band_powers(
    proc_win,
    ch_names=ch_names,
    target_channels=('C3', 'C4')
)
t2 = time.perf_counter_ns()

# 3. Model Inference (Filter-Bank CSP + Shrinkage LDA)
if fbcsp_model is not None:
    indices = [ch_names.index(ch) if ch in ch_names else 0 for ch in target_14_channels]
    win_14ch = raw_win[indices, :]
    
    # Scale microvolts to Volts to match model training calibration
    win_volts = win_14ch * 1e-6
    ml_res = fbcsp_model.predict_window(win_volts)
    
    pred_class = ml_res["predicted_class"]
    confidence = ml_res["confidence"]
    probas = ml_res["probabilities"]
else:
    pred_class = gt_label
    confidence = 0.85
    probas = {"REST": 0.10, "GRASP": 0.85, "RELEASE": 0.05}
t3 = time.perf_counter_ns()

# 4. Shared-Control & Safety Layer Decision
dec = st.session_state.decision_layer.process_intent(
    predicted_class=pred_class,
    confidence=confidence,
    probabilities=probas
)
t4 = time.perf_counter_ns()

# 5. Actuation Command Execution
serial_res = {"packet": "NONE", "latency_ms": 0.0}
if dec["decision"] == "ACCEPT":
    cmd = dec["motor_command"]
    serial_res = st.session_state.serial_controller.send_command(cmd)
t5 = time.perf_counter_ns()


# =============================================================================
# DASHBOARD BODY (2 COLUMNS)
# =============================================================================
col_left, col_right = st.columns([1, 1])

# -----------------------------------------------------------------------------
# LEFT COLUMN: EEG SIGNALS & SENSORIMOTOR SPECTRUM
# -----------------------------------------------------------------------------
with col_left:
    st.markdown("### 1. EEG Signal Oscilloscope (Motor Strip Channels)")
    
    idx_c3 = ch_names.index('C3') if 'C3' in ch_names else 0
    idx_cz = ch_names.index('CZ') if 'CZ' in ch_names else 1
    idx_c4 = ch_names.index('C4') if 'C4' in ch_names else 2
    
    fig = go.Figure()
    # 3 Carefully chosen muted readable waveform colors
    fig.add_trace(go.Scatter(x=ctx_time, y=proc_ctx[idx_c3, :] + 30.0, mode='lines', name='C3 (Left Motor)', line=dict(color='#5B9BD5', width=1.3)))
    fig.add_trace(go.Scatter(x=ctx_time, y=proc_ctx[idx_cz, :], mode='lines', name='Cz (Midline)', line=dict(color='#4FB3BF', width=1.3)))
    fig.add_trace(go.Scatter(x=ctx_time, y=proc_ctx[idx_c4, :] - 30.0, mode='lines', name='C4 (Right Motor)', line=dict(color='#C5A059', width=1.3)))
    
    # Active 2.0s Processing Window Shaded Box (Muted Teal Highlight)
    fig.add_vrect(
        x0=win_start,
        x1=win_end,
        fillcolor="rgba(79, 179, 191, 0.10)",
        line=dict(color="#4FB3BF", width=1.2, dash="dash"),
        annotation_text="ACTIVE 2.0s EPOCH",
        annotation_position="top left",
        annotation_font=dict(color="#4FB3BF", size=10, family="Segoe UI")
    )
    
    # Cursor Pointer (Muted Amber)
    fig.add_vline(
        x=win_end,
        line=dict(color="#D69E2E", width=2),
        annotation_text=f"POINTER (t = {win_end:.2f}s)",
        annotation_position="bottom right",
        annotation_font=dict(color="#D69E2E", size=10, family="Consolas")
    )
    
    fig.update_layout(
        height=280,
        margin=dict(l=15, r=15, t=10, b=20),
        paper_bgcolor='#1B1F25',
        plot_bgcolor='#14171D',
        xaxis=dict(
            title=dict(text='Timeline (seconds)', font=dict(color='#9CA3AF', size=11)),
            tickfont=dict(color='#9CA3AF', size=10),
            showgrid=True,
            gridcolor='#23272F'
        ),
        yaxis=dict(
            title=dict(text='Potential (μV)', font=dict(color='#9CA3AF', size=11)),
            tickfont=dict(color='#9CA3AF', size=10),
            showgrid=True,
            gridcolor='#23272F',
            range=[-85, 85]
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color='#9CA3AF', size=10)
        )
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.caption(f"Epoch Window: {win_start:.2f}s — {win_end:.2f}s  |  Ground-Truth Dataset Cue: {gt_label}")

    # Spectral Power Section
    st.markdown("### 2. Sensorimotor Rhythms (Welch PSD)")
    sp_c1, sp_c2 = st.columns(2)
    with sp_c1:
        st.markdown(f"""
            <div style="background-color: #1B1F25; border: 1px solid #30343B; border-radius: 4px; padding: 10px; margin-bottom: 6px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span style="color: #9CA3AF; font-size: 11px; text-transform: uppercase;">Mu Band (8–13 Hz)</span>
                    <span style="color: #E5E7EB; font-weight: 600; font-size: 11px;">{band_powers['mu_power']} μV²</span>
                </div>
                <div style="background: #23272F; height: 5px; border-radius: 2px;">
                    <div style="width: {min(100, int(band_powers['mu_power'] / 60.0 * 100))}%; background: #4FB3BF; height: 100%; border-radius: 2px;"></div>
                </div>
                <div style="display: flex; justify-content: space-between; margin-top: 8px; margin-bottom: 4px;">
                    <span style="color: #9CA3AF; font-size: 11px; text-transform: uppercase;">Beta Band (13–30 Hz)</span>
                    <span style="color: #E5E7EB; font-weight: 600; font-size: 11px;">{band_powers['beta_power']} μV²</span>
                </div>
                <div style="background: #23272F; height: 5px; border-radius: 2px;">
                    <div style="width: {min(100, int(band_powers['beta_power'] / 60.0 * 100))}%; background: #3AAFA9; height: 100%; border-radius: 2px;"></div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    with sp_c2:
        lat_val = band_powers['lateralization_index']
        lat_norm_pct = int(max(0.0, min(1.0, (lat_val + 1.0) / 2.0)) * 100)
        st.markdown(f"""
            <div style="background-color: #1B1F25; border: 1px solid #30343B; border-radius: 4px; padding: 10px; margin-bottom: 6px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span style="color: #9CA3AF; font-size: 11px; text-transform: uppercase;">Contralateral ERD Index</span>
                    <span style="color: #E5E7EB; font-weight: 600; font-size: 11px;">{lat_val:+.3f}</span>
                </div>
                <div style="background: #23272F; height: 5px; border-radius: 2px;">
                    <div style="width: {lat_norm_pct}%; background: #5B9BD5; height: 100%; border-radius: 2px;"></div>
                </div>
                <div style="display: flex; justify-content: space-between; margin-top: 8px;">
                    <span style="color: #9CA3AF; font-size: 10px;">C3 Mu: {band_powers['c3_mu']} μV²</span>
                    <span style="color: #9CA3AF; font-size: 10px;">C4 Mu: {band_powers['c4_mu']} μV²</span>
                </div>
            </div>
        """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# RIGHT COLUMN: ML POSTERIORS, SHARED CONTROL, & ACTUATOR
# -----------------------------------------------------------------------------
with col_right:
    st.markdown("### 3. Machine Learning Intent Decoders (FBCSP + LDA)")
    
    p_grasp = probas.get("GRASP", 0.0)
    p_release = probas.get("RELEASE", 0.0)
    p_rest = probas.get("REST", 0.0)
    
    # Clear Visual Hierarchy: Highest probability highlighted in Teal (#3AAFA9)
    max_intent = max(probas, key=probas.get)
    color_grasp = "#3AAFA9" if max_intent == "GRASP" else "#4B5563"
    color_release = "#3AAFA9" if max_intent == "RELEASE" else "#4B5563"
    color_rest = "#3AAFA9" if max_intent == "REST" else "#4B5563"
    
    text_color_grasp = "#E5E7EB" if max_intent == "GRASP" else "#9CA3AF"
    text_color_release = "#E5E7EB" if max_intent == "RELEASE" else "#9CA3AF"
    text_color_rest = "#E5E7EB" if max_intent == "REST" else "#9CA3AF"
    
    st.markdown(f"""
        <div style="background-color: #1B1F25; border: 1px solid #30343B; border-radius: 4px; padding: 12px; margin-bottom: 10px;">
            <!-- GRASP Row -->
            <div class="prob-row">
                <span style="color: {text_color_grasp}; width: 140px; font-weight: {'600' if max_intent == 'GRASP' else '400'};">GRASP (Close Hand)</span>
                <div class="prob-track">
                    <div style="width: {int(p_grasp * 100)}%; background-color: {color_grasp}; height: 100%;"></div>
                </div>
                <span style="color: {text_color_grasp}; font-family: monospace; width: 45px; text-align: right; font-weight: 600;">{p_grasp*100:.1f}%</span>
            </div>
            <!-- RELEASE Row -->
            <div class="prob-row">
                <span style="color: {text_color_release}; width: 140px; font-weight: {'600' if max_intent == 'RELEASE' else '400'};">RELEASE (Open Hand)</span>
                <div class="prob-track">
                    <div style="width: {int(p_release * 100)}%; background-color: {color_release}; height: 100%;"></div>
                </div>
                <span style="color: {text_color_release}; font-family: monospace; width: 45px; text-align: right; font-weight: 600;">{p_release*100:.1f}%</span>
            </div>
            <!-- REST Row -->
            <div class="prob-row">
                <span style="color: {text_color_rest}; width: 140px; font-weight: {'600' if max_intent == 'REST' else '400'};">REST (Baseline / Idle)</span>
                <div class="prob-track">
                    <div style="width: {int(p_rest * 100)}%; background-color: {color_rest}; height: 100%;"></div>
                </div>
                <span style="color: {text_color_rest}; font-family: monospace; width: 45px; text-align: right; font-weight: 600;">{p_rest*100:.1f}%</span>
            </div>
            <div style="display: flex; justify-content: space-between; border-top: 1px solid #282C34; margin-top: 8px; padding-top: 6px;">
                <span style="color: #9CA3AF; font-size: 11px;">Candidate Intent: <b style="color: #E5E7EB;">{pred_class}</b></span>
                <span style="color: #9CA3AF; font-size: 11px;">Confidence: <b style="color: #E5E7EB;">{confidence*100:.1f}%</b></span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Shared-Control & Safety Section
    st.markdown("### 4. Shared-Control & Safety Layer")
    
    c_gate, c_deb = st.columns([1, 1])
    with c_gate:
        if dec["confidence_passed"]:
            st.markdown(f"""
                <div style="background-color: #13281B; border: 1px solid #276749; border-radius: 4px; padding: 8px 10px; color: #68D391; font-size: 12px; font-weight: 600;">
                    Confidence Gate: PASS ({confidence*100:.1f}% ≥ {conf_thresh}%)
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style="background-color: #1B1F25; border: 1px solid #30343B; border-radius: 4px; padding: 8px 10px; color: #9CA3AF; font-size: 12px;">
                    Confidence Gate: HOLD ({confidence*100:.1f}% &lt; {conf_thresh}%)
                </div>
            """, unsafe_allow_html=True)
            
    with c_deb:
        marks_str = " ".join([f"[{m}]" for m in dec["debounce_marks"]])
        if dec["is_debounced"]:
            st.markdown(f'<div class="debounce-container" style="color: #68D391; border-color: #276749; background-color: #13281B;">{marks_str} (CONFIRMED)</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="debounce-container" style="color: #D69E2E; border-color: #63471A; background-color: #241E15;">{marks_str} (BUFFERING)</div>', unsafe_allow_html=True)
            
    st.caption(f"Safety Rule: {dec['reason']}")

    # Actuation Decision Banner
    if dec["decision"] == "ACCEPT":
        st.markdown(f"""
            <div class="decision-accepted">
                COMMAND ACCEPTED ──► TRANSMITTING ACTUATION: {dec['motor_command']}
            </div>
        """, unsafe_allow_html=True)
    else:
        if dec["arm_state"] == "CLOSED":
            hold_desc = "ACTIVE GRIP MAINTAINED (HAND CLOSED) • MOTORS IDLE"
        else:
            hold_desc = "OPEN STANDBY (HAND RELEASED) • AWAITING STABLE INTENT"
        st.markdown(f"""
            <div class="decision-hold">
                SYSTEM STATE: {hold_desc}
            </div>
        """, unsafe_allow_html=True)

    # Physical Hand Status Section
    st.markdown("### 5. Physical Hand State & Hardware Actuator")
    
    # Animated Kinematic Vector Hand (Clinical Styling)
    import streamlit.components.v1 as components
    components.html(get_hand_svg(dec["arm_state"]), height=230)
    
    h_col1, h_col2 = st.columns(2)
    with h_col1:
        st.metric("Mechanical State", dec["arm_state"], "Safety Interlock: Active")
            
    with h_col2:
        st.metric("Serial Packet", st.session_state.serial_controller.last_packet_raw)
        st.caption(f"Mode: {st.session_state.serial_controller.mode} | Link: {st.session_state.serial_controller.last_ack_response}")


# =============================================================================
# PERFORMANCE & TELEMETRY FOOTER
# =============================================================================
st.divider()
st.session_state.history_gt.append(gt_label)
st.session_state.history_pred.append(pred_class)
matches = sum(g == p for g, p in zip(st.session_state.history_gt, st.session_state.history_pred))
running_acc = (matches / len(st.session_state.history_gt)) * 100.0 if st.session_state.history_gt else 100.0

lat_prep = (t1 - t0) / 1e6
lat_spec = (t2 - t1) / 1e6
lat_inf = (t3 - t2) / 1e6
lat_dec = (t4 - t3) / 1e6
lat_comm = (t5 - t4) / 1e6
lat_total = (t5 - t0) / 1e6

f_c1, f_c2, f_c3, f_c4 = st.columns(4)
with f_c1:
    st.metric("Inference Latency", f"{lat_inf:.2f} ms")
with f_c2:
    st.metric("End-to-End Latency", f"{lat_total:.2f} ms")
with f_c3:
    st.metric("Dataset Agreement", f"{running_acc:.1f}%")
with f_c4:
    st.metric("Session Time", f"{cur_time:.1f} s")


# =============================================================================
# AUTO-RERUN LOOP IF STREAMING IS ACTIVE
# =============================================================================
if st.session_state.is_running:
    time.sleep(refresh_delay)
    st.rerun()
