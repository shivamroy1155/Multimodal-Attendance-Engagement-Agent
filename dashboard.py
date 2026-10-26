"""
Streamlit Dashboard — Milestone 6, Activity 6.2
==============================================
Launch: streamlit run dashboard.py

Features:
  - Live engagement score gauge
  - Real-time face count & audio metrics
  - Engagement timeline chart
  - Attendance table
  - Teaching suggestions feed
  - Student analytics breakdown
  - Session event log
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multimodal Attendance & Engagement Agent",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e, #16213e);
        border: 1px solid #2d3561;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #00d4ff;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8892b0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .suggestion-card {
        border-left: 4px solid #00d4ff;
        background: #1a1f2e;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 6px 0;
    }
    .suggestion-card.High { border-color: #ff4d4d; }
    .suggestion-card.Medium { border-color: #ffa500; }
    .suggestion-card.Low { border-color: #00d4ff; }
    .event-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-Critical { background: #ff4d4d22; color: #ff4d4d; border: 1px solid #ff4d4d; }
    .badge-Warning  { background: #ffa50022; color: #ffa500; border: 1px solid #ffa500; }
    .badge-Info     { background: #00d4ff22; color: #00d4ff; border: 1px solid #00d4ff; }
    .stProgress > div > div { background-color: #00d4ff !important; }
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ─────────────────────────────────────────────

def init_state():
    defaults = {
        "running": False,
        "tick": 0,
        "engagement_history": [],
        "face_count_history": [],
        "speech_history": [],
        "noise_history": [],
        "time_axis": [],
        "attendance_records": [],
        "suggestions": [],
        "events": [],
        "student_analytics": [],
        "session_start": None,
        "current_score": 0.0,
        "current_trend": "Stable",
        "current_faces": 0,
        "current_speech": False,
        "current_noise": 0.0,
        "last_alert": "",
        "agent": None,
        "vision": None,
        "audio": None,
        "analytics": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/classroom.png", width=64)
    st.title("🎓 Session Control")
    st.markdown("---")

    expected = st.slider("Expected Participants", 1, 30, 5)
    presence_thresh = st.slider("Attendance Threshold (sec)", 2, 30, 5)
    demo_mode = st.checkbox("Demo Mode (no camera)", value=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    start_btn = col1.button("▶ Start", use_container_width=True, type="primary")
    stop_btn  = col2.button("⏹ Stop",  use_container_width=True)

    st.markdown("---")
    st.markdown("**Load Saved Report**")
    report_files = list((ROOT / "outputs" / "reports").glob("*.json"))
    if report_files:
        selected_report = st.selectbox(
            "Report", [f.name for f in report_files], index=0
        )
    else:
        selected_report = None
        st.caption("No saved reports found.")

    load_report_btn = st.button("📂 Load Report", use_container_width=True)

    st.markdown("---")
    st.caption("Multimodal Attendance & Engagement Agent\n\nBuilt with OpenCV · MediaPipe · Librosa · Streamlit")


# ── Start / Stop logic ───────────────────────────────────────────────────────

if start_btn and not st.session_state.running:
    from modules.vision_module import VisionModule
    from modules.audio_module import AudioModule
    from modules.multimodal_agent import MultimodalAgent
    from modules.analytics_engine import AnalyticsEngine

    st.session_state.vision = VisionModule(presence_threshold_sec=presence_thresh)
    st.session_state.audio = AudioModule()
    st.session_state.agent = MultimodalAgent(
        expected_participants=expected, window_size=30
    )
    st.session_state.analytics = AnalyticsEngine()
    st.session_state.running = True
    st.session_state.session_start = time.time()
    st.session_state.engagement_history = []
    st.session_state.face_count_history = []
    st.session_state.speech_history = []
    st.session_state.noise_history = []
    st.session_state.time_axis = []
    st.session_state.attendance_records = []
    st.session_state.suggestions = []
    st.session_state.events = []
    st.session_state.student_analytics = []
    st.session_state.tick = 0

    if demo_mode:
        st.session_state.audio.inject_synthetic_audio(duration_sec=60, speech_ratio=0.6)
    else:
        try:
            st.session_state.audio.start_stream()
        except Exception:
            pass
    st.success("Session started!")

if stop_btn and st.session_state.running:
    st.session_state.running = False
    try:
        st.session_state.audio.stop_stream()
    except Exception:
        pass
    st.info("Session stopped.")


# ── Simulate one tick in demo mode ───────────────────────────────────────────

def simulate_tick(tick: int) -> dict:
    """Generate synthetic vision result for demo."""
    n = max(0, int(expected * 0.6 + expected * 0.4 * np.sin(tick * 0.2)
                   + np.random.randint(-1, 2)))
    persons = {}
    for i in range(n):
        persons[i] = {
            "label": f"Student {i+1:02d}",
            "present": tick > 5,
            "duration": tick * 0.5,
            "activity": float(np.clip(np.random.beta(2, 3), 0, 1)),
        }
    return {"face_count": n, "tracked_persons": persons,
            "attendance_log": [], "frame_idx": tick}


# ── Live tick update ─────────────────────────────────────────────────────────

if st.session_state.running:
    st.session_state.tick += 1
    tick = st.session_state.tick

    agent = st.session_state.agent
    audio = st.session_state.audio
    analytics_eng = st.session_state.analytics

    vision_result = simulate_tick(tick) if demo_mode else {"face_count": 0,
                                                            "tracked_persons": {},
                                                            "attendance_log": [],
                                                            "frame_idx": tick}
    audio_state = audio.get_current_state()
    state = agent.update(vision_result, audio_state)

    elapsed = round(time.time() - st.session_state.session_start, 1)
    st.session_state.engagement_history.append(state.engagement_score)
    st.session_state.face_count_history.append(state.face_count)
    st.session_state.speech_history.append(1 if state.speech_active else 0)
    st.session_state.noise_history.append(state.noise_level)
    st.session_state.time_axis.append(elapsed)

    st.session_state.current_score = state.engagement_score
    st.session_state.current_trend = state.trend
    st.session_state.current_faces = state.face_count
    st.session_state.current_speech = state.speech_active
    st.session_state.current_noise = state.noise_level
    st.session_state.last_alert = state.alert or ""

    new_suggestions = analytics_eng.generate_suggestions(
        state, agent.get_events(), agent.get_timeline(), audio_state
    )
    if new_suggestions:
        st.session_state.suggestions = new_suggestions + st.session_state.suggestions[:9]

    new_events = agent.get_events()
    st.session_state.events = new_events[-20:]

    # Synthetic attendance
    if tick == 10:
        for i in range(min(expected, 4)):
            st.session_state.attendance_records.append({
                "ID": i + 1,
                "Name": f"Student {i+1:02d}",
                "Marked At": time.strftime("%H:%M:%S"),
                "Duration (s)": round(5.0 + i * 1.2, 1),
            })

    # Auto-rerun for live updates
    time.sleep(0.3)
    st.rerun()


# ── Load report ──────────────────────────────────────────────────────────────

report_data = None
if load_report_btn and selected_report:
    report_path = ROOT / "outputs" / "reports" / selected_report
    with open(report_path) as f:
        report_data = json.load(f)


# ── Main layout ──────────────────────────────────────────────────────────────

st.markdown("# 🎓 Multimodal Attendance & Engagement Agent")
st.markdown("*AI-powered classroom intelligence — Vision · Audio · Agent Reasoning*")
st.markdown("---")

# ── KPI Row ──────────────────────────────────────────────────────────────────
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

score = st.session_state.current_score
trend_sym = {"Rising": "▲", "Falling": "▼", "Stable": "●"}.get(
    st.session_state.current_trend, "●"
)

kpi1.metric("🎯 Engagement", f"{score:.0%}", delta=st.session_state.current_trend)
kpi2.metric("👥 Faces Detected", st.session_state.current_faces)
kpi3.metric("🔊 Speech", "Active ✓" if st.session_state.current_speech else "Silent")
kpi4.metric("📢 Noise Level", f"{st.session_state.current_noise:.4f}")
kpi5.metric("⏱ Session Tick", st.session_state.tick)

st.markdown("---")

# ── Engagement Gauge + Timeline ──────────────────────────────────────────────
left_col, right_col = st.columns([1, 2])

with left_col:
    st.markdown("### 🎯 Live Engagement Score")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score * 100,
        delta={"reference": 50, "suffix": "%"},
        number={"suffix": "%", "font": {"size": 36}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#8892b0"},
            "bar": {"color": "#00d4ff", "thickness": 0.25},
            "bgcolor": "#1a1f2e",
            "borderwidth": 2,
            "bordercolor": "#2d3561",
            "steps": [
                {"range": [0, 30],  "color": "#2d0a0a"},
                {"range": [30, 60], "color": "#2d2005"},
                {"range": [60, 100],"color": "#072d1a"},
            ],
            "threshold": {
                "line": {"color": "#00d4ff", "width": 3},
                "thickness": 0.8,
                "value": score * 100,
            },
        },
        title={"text": f"Trend: {st.session_state.current_trend} {trend_sym}",
               "font": {"color": "#8892b0", "size": 14}},
    ))
    fig_gauge.update_layout(
        height=280, paper_bgcolor="#0e1117", font_color="#cdd6f4",
        margin=dict(t=30, b=10, l=20, r=20)
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    if st.session_state.last_alert:
        st.warning(f"⚠️ {st.session_state.last_alert}")

with right_col:
    st.markdown("### 📈 Engagement Timeline")
    history = st.session_state.engagement_history
    if history:
        t_axis = st.session_state.time_axis
        df_eng = pd.DataFrame({
            "Time (s)": t_axis,
            "Engagement": [s * 100 for s in history],
            "Faces": st.session_state.face_count_history,
        })
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=df_eng["Time (s)"], y=df_eng["Engagement"],
            mode="lines", name="Engagement %",
            line=dict(color="#00d4ff", width=2),
            fill="tozeroy", fillcolor="rgba(0,212,255,0.08)",
        ))
        fig_line.add_hline(y=60, line_dash="dot", line_color="#00ff88",
                           annotation_text="Target (60%)", annotation_font_color="#00ff88")
        fig_line.add_hline(y=30, line_dash="dot", line_color="#ff4d4d",
                           annotation_text="Alert (30%)", annotation_font_color="#ff4d4d")
        fig_line.update_layout(
            height=280, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            font_color="#cdd6f4", margin=dict(t=10, b=30, l=40, r=20),
            xaxis=dict(gridcolor="#2d3561"), yaxis=dict(gridcolor="#2d3561", range=[0, 105]),
            legend=dict(orientation="h", y=1.05),
        )
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("Start a session to see the engagement timeline.")

st.markdown("---")

# ── Audio + Faces Chart ──────────────────────────────────────────────────────
a_col, b_col = st.columns(2)

with a_col:
    st.markdown("### 🔊 Audio Activity")
    if st.session_state.speech_history:
        df_audio = pd.DataFrame({
            "Time (s)": st.session_state.time_axis,
            "Speech": st.session_state.speech_history,
            "Noise": st.session_state.noise_history,
        })
        fig_audio = go.Figure()
        fig_audio.add_trace(go.Bar(
            x=df_audio["Time (s)"], y=df_audio["Speech"],
            name="Speech Active", marker_color="#00ff88", opacity=0.75,
        ))
        fig_audio.add_trace(go.Scatter(
            x=df_audio["Time (s)"], y=df_audio["Noise"] * 20,
            name="Noise (×20)", line=dict(color="#ffa500", width=1.5),
        ))
        fig_audio.update_layout(
            height=220, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            font_color="#cdd6f4", margin=dict(t=10, b=30, l=40, r=20),
            xaxis=dict(gridcolor="#2d3561"), yaxis=dict(gridcolor="#2d3561"),
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig_audio, use_container_width=True)
    else:
        st.info("No audio data yet.")

with b_col:
    st.markdown("### 👥 Face Count Over Time")
    if st.session_state.face_count_history:
        df_faces = pd.DataFrame({
            "Time (s)": st.session_state.time_axis,
            "Faces": st.session_state.face_count_history,
        })
        fig_faces = px.area(
            df_faces, x="Time (s)", y="Faces",
            color_discrete_sequence=["#a855f7"],
        )
        fig_faces.update_layout(
            height=220, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            font_color="#cdd6f4", margin=dict(t=10, b=30, l=40, r=20),
            xaxis=dict(gridcolor="#2d3561"), yaxis=dict(gridcolor="#2d3561"),
        )
        st.plotly_chart(fig_faces, use_container_width=True)
    else:
        st.info("No face count data yet.")

st.markdown("---")

# ── Attendance + Suggestions ──────────────────────────────────────────────────
att_col, sug_col = st.columns([1, 1])

with att_col:
    st.markdown("### ✅ Attendance Log")
    if st.session_state.attendance_records:
        df_att = pd.DataFrame(st.session_state.attendance_records)
        st.dataframe(df_att, use_container_width=True, height=220,
                     hide_index=True)
    else:
        st.info("Attendance records will appear as participants are detected.")

with sug_col:
    st.markdown("### 💡 Teaching Suggestions")
    if st.session_state.suggestions:
        for sug in st.session_state.suggestions[:5]:
            priority_colors = {"High": "#ff4d4d", "Medium": "#ffa500", "Low": "#00d4ff"}
            color = priority_colors.get(sug.priority, "#00d4ff")
            st.markdown(
                f"""<div class="suggestion-card {sug.priority}">
                    <strong style="color:{color};">[{sug.priority}] {sug.title}</strong><br>
                    <small style="color:#8892b0;">{sug.category} · {
                        time.strftime("%H:%M:%S", time.localtime(sug.timestamp))
                    }</small><br>
                    <span style="color:#cdd6f4;">{sug.body}</span>
                </div>""",
                unsafe_allow_html=True
            )
    else:
        st.info("Suggestions will appear based on session dynamics.")

st.markdown("---")

# ── Events Log ───────────────────────────────────────────────────────────────
st.markdown("### 📋 Session Event Log")
if st.session_state.events:
    rows = []
    for e in reversed(st.session_state.events):
        rows.append({
            "Time": time.strftime("%H:%M:%S", time.localtime(e.timestamp)),
            "Event": e.event_type,
            "Severity": e.severity,
            "Description": e.description,
            "Engagement": f"{e.engagement_at_event:.0%}",
        })
    df_events = pd.DataFrame(rows)
    st.dataframe(df_events, use_container_width=True, height=180, hide_index=True)
else:
    st.info("No events logged yet.")

# ── Loaded Report View ────────────────────────────────────────────────────────
if report_data:
    st.markdown("---")
    st.markdown("## 📊 Loaded Session Report")

    r_col1, r_col2, r_col3, r_col4 = st.columns(4)
    r_col1.metric("Duration", f"{report_data.get('session_duration_sec',0):.0f}s")
    r_col2.metric("Avg Engagement", f"{report_data.get('avg_engagement',0):.0%}")
    r_col3.metric("Attendance", f"{report_data.get('attendance_count',0)}/{report_data.get('total_participants',0)}")
    r_col4.metric("Low Periods", report_data.get('low_engagement_periods', 0))

    if report_data.get("student_analytics"):
        st.markdown("#### Student Analytics")
        df_students = pd.DataFrame(report_data["student_analytics"])
        st.dataframe(df_students, use_container_width=True, hide_index=True)

    if report_data.get("suggestions"):
        st.markdown("#### Suggestions from Report")
        for s in report_data["suggestions"]:
            st.markdown(f"**[{s['priority']}] {s['title']}** — {s['body']}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center style='color:#8892b0;font-size:0.8rem;'>"
    "Multimodal Attendance & Engagement Agent · "
    "Built with ❤️ using OpenCV · MediaPipe · Librosa · Streamlit"
    "</center>",
    unsafe_allow_html=True
)
