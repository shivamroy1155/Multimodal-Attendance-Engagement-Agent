"""
Flask API Gateway & Route Handlers — Figure 1 Architecture
==========================================================
Frontend Layer / API Gateway connecting:
  - HTML5 & JavaScript Controllers
  - AI Service Layer (Google Gemini 2.0 Flash, TTS Engine, Code Gen)
  - Multimodal Agent & Computer Vision / Audio Modules
  - Storage Layer (Generated Audio, Generated Code, Uploaded Files)
"""

import os
import sys
import time
import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from modules.vision_module import VisionModule
from modules.audio_module import AudioModule, AudioState
from modules.multimodal_agent import MultimodalAgent
from modules.analytics_engine import AnalyticsEngine
from modules.genai_service import GeminiAIService
from modules.tts_engine import TTSEngine
from modules.code_executor import CodeExecutor
from modules.genai_utils import synthesize_recommendations

app = Flask(__name__, template_folder="templates", static_folder="static")

# Storage paths
STORAGE_DIR = ROOT / "storage"
STORAGE_AUDIO = STORAGE_DIR / "generated_audio"
STORAGE_CODE = STORAGE_DIR / "generated_code"
STORAGE_UPLOAD = STORAGE_DIR / "uploaded_files"

for d in [STORAGE_AUDIO, STORAGE_CODE, STORAGE_UPLOAD]:
    d.mkdir(parents=True, exist_ok=True)

# Instantiate Core System Components
vision_module = VisionModule(presence_threshold_sec=3.0, fps_hint=10)
audio_module = AudioModule()
multimodal_agent = MultimodalAgent(expected_participants=5, window_size=25)
analytics_engine = AnalyticsEngine()
gemini_service = GeminiAIService()
tts_engine = TTSEngine()
code_executor = CodeExecutor()

# Pre-inject synthetic audio so demo works right out of the box
audio_module.inject_synthetic_audio(duration_sec=30.0, speech_ratio=0.5)

# Session tracking
session_state = {
    "tick": 0,
    "start_time": time.time(),
    "attendance": [
        {"id": 1, "name": "Student 01 (Aarav)", "duration": "14.2s", "status": "Present"},
        {"id": 2, "name": "Student 02 (Priya)", "duration": "12.8s", "status": "Present"},
        {"id": 3, "name": "Student 03 (Rohan)", "duration": "9.5s", "status": "Present"},
    ],
    "history": [],
}


@app.route("/")
def index():
    """Renders the HTML5 Frontend Dashboard (Frontend Layer)."""
    return render_template("index.html", is_gemini_online=gemini_service.is_online)


@app.route("/api/status")
def get_status():
    """Returns the current state of the Multimodal Agent."""
    audio_state = audio_module.get_current_state()
    agent_state = multimodal_agent.get_session_state()
    return jsonify({
        "tick": session_state["tick"],
        "engagement_score": round(agent_state.engagement_score, 3),
        "trend": agent_state.trend,
        "face_count": agent_state.face_count,
        "speech_active": agent_state.speech_active,
        "noise_level": round(agent_state.noise_level, 4),
        "inactivity_duration": agent_state.inactivity_duration,
        "is_gemini_online": gemini_service.is_online,
        "model_name": gemini_service.model_name,
    })


@app.route("/api/tick", methods=["POST"])
def tick():
    """Advances one processing tick — simulates or analyzes vision & audio signals."""
    session_state["tick"] += 1
    t = session_state["tick"]
    
    # Simulate realistic classroom dynamics
    import numpy as np
    n_faces = max(1, int(3 + 2 * np.sin(t * 0.25) + np.random.randint(-1, 2)))
    tracked = {
        i: {
            "label": f"Student {i+1:02d}",
            "present": t > 3,
            "activity": float(np.clip(np.random.beta(2, 3), 0.1, 0.9)),
        } for i in range(n_faces)
    }
    
    vision_result = {
        "face_count": n_faces,
        "tracked_persons": tracked,
        "attendance_log": [],
        "frame_idx": t,
    }
    
    audio_state = audio_module.get_current_state()
    state = multimodal_agent.update(vision_result, audio_state)
    
    session_state["history"].append(state.engagement_score)
    if len(session_state["history"]) > 40:
        session_state["history"].pop(0)

    # Dynamic attendance marking
    if t == 5 and len(session_state["attendance"]) < 4:
        session_state["attendance"].append({
            "id": 4, "name": "Student 04 (Ananya)", "duration": "5.1s", "status": "Present"
        })

    return jsonify({
        "tick": t,
        "engagement_score": state.engagement_score,
        "trend": state.trend,
        "face_count": state.face_count,
        "speech_active": state.speech_active,
        "noise_level": state.noise_level,
        "history": session_state["history"],
        "alert": state.alert,
    })


@app.route("/api/gemini/suggest", methods=["POST"])
def gemini_suggest():
    """AI Service Layer: Triggers Google Gemini 2.0 Flash teaching suggestions."""
    data = request.get_json(silent=True) or {}
    metrics = {
        "engagement_score": multimodal_agent.get_engagement_score(),
        "trend": multimodal_agent.get_session_state().trend,
        "face_count": multimodal_agent.get_session_state().face_count,
        "speech_active": multimodal_agent.get_session_state().speech_active,
        "inactivity_duration": multimodal_agent.get_session_state().inactivity_duration,
    }
    if data:
        metrics.update(data)

    gemini_insight = gemini_service.generate_teaching_insight(metrics)
    rule_suggs = analytics_engine.generate_suggestions(
        multimodal_agent.get_session_state(),
        multimodal_agent.get_events(),
        multimodal_agent.get_timeline(),
        audio_module.get_current_state()
    )
    combined = synthesize_recommendations(rule_suggs, gemini_insight)
    return jsonify({
        "status": "success",
        "gemini_insight": gemini_insight,
        "all_recommendations": combined,
    })


@app.route("/api/tts/alert", methods=["POST"])
def tts_alert():
    """AI Service Layer: Synthesizes spoken audio alert into storage/generated_audio/."""
    data = request.get_json(silent=True) or {}
    text = data.get("text", "Attention: Classroom engagement has dropped. Consider an interactive pause.")
    audio_path = tts_engine.synthesize(text)
    filename = Path(audio_path).name
    return jsonify({
        "status": "success",
        "text": text,
        "audio_url": f"/storage/generated_audio/{filename}",
        "filename": filename
    })


@app.route("/api/code/generate", methods=["POST"])
def generate_and_execute_code():
    """AI Service Layer: Generates code with Gemini Code Engine & executes safely."""
    metrics = {
        "engagement_score": multimodal_agent.get_engagement_score(),
        "face_count": multimodal_agent.get_session_state().face_count,
        "inactivity_duration": multimodal_agent.get_session_state().inactivity_duration,
    }
    code = gemini_service.generate_analytics_code(metrics)
    exec_result = code_executor.save_and_execute(code, filename_prefix="gemini_analytics")
    return jsonify({
        "status": "success",
        "generated_code": code,
        "execution": exec_result,
    })


@app.route("/api/upload", methods=["POST"])
def upload_file():
    """Storage Layer: Handles uploaded audio/video media files."""
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    filename = f"{int(time.time())}_{file.filename}"
    save_path = STORAGE_UPLOAD / filename
    file.save(str(save_path))

    return jsonify({
        "status": "uploaded",
        "filename": filename,
        "file_url": f"/storage/uploaded_files/{filename}"
    })


@app.route("/api/attendance")
def get_attendance():
    """Returns real-time sustained presence attendance records."""
    return jsonify({"attendance": session_state["attendance"]})


@app.route("/api/report")
def get_report():
    """Returns comprehensive end-of-session analytics report."""
    summary = multimodal_agent.get_session_summary()
    report = analytics_engine.generate_session_report(
        agent_summary=summary,
        engagement_history=multimodal_agent.get_timeline(),
        session_state=multimodal_agent.get_session_state(),
        attendance_log=[],
        tracked_persons={},
    )
    return jsonify({
        "summary": summary,
        "report_text": report.summary_text,
        "avg_engagement": report.avg_engagement,
        "total_participants": report.total_participants,
    })


@app.route("/storage/<folder>/<filename>")
def serve_storage(folder, filename):
    """Serves generated audio files, generated code, and uploaded files."""
    target_dir = STORAGE_DIR / folder
    if not target_dir.exists():
        return jsonify({"error": "Folder not found"}), 404
    return send_from_directory(str(target_dir), filename)


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  Multimodal Attendance & Engagement Agent — Flask API Gateway")
    print("  Architecture: Frontend Layer + AI Service Layer (Gemini 2.0 Flash)")
    print("  Open in your web browser: http://127.0.0.1:5000")
    print("=" * 65 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
