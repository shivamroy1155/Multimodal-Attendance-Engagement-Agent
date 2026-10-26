"""
Main Application Entry Point — Milestone 6, Activity 6.1
=======================================================
Run the Multimodal Attendance & Engagement Agent from webcam or video file.

Usage:
    python main.py                          # Webcam mode
    python main.py --source video.mp4       # Video file mode
    python main.py --source video.mp4 --audio audio.wav
    python main.py --demo                   # Demo mode (no camera needed)
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Force UTF-8 output on Windows to avoid cp1252 emoji errors
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Project imports ──────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from modules.vision_module import VisionModule
from modules.audio_module import AudioModule, AudioState
from modules.multimodal_agent import MultimodalAgent
from modules.analytics_engine import AnalyticsEngine

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "outputs" / "logs" / "session.log", mode="w"),
    ],
)
logger = logging.getLogger("main")

OUTPUT_DIR = ROOT / "outputs"
LOGS_DIR = OUTPUT_DIR / "logs"
REPORTS_DIR = OUTPUT_DIR / "reports"


# ── Argument parsing ─────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Multimodal Attendance & Engagement Agent"
    )
    parser.add_argument("--source", default="0",
                        help="Video source: 0=webcam, or path to video file")
    parser.add_argument("--audio", default=None,
                        help="Optional audio file to process alongside video")
    parser.add_argument("--expected", type=int, default=5,
                        help="Expected number of participants")
    parser.add_argument("--threshold", type=float, default=5.0,
                        help="Seconds of presence required to mark attendance")
    parser.add_argument("--demo", action="store_true",
                        help="Run in demo mode (synthetic data, no camera required)")
    parser.add_argument("--no-display", action="store_true",
                        help="Suppress video window (headless mode)")
    parser.add_argument("--save-video", default=None,
                        help="Path to save annotated video output")
    return parser.parse_args()


# ── Report writers ───────────────────────────────────────────────────────────

def save_attendance_csv(attendance_log, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["TrackID", "Label", "MarkedAt", "DurationBeforeMark_sec"])
        for rec in attendance_log:
            writer.writerow([
                rec.track_id, rec.label,
                time.strftime("%H:%M:%S", time.localtime(rec.marked_at)),
                rec.duration_before_mark,
            ])
    logger.info(f"Attendance log saved -> {path}")


def save_session_report(report, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    report_dict = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(report.generated_at)),
        "session_duration_sec": report.session_duration_sec,
        "total_participants": report.total_participants,
        "attendance_count": report.attendance_count,
        "avg_engagement": report.avg_engagement,
        "engagement_trend": report.engagement_trend,
        "low_engagement_periods": report.low_engagement_periods,
        "suggestions": [
            {"title": s.title, "body": s.body, "priority": s.priority, "category": s.category}
            for s in report.suggestions
        ],
        "student_analytics": [
            {
                "label": s.label,
                "present": s.marked_present,
                "activity": s.avg_activity_score,
                "participation": s.participation_level,
                "rating": s.engagement_rating,
                "notes": s.notes,
            }
            for s in report.student_analytics
        ],
    }
    with open(path.with_suffix(".json"), "w") as f:
        json.dump(report_dict, f, indent=2)
    with open(path.with_suffix(".txt"), "w") as f:
        f.write(report.summary_text)
    logger.info(f"Session report saved -> {path.with_suffix('.json')}")
    logger.info(f"Summary text saved   -> {path.with_suffix('.txt')}")


# ── Demo mode ────────────────────────────────────────────────────────────────

def run_demo_mode():
    """
    Demo mode — simulates a classroom session with synthetic data.
    No webcam or audio hardware required.
    """
    logger.info("=" * 60)
    logger.info("  DEMO MODE — Synthetic Classroom Simulation")
    logger.info("=" * 60)

    vision = VisionModule(presence_threshold_sec=2.0, fps_hint=10)
    audio = AudioModule()
    agent = MultimodalAgent(expected_participants=5, window_size=20)
    analytics = AnalyticsEngine()

    # Inject synthetic audio
    audio.inject_synthetic_audio(duration_sec=30.0, speech_ratio=0.55)

    # Simulate 30 ticks of synthetic session data
    for tick in range(60):
        # Synthetic vision result
        n_faces = max(0, int(3 + 2 * np.sin(tick * 0.3) + np.random.randint(-1, 2)))
        tracked_persons = {}
        for i in range(n_faces):
            tracked_persons[i] = {
                "label": f"Student {i+1:02d}",
                "present": tick > 5,
                "duration": tick * 0.5,
                "activity": float(np.clip(np.random.beta(2, 3), 0, 1)),
            }

        vision_result = {
            "face_count": n_faces,
            "tracked_persons": tracked_persons,
            "attendance_log": [],
            "frame_idx": tick,
        }

        audio_state = audio.get_current_state()
        state = agent.update(vision_result, audio_state)
        suggestions = analytics.generate_suggestions(
            state, agent.get_events(), agent.get_timeline(), audio_state
        )

        if tick % 10 == 0:
            logger.info(
                f"Tick {tick:3d} | Faces: {n_faces} | "
                f"Engagement: {state.engagement_score:.2f} | "
                f"Trend: {state.trend} | "
                f"Speech: {'YES' if state.speech_active else 'no '}"
            )
        if suggestions:
            for s in suggestions:
                logger.info(f"  [TIP][{s.priority}] {s.title}")

        time.sleep(0.05)

    # Generate synthetic attendance
    class FakeRecord:
        def __init__(self, tid, label, dur):
            self.track_id = tid
            self.label = label
            self.marked_at = time.time()
            self.duration_before_mark = dur

    fake_attendance = [FakeRecord(i, f"Student {i+1:02d}", 5.0 + i) for i in range(4)]

    # Final report
    summary = agent.get_session_summary()
    report = analytics.generate_session_report(
        agent_summary=summary,
        engagement_history=agent.get_timeline(),
        session_state=agent.get_session_state(),
        attendance_log=fake_attendance,
        tracked_persons={},
    )
    print("\n" + report.summary_text)

    ts = time.strftime("%Y%m%d_%H%M%S")
    save_attendance_csv(fake_attendance, LOGS_DIR / f"attendance_{ts}.csv")
    save_session_report(report, REPORTS_DIR / f"report_{ts}")
    logger.info("Demo session complete.")


# ── Live / Video mode ────────────────────────────────────────────────────────

def run_live_mode(args):
    """Milestone 6, Activity 6.1 — Run core application on webcam or video file."""

    # ── Initialise modules ──
    vision = VisionModule(
        presence_threshold_sec=args.threshold,
        fps_hint=15,
        use_face_mesh=True,
    )
    audio = AudioModule()
    agent = MultimodalAgent(
        expected_participants=args.expected,
        window_size=30,
    )
    analytics = AnalyticsEngine()

    # ── Open video source ──
    source = 0 if args.source == "0" else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Cannot open video source: {source}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Video source opened: {source} | {frame_w}x{frame_h} @ {fps:.1f} FPS")

    # ── Audio setup ──
    if args.audio:
        logger.info(f"Processing audio file: {args.audio}")
        audio.process_file(args.audio)
    else:
        # Try live mic capture; silently skip if not available
        try:
            audio.start_stream()
        except Exception as e:
            logger.warning(f"Mic capture unavailable: {e}")

    # ── Optional video writer ──
    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save_video, fourcc, fps, (frame_w, frame_h))

    # ── Main loop ──
    AGENT_TICK_EVERY = max(1, int(fps / 5))   # update agent ~5x per second
    suggestion_cooldown = 0
    last_suggestions = []
    frame_idx = 0
    session_start = time.time()

    logger.info("Session started. Press Q to quit.")
    print("\n  +=============================================+")
    print("  |  Multimodal Attendance & Engagement Agent  |")
    print("  |  Press  Q  to end session                  |")
    print("  +=============================================+\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video stream.")
                break

            frame_idx += 1

            # ── Vision processing ──
            vision_result = vision.process_frame(frame)
            annotated = vision_result["frame"]

            # ── Agent update (throttled) ──
            state = None
            if frame_idx % AGENT_TICK_EVERY == 0:
                audio_state = audio.get_current_state()
                state = agent.update(vision_result, audio_state)

                # Analytics / suggestions every 10 agent ticks
                suggestion_cooldown += 1
                if suggestion_cooldown >= 10:
                    suggestion_cooldown = 0
                    new_suggestions = analytics.generate_suggestions(
                        state, agent.get_events(),
                        agent.get_timeline(), audio_state
                    )
                    if new_suggestions:
                        last_suggestions = new_suggestions
                        for s in new_suggestions:
                            logger.info(f"  💡 [{s.priority}] {s.title}: {s.body[:80]}…")

            # ── HUD overlay ──
            if state:
                _draw_hud(annotated, state, last_suggestions)

            # ── Display ──
            if not args.no_display:
                cv2.imshow("Multimodal Attendance & Engagement Agent", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    logger.info("User requested quit.")
                    break

            if writer:
                writer.write(annotated)

    finally:
        cap.release()
        audio.stop_stream()
        if writer:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    # ── Final report ──
    logger.info("Generating session report…")
    summary = agent.get_session_summary()
    report = analytics.generate_session_report(
        agent_summary=summary,
        engagement_history=agent.get_timeline(),
        session_state=agent.get_session_state(),
        attendance_log=vision.get_attendance_log(),
        tracked_persons={},
        vision_module=vision,
        audio_state=audio.get_current_state(),
    )
    print("\n" + report.summary_text)

    ts = time.strftime("%Y%m%d_%H%M%S")
    save_attendance_csv(vision.get_attendance_log(), LOGS_DIR / f"attendance_{ts}.csv")
    save_session_report(report, REPORTS_DIR / f"report_{ts}")
    logger.info(f"Session ended. Duration: {time.time() - session_start:.0f}s")


def _draw_hud(frame: np.ndarray, state, suggestions: list):
    """Overlay live engagement HUD on the annotated frame."""
    h, w = frame.shape[:2]

    # Right panel background
    panel_w = 260
    overlay = frame.copy()
    cv2.rectangle(overlay, (w - panel_w, 0), (w, 200), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    score = state.engagement_score
    trend_sym = {"Rising": "▲", "Falling": "▼", "Stable": "●"}.get(state.trend, "●")
    color = (0, 220, 80) if score > 0.6 else (0, 165, 255) if score > 0.35 else (0, 50, 230)

    cv2.putText(frame, "Engagement", (w - panel_w + 8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(frame, f"{score:.0%}  {trend_sym}", (w - panel_w + 8, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)

    # Bar
    bar_w = panel_w - 20
    cv2.rectangle(frame, (w - panel_w + 8, 60), (w - 12, 75), (60, 60, 60), -1)
    cv2.rectangle(frame, (w - panel_w + 8, 60),
                  (w - panel_w + 8 + int(bar_w * score), 75), color, -1)

    cv2.putText(frame, f"Faces: {state.face_count}", (w - panel_w + 8, 98),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)
    cv2.putText(frame, f"Speech: {'Active' if state.speech_active else 'Silent'}",
                (w - panel_w + 8, 116),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)
    cv2.putText(frame, f"Noise: {state.noise_level:.4f}", (w - panel_w + 8, 134),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

    # Alert
    if state.alert:
        text = state.alert[:38]
        cv2.putText(frame, f"! {text}", (w - panel_w + 8, 158),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 80, 255), 1)

    # Suggestions ticker
    if suggestions:
        s = suggestions[0]
        cv2.rectangle(frame, (0, h - 36), (w, h), (20, 20, 20), -1)
        cv2.putText(frame, f"TIP [{s.priority}]: {s.title}",
                    (8, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 220), 1)


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    if args.demo:
        run_demo_mode()
    else:
        run_live_mode(args)
