# 🎓 Multimodal Attendance & Engagement Agent

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0+-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-0097A7?logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![Librosa](https://img.shields.io/badge/Librosa-0.10-FF6F00)](https://librosa.org)
[![Google Gemini](https://img.shields.io/badge/Google_Gemini-2.0_Flash-4285F4?logo=google&logoColor=white)](https://ai.google.dev/)
[![Tests](https://img.shields.io/badge/Unit_Tests-34%20Passed-10b981)]()

> **Capstone Project:** Generative AI with Google Cloud &bull; SkillWallet  
> **Student / Assignee:** Shivam Kumar Roy  
> **Category:** Artificial Intelligence / Computer Vision / Multimodal Audio & Vision Agent  
> **Execution Environment:** Offline / Local Execution (Privacy-Preserving) + Google Gemini 2.0 Flash AI Service

---

## 📌 Submission Checklist for Mentor Evaluation (SkillWallet)
When submitting on **SkillWallet**:
1. **GitHub Link:** Push this codebase to your GitHub repository and paste the link into the **[Add Github Link]** button on SkillWallet.
2. **Demo Link:** Run either the **Web App (`python app.py`)** or **Streamlit Dashboard (`streamlit run dashboard.py`)**, record a 1-2 min video or deploy to Hugging Face / Streamlit Community Cloud / Render, and paste the link into **[Add Demo Link]**.

---

## 🧠 Project Overview

In traditional classrooms and remote sessions, attendance tracking is manual or limited to one-frame biometrics that fail to capture meaningful learner attention, interaction trends, or session pacing.

The **Multimodal Attendance & Engagement Agent** was engineered as an AI-powered, privacy-preserving classroom intelligence system. It combines:
- 🎥 **Vision Module (OpenCV & MediaPipe):** Multi-person face tracking, sustained presence verification, and head-pose activity estimation.
- 🔊 **Audio Module (Librosa & NumPy):** Real-time RMS energy, Zero-Crossing Rate (ZCR), spectral centroid, and noise floor estimation.
- 🤖 **Multimodal Agent:** Mathematical signal fusion, temporal EMA smoothing, and rule-based anomaly detection.
- ⚡ **AI Service Layer (Google Gemini 2.0 Flash):** Real-time pedagogical recommendations, Text-to-Speech audio alerts, and automated analytics code generation.
- 📊 **Interactive Interfaces:**
  - **Flask API Gateway & Web UI** (Architecture as per Figure 1)
  - **Streamlit Analytics Dashboard** (as per Course Instructions)

---

## 🏗️ Technical Architecture (Figure 1 Alignment)

```
┌────────────────────────────────────────────────────────────────────────┐
│                        FRONTEND LAYER                                   │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────────────┐  │
│  │ HTML5 Templates  │  │   CSS3 Styling   │  │ JavaScript Controllers│  │
│  └─────────┬────────┘  └─────────┬────────┘  └───────────┬───────────┘  │
│            └─────────────────────┼───────────────────────┘              │
│                                  ▼                                      │
│                      ┌───────────────────────┐                          │
│                      │   Flask API Gateway   │                          │
│                      └───────────┬───────────┘                          │
│                                  ▼                                      │
│                      ┌───────────────────────┐                          │
│                      │     Route Handlers    │                          │
│                      └───────────┬───────────┘                          │
│                                  ▼                                      │
│                      ┌───────────────────────┐                          │
│                      │      GenAI Utils      │                          │
│                      └───────────┬───────────┘                          │
│                                  ▼                                      │
│                      ┌───────────────────────┐                          │
│                      │      Audio Utils      │                          │
│                      └───────────┬───────────┘                          │
│                                  ▼                                      │
│                      ┌───────────────────────┐                          │
│                      │     Code Executor     │                          │
│                      └───────────┬───────────┘                          │
└──────────────────────────────────┼──────────────────────────────────────┘
                                   │
            ┌──────────────────────┴──────────────────────┐
            ▼                                             ▼
┌───────────────────────────┐                 ┌───────────────────────────┐
│     AI SERVICE LAYER      │                 │       STORAGE LAYER       │
│ • Google Gemini 2.0 Flash │                 │ • Generated Audio Files   │
│ • Gemini Image Generation │                 │ • Generated Code Files    │
│ • Text-to-Speech Engine   │                 │ • Uploaded Files          │
│ • Code Generation Engine  │                 │ • Local Browser Storage   │
└───────────────────────────┘                 └───────────────────────────┘
```

---

## 🎯 Milestones & Tasks Completed

### Milestone 1: Requirements Specification and Environment Setup
- **Activity 1.1:** Created `requirements.txt` with all vision, audio, GenAI, and dashboard dependencies.
- **Activity 1.2:** Created `setup_env.py` automated dependency installer and environment verification script.

### Milestone 2: Vision Module Development
- **Activity 2.1:** Face detection with MediaPipe/OpenCV and multi-person Centroid Tracking.
- **Activity 2.2:** Presence-based attendance logic requiring sustained presence window (e.g. 5s) to eliminate false positives from transitory entries.

### Milestone 3: Audio Module Development
- **Activity 3.1:** Audio signal extraction (microphone live stream, audio file input, or synthetic classroom simulation).
- **Activity 3.2:** RMS energy calculation, Zero Crossing Rate (ZCR), noise floor tracking, and speech activity detection.

### Milestone 4: Multimodal Agent Implementation
- **Activity 4.1:** Multimodal signal fusion integrating vision presence (30%), head attention (25%), audio speech activity (25%), and activity penalty (20%).
- **Activity 4.2:** Composite engagement scoring logic, exponential moving average (EMA) smoothing, and trend estimation (Rising/Falling/Stable).

### Milestone 5: Analytics and Recommendation Engine
- **Activity 5.1:** Context-aware teaching suggestions (Icebreakers, Pacing checks, Re-engagement alerts, Discussion prompts).
- **Activity 5.2:** Per-student analytics generation, attendance duration logs, and structured session reports.

### Milestone 6: Interface, Testing, and Deployment
- **Activity 6.1:** Core CLI pipeline (`python main.py --demo` or `python main.py`).
- **Activity 6.2:** Interactive Web & Streamlit Dashboards (`python app.py` and `streamlit run dashboard.py`).
- **Activity 6.3:** Complete validation test suite (`python tests/test_all.py` — **34/34 tests passed**).

---

## 📁 Repository Structure

```
Multimodel/
│
├── app.py                      # Flask API Gateway & Web Application (Fig 1)
├── main.py                     # Milestone 6.1 — Core pipeline (Webcam/Video/Demo)
├── dashboard.py                # Milestone 6.2 — Streamlit Dashboard
├── setup_env.py                # Milestone 1.2 — Automated environment setup
├── requirements.txt            # Milestone 1.1 — Dependency list
├── .gitignore                  # Clean Git repository configuration
├── README.md                   # Complete documentation
│
├── modules/
│   ├── __init__.py
│   ├── vision_module.py        # Milestone 2 — Face tracking & attendance
│   ├── audio_module.py         # Milestone 3 — Audio extraction & speech detection
│   ├── multimodal_agent.py     # Milestone 4 — Multimodal signal fusion & scoring
│   ├── analytics_engine.py    # Milestone 5 — Teaching suggestions & student reports
│   ├── genai_service.py       # AI Service Layer — Google Gemini 2.0 Flash
│   ├── tts_engine.py          # AI Service Layer — Text-to-Speech Engine
│   ├── code_executor.py       # Utility Layer — Code Execution Engine
│   ├── genai_utils.py         # Utility Layer — GenAI prompt formatting
│   └── audio_utils.py         # Utility Layer — Signal & WAV utilities
│
├── templates/
│   └── index.html             # Frontend Layer — HTML5 Dashboard Template
│
├── static/
│   ├── css/style.css          # Frontend Layer — CSS3 Styling (Dark Mode)
│   └── js/app.js              # Frontend Layer — JavaScript Controllers
│
├── storage/                    # Storage Layer
│   ├── generated_audio/       # Spoken alerts from TTS Engine
│   ├── generated_code/        # Scripts generated by Code Generation Engine
│   └── uploaded_files/        # User-uploaded audio/video
│
├── outputs/
│   ├── logs/                  # Timestamped CSV attendance logs & session logs
│   └── reports/               # JSON and formatted text analytics reports
│
└── tests/
    └── test_all.py            # Milestone 6.3 — 34 Unit & Integration Tests
```

---

## 🚀 How to Run the Project

### 1. Quick Setup
```bash
# Navigate to the project directory
cd Multimodel

# Run setup verification
python setup_env.py
```

### 2. Launch the Flask Web Application (Figure 1 Architecture)
```bash
python app.py
```
Open **`http://127.0.0.1:5000`** in your browser.
- Real-time multimodal video monitor with HUD.
- Live engagement gauge & Chart.js timeline.
- Google Gemini 2.0 Flash suggestion engine.
- TTS voice alerts & Code Generation Executor.
- Automated sustained presence attendance table.

### 3. Launch the Streamlit Dashboard (Course Requirement)
```bash
streamlit run dashboard.py
```

### 4. Run Core Application via CLI
```bash
# Demo mode (synthetic simulation — no webcam needed):
python main.py --demo

# Live webcam mode:
python main.py

# Video file mode:
python main.py --source classroom.mp4 --audio discussion.wav
```

### 5. Run the Validation Test Suite
```bash
python tests/test_all.py
```
*(All 34 test cases validate trackers, vision logic, audio processing, fusion agent, Gemini AI service, TTS engine, and Flask gateway).*

---

## 🔑 Google Gemini 2.0 Flash Configuration (Optional)
The project is built to run **100% offline** by default using intelligent local rule engines.  
To enable online Google Gemini 2.0 Flash features:
```bash
# Windows PowerShell:
$env:GEMINI_API_KEY="your-google-gemini-api-key"

# Windows CMD:
set GEMINI_API_KEY=your-google-gemini-api-key

# Linux/macOS:
export GEMINI_API_KEY="your-google-gemini-api-key"
```

---

## 🎓 Conclusion

The **Multimodal Attendance & Engagement Agent** demonstrates the seamless integration of computer vision, audio signal processing, rule-based agent reasoning, and Google Gemini 2.0 Flash in educational technology. By marking attendance through sustained presence rather than fleeting single frames and fusing acoustic and visual cues, it empowers educators with real-time, privacy-preserving classroom insights.
