#!/usr/bin/env python3
"""
Environment Setup Script — Milestone 1, Activity 1.2
=====================================================
Run this script to verify your Python environment and install
all required dependencies for the Multimodal Attendance & Engagement Agent.

Usage:
    python setup_env.py
"""

import subprocess
import sys
import platform

REQUIRED_PYTHON = (3, 9)

def check_python():
    v = sys.version_info
    print(f"  Python: {sys.version}")
    if v < REQUIRED_PYTHON:
        print(f"  ✗ Python {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}+ required.")
        sys.exit(1)
    print(f"  ✓ Python version OK")

def install_requirements():
    print("\n  Installing dependencies from requirements.txt …")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt",
         "--upgrade", "--quiet"],
        capture_output=False,
    )
    if result.returncode == 0:
        print("  ✓ All packages installed successfully.")
    else:
        print("  ⚠ Some packages may have failed. Check output above.")

def verify_imports():
    print("\n  Verifying critical imports …")
    checks = [
        ("numpy",       "numpy"),
        ("pandas",      "pandas"),
        ("cv2",         "opencv-python"),
        ("mediapipe",   "mediapipe"),
        ("librosa",     "librosa"),
        ("streamlit",   "streamlit"),
        ("plotly",      "plotly"),
        ("scipy",       "scipy"),
        ("sounddevice", "sounddevice  (optional — mic input)"),
        ("soundfile",   "soundfile"),
    ]
    for module, pkg in checks:
        try:
            __import__(module)
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ✗ {pkg}  ← not installed")

def print_run_instructions():
    print("\n" + "=" * 58)
    print("  SETUP COMPLETE — Run the project:")
    print("=" * 58)
    print()
    print("  1. Demo mode (no camera needed):")
    print("     python main.py --demo")
    print()
    print("  2. Webcam mode:")
    print("     python main.py")
    print()
    print("  3. Video file mode:")
    print("     python main.py --source path/to/video.mp4")
    print()
    print("  4. Streamlit Dashboard:")
    print("     streamlit run dashboard.py")
    print()
    print("  5. Run tests:")
    print("     python tests/test_all.py")
    print("=" * 58)

if __name__ == "__main__":
    print("=" * 58)
    print("  Multimodal Attendance & Engagement Agent")
    print("  Environment Setup — Milestone 1, Activity 1.2")
    print("=" * 58)
    print(f"\n  OS: {platform.system()} {platform.release()}")
    check_python()
    install_requirements()
    verify_imports()
    print_run_instructions()
