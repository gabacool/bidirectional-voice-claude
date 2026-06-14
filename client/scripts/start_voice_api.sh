#!/bin/bash
# Start the LAN voice API (STT /transcribe + TTS /synthesize) for remote agents.
# Bound to 0.0.0.0:9900 (LAN only). Run by the LaunchAgent, or manually.

export PATH="/opt/homebrew/bin:$PATH"
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"          # .../client
VENV_DIR="$(dirname "$PROJECT_DIR")/venv"       # .../venv

source "$VENV_DIR/bin/activate"
exec python "$PROJECT_DIR/voice_api.py"
