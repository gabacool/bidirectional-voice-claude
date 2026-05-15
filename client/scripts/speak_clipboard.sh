#!/bin/bash
# Speak clipboard text via TTS daemon
# Called by Hammerspoon on Option+S
# Works from any app — copy text, press Option+S
#
# Daemon reads clipboard at processing time (always fresh).
# First press auto-starts daemon (~30s model load).
# Pressing Option+S while speaking interrupts and speaks new clipboard.

# Hammerspoon doesn't inherit user's shell PATH — add Homebrew
export PATH="/opt/homebrew/bin:$PATH"
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$(dirname "$PROJECT_DIR")/venv"

DAEMON_PORT=8089
DAEMON_URL="http://127.0.0.1:$DAEMON_PORT"

# Fast path: daemon already running
if curl -s --max-time 1 "$DAEMON_URL/health" > /dev/null 2>&1; then
    curl -s -X POST "$DAEMON_URL/speak" >> /tmp/tts_debug.log 2>&1
    exit 0
fi

# Slow path: start daemon, wait for model to load, then speak
echo "$(date): Starting TTS daemon..." >> /tmp/tts_debug.log
source "$VENV_DIR/bin/activate"
python "$PROJECT_DIR/tts_daemon.py" >> /tmp/tts_daemon.log 2>&1 &

for i in $(seq 1 120); do
    if curl -s --max-time 1 "$DAEMON_URL/health" > /dev/null 2>&1; then
        echo "$(date): Daemon ready" >> /tmp/tts_debug.log
        curl -s -X POST "$DAEMON_URL/speak" >> /tmp/tts_debug.log 2>&1
        exit 0
    fi
    sleep 0.5
done

echo "$(date): Daemon failed to start within 60s" >> /tmp/tts_debug.log
exit 1
