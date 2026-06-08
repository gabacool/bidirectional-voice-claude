#!/bin/bash
# Speak selected text via TTS daemon — called by Hammerspoon on Option+S
#
# Text is passed as $1 (the selected text, captured by Hammerspoon without
# polluting the clipboard) and sent to the daemon as the POST body. If no
# argument is given, the daemon falls back to reading the clipboard.
#
# First press auto-starts the daemon (~30s model load).
# Option+S is a toggle: speak (when idle) / pause / resume. Option+X stops.

# Hammerspoon doesn't inherit user's shell PATH — add Homebrew
export PATH="/opt/homebrew/bin:$PATH"
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$(dirname "$PROJECT_DIR")/venv"

DAEMON_PORT=8089
DAEMON_URL="http://127.0.0.1:$DAEMON_PORT"
TEXT="$1"

post_speak() {
    if [ -n "$TEXT" ]; then
        # --data-raw sends the body verbatim (no '@file' interpretation) and
        # lets curl set Content-Length, which the daemon needs to read the body.
        curl -s -X POST "$DAEMON_URL/speak" --data-raw "$TEXT" >> /tmp/tts_debug.log 2>&1
    else
        # No selection passed — daemon reads the clipboard itself.
        curl -s -X POST "$DAEMON_URL/speak" >> /tmp/tts_debug.log 2>&1
    fi
}

# Fast path: daemon already running
if curl -s --max-time 1 "$DAEMON_URL/health" > /dev/null 2>&1; then
    post_speak
    exit 0
fi

# Slow path: start daemon, wait for model to load, then speak
echo "$(date): Starting TTS daemon..." >> /tmp/tts_debug.log
source "$VENV_DIR/bin/activate"
python "$PROJECT_DIR/tts_daemon.py" >> /tmp/tts_daemon.log 2>&1 &

for i in $(seq 1 120); do
    if curl -s --max-time 1 "$DAEMON_URL/health" > /dev/null 2>&1; then
        echo "$(date): Daemon ready" >> /tmp/tts_debug.log
        post_speak
        exit 0
    fi
    sleep 0.5
done

echo "$(date): Daemon failed to start within 60s" >> /tmp/tts_debug.log
exit 1
