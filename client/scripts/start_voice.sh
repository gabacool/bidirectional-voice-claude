#!/bin/bash
# Start voice recording - called by Hammerspoon

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLIENT_DIR="$(dirname "$SCRIPT_DIR")"
PID_FILE="/tmp/parakeet_voice.pid"
LOG_FILE="/tmp/parakeet_voice.log"

# Check if already recording
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Already recording (PID $OLD_PID)"
        exit 1
    fi
fi

cd "$CLIENT_DIR"
source "$CLIENT_DIR/../venv/bin/activate"

# Start recording in background
python voice_client.py > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

echo "Recording started (PID $(cat $PID_FILE))"
