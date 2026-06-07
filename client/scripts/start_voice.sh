#!/bin/bash
# Start voice recording - called by Hammerspoon

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLIENT_DIR="$(dirname "$SCRIPT_DIR")"
PID_FILE="/tmp/parakeet_voice.pid"
LOG_FILE="/tmp/parakeet_voice.log"

# Kill any existing/orphaned recorder so we never hold two mic streams or
# leave one running forever (e.g. after a Hammerspoon reload lost the stop).
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Stopping stale recorder (PID $OLD_PID)"
        kill -USR1 "$OLD_PID" 2>/dev/null
        sleep 0.3
        kill -9 "$OLD_PID" 2>/dev/null
    fi
    rm -f "$PID_FILE"
fi
# Belt-and-suspenders: sweep any other voice_client.py the pid file missed.
pkill -f "voice_client.py" 2>/dev/null

cd "$CLIENT_DIR"
source "$CLIENT_DIR/../venv/bin/activate"

# Start recording in background
python voice_client.py > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

echo "Recording started (PID $(cat $PID_FILE))"
