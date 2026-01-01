#!/bin/bash
# Stop voice recording and trigger paste - called by Hammerspoon

PID_FILE="/tmp/parakeet_voice.pid"
LOG_FILE="/tmp/parakeet_voice.log"

if [ ! -f "$PID_FILE" ]; then
    echo "Not recording"
    exit 1
fi

PID=$(cat "$PID_FILE")

if kill -0 "$PID" 2>/dev/null; then
    # Send SIGUSR1 to stop recording gracefully
    kill -USR1 "$PID"

    # Wait for process to finish (up to 5 seconds)
    for i in {1..50}; do
        if ! kill -0 "$PID" 2>/dev/null; then
            break
        fi
        sleep 0.1
    done

    # Force kill if still running
    if kill -0 "$PID" 2>/dev/null; then
        kill -9 "$PID"
    fi
fi

rm -f "$PID_FILE"

# Show log output
if [ -f "$LOG_FILE" ]; then
    cat "$LOG_FILE"
fi

echo "Recording stopped"
