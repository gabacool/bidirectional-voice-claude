#!/bin/bash
# Stop voice recording and trigger paste - called by Hammerspoon

PID_FILE="/tmp/parakeet_voice.pid"
LOG_FILE="/tmp/parakeet_voice.log"

# stdout is reserved for the transcription only (Hammerspoon pastes it).
# All diagnostics go to stderr.
TRANSCRIPTION_FILE="/tmp/parakeet_transcription.txt"

if [ ! -f "$PID_FILE" ]; then
    echo "Not recording" >&2
    exit 1
fi

PID=$(cat "$PID_FILE")

if kill -0 "$PID" 2>/dev/null; then
    # Send SIGUSR1 to stop recording gracefully
    kill -USR1 "$PID"

    # Wait for transcription to finish (up to 30 seconds). The model is
    # preloaded so this is normally 1-2s; the margin avoids cutting off
    # a longer utterance before its text is written.
    for i in {1..300}; do
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

# Diagnostics to stderr
if [ -f "$LOG_FILE" ]; then
    cat "$LOG_FILE" >&2
fi
echo "Recording stopped" >&2

# Emit ONLY the transcription on stdout for Hammerspoon to paste
if [ -f "$TRANSCRIPTION_FILE" ]; then
    cat "$TRANSCRIPTION_FILE"
fi
