#!/bin/bash
# Hard-stop TTS playback — called by Hammerspoon on Option+X.
# Aborts the current utterance so the next Option+S starts fresh.

export PATH="/opt/homebrew/bin:$PATH"

DAEMON_URL="http://127.0.0.1:8089"

# Only meaningful if the daemon is running; no-op otherwise.
if curl -s --max-time 1 "$DAEMON_URL/health" > /dev/null 2>&1; then
    curl -s -X POST "$DAEMON_URL/stop" >> /tmp/tts_debug.log 2>&1
fi
