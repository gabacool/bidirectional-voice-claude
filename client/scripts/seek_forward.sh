#!/bin/bash
# Fast-forward the current TTS playback by ~tts_seek_seconds — Hammerspoon Option+>
# Capped at how much audio has been generated so far. No-op if nothing is playing.

export PATH="/opt/homebrew/bin:$PATH"

DAEMON_URL="http://127.0.0.1:8089"

if curl -s --max-time 1 "$DAEMON_URL/health" > /dev/null 2>&1; then
    curl -s -X POST "$DAEMON_URL/seek/forward" >> /tmp/tts_debug.log 2>&1
fi
