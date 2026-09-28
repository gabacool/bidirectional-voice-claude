#!/bin/bash
# Rewind the current TTS playback by ~tts_seek_seconds — Hammerspoon Option+<
# No-op if the service isn't running or nothing is playing.

export PATH="/opt/homebrew/bin:$PATH"

VOICE_URL="http://127.0.0.1:9900"

if curl -s --max-time 1 "$VOICE_URL/health" > /dev/null 2>&1; then
    curl -s -X POST "$VOICE_URL/seek/back" >> /tmp/tts_debug.log 2>&1
fi
