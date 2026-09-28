#!/bin/bash
# Fast-forward the current TTS playback by ~tts_seek_seconds — Hammerspoon Option+>
# Capped at how much audio has been generated so far. No-op if nothing is playing.

export PATH="/opt/homebrew/bin:$PATH"

VOICE_URL="http://127.0.0.1:9900"

if curl -s --max-time 1 "$VOICE_URL/health" > /dev/null 2>&1; then
    curl -s -X POST "$VOICE_URL/seek/forward" >> /tmp/tts_debug.log 2>&1
fi
