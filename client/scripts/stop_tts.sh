#!/bin/bash
# Hard-stop TTS playback — Hammerspoon double Option+S.
# Aborts the current utterance so the next Option+S starts fresh.

export PATH="/opt/homebrew/bin:$PATH"

VOICE_URL="http://127.0.0.1:9900"

if curl -s --max-time 1 "$VOICE_URL/health" > /dev/null 2>&1; then
    curl -s -X POST "$VOICE_URL/stop" >> /tmp/tts_debug.log 2>&1
fi
