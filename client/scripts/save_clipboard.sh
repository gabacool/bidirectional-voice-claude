#!/bin/bash
# Save clipboard TTS as a WAV in ~/Downloads — Hammerspoon Option+Shift+S.
#
# Same engine/voice as Option+S. Posts to localhost :9900 /save.
# Text is $1 (Hammerspoon selection). Empty → service reads the clipboard.

export PATH="/opt/homebrew/bin:$PATH"
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

VOICE_URL="http://127.0.0.1:9900"
TEXT="$1"

post_save() {
    if [ -n "$TEXT" ]; then
        curl -s -X POST "$VOICE_URL/save" --data-raw "$TEXT"
    else
        curl -s -X POST "$VOICE_URL/save"
    fi
}

ensure_voice_api() {
    if curl -s --max-time 1 "$VOICE_URL/health" > /dev/null 2>&1; then
        return 0
    fi
    echo "$(date): voice_api down, kickstarting LaunchAgent..." >> /tmp/tts_debug.log
    launchctl kickstart -k "gui/$(id -u)/com.gabagool.voiceapi" 2>/dev/null || true
    for i in $(seq 1 60); do
        if curl -s --max-time 1 "$VOICE_URL/health" > /dev/null 2>&1; then
            echo "$(date): voice_api ready" >> /tmp/tts_debug.log
            return 0
        fi
        sleep 0.5
    done
    echo "$(date): voice_api failed to start within 30s" >> /tmp/tts_debug.log
    return 1
}

if ensure_voice_api; then
    path=$(post_save)
    echo "$(date): save -> $path" >> /tmp/tts_debug.log
    printf '%s' "$path"
    exit 0
fi
exit 1
