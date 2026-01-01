#!/bin/bash
# Speak Claude's last response via TTS server
# Called by Hammerspoon on Option+S

export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$(dirname "$PROJECT_DIR")/venv"

# Capture iTerm2 terminal content
TERM_CONTENT=$(osascript -e '
tell application "iTerm2"
    tell current session of current window
        set termContent to contents
    end tell
end tell
return termContent
' 2>/dev/null)

if [ -z "$TERM_CONTENT" ]; then
    echo "$(date): Failed to capture terminal" >> /tmp/tts_debug.log
    exit 1
fi

# Extract last response using Python with proper encoding
RESPONSE=$(echo "$TERM_CONTENT" | python3 -c "
# -*- coding: utf-8 -*-
import sys

content = sys.stdin.read()
lines = content.strip().split('\n')

response_lines = []
bullet = '\u23fa'  # ⏺ character

for line in lines:
    stripped = line.strip()
    if stripped.startswith(bullet):
        rest = stripped[1:].strip()
        # Skip tool calls
        if rest and '(' not in rest[:30] and not any(rest.startswith(t) for t in ['Bash', 'Write', 'Read', 'Edit', 'Todo', 'Grep', 'Glob']):
            response_lines.append(rest)

if response_lines:
    # Get only the last response block
    print(response_lines[-1])
")

echo "$(date): Captured ${#RESPONSE} chars" >> /tmp/tts_debug.log

if [ -z "$RESPONSE" ] || [ ${#RESPONSE} -lt 5 ]; then
    echo "$(date): No response - trying fallback" >> /tmp/tts_debug.log
    # Fallback: just use clipboard
    RESPONSE=$(pbpaste)
fi

if [ -z "$RESPONSE" ] || [ ${#RESPONSE} -lt 5 ]; then
    echo "$(date): Still no response" >> /tmp/tts_debug.log
    exit 1
fi

echo "$RESPONSE" >> /tmp/tts_debug.log

# Send to TTS
echo "$RESPONSE" | pbcopy

source "$VENV_DIR/bin/activate"
python "$PROJECT_DIR/tts_client.py" 2>&1 | tee -a /tmp/tts_client.log
