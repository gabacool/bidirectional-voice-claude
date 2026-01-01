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

bullet = '\u23fa'  # ⏺ character
all_responses = []
current_block = []
in_response = False

for line in lines:
    stripped = line.strip()

    # Skip empty lines
    if not stripped:
        continue

    # Check for markers that END a response block
    if stripped.startswith('⎿') or stripped.startswith('│') or stripped.startswith('∴') or stripped.startswith('✢') or stripped.startswith('✻'):
        if current_block:
            all_responses.append(' '.join(current_block))
            current_block = []
        in_response = False
        continue

    # Check for user input (starts with >)
    if stripped.startswith('>'):
        if current_block:
            all_responses.append(' '.join(current_block))
            current_block = []
        in_response = False
        continue

    # Check if this is a Claude text response line (starts with ⏺)
    if stripped.startswith(bullet):
        rest = stripped[1:].strip()
        # Skip tool calls like ⏺ Bash(...)
        if rest and '(' in rest[:30] or any(rest.startswith(t) for t in ['Bash', 'Write', 'Read', 'Edit', 'Todo', 'Grep', 'Glob']):
            if current_block:
                all_responses.append(' '.join(current_block))
                current_block = []
            in_response = False
        else:
            # This is actual response text
            if rest:
                current_block.append(rest)
            in_response = True
        continue

    # Continuation lines (indented content, bullets, options)
    # These don't start with ⏺ but are part of the response
    if in_response and stripped:
        # Skip terminal UI elements
        if not any(stripped.startswith(x) for x in ['╭', '╰', '───', '$', 'tokens']) and '@' not in stripped[:20]:
            current_block.append(stripped)

# Don't forget the last block
if current_block:
    all_responses.append(' '.join(current_block))

# Get the last complete response block
if all_responses:
    print(all_responses[-1])
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
