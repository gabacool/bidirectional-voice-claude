#!/bin/bash
# Start the Parakeet ASR WebSocket server

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="$(dirname "$SCRIPT_DIR")"

cd "$SERVER_DIR"
source venv/bin/activate
python server.py --host 0.0.0.0 --port 8087
