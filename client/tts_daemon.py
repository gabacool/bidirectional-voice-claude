#!/usr/bin/env python3
"""
Persistent local TTS daemon — loads model once, serves requests via HTTP.

Start:    python tts_daemon.py
Health:   curl http://127.0.0.1:8089/health
Speak:    curl -X POST http://127.0.0.1:8089/speak
          (reads clipboard at processing time, or send text in body)
Stop:     kill $(cat /tmp/tts_daemon.pid)

Auto-started by speak_clipboard.sh on first Option+S press.
"""

import os
import signal
import subprocess
import sys
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import yaml

from tts_client import LocalTTS, SeekControl

PID_FILE = Path('/tmp/tts_daemon.pid')
DEFAULT_PORT = 8089
SAMPLE_RATE = 24000  # playback rate; seek deltas are expressed in these samples


class TTSHandler(BaseHTTPRequestHandler):

    def do_POST(self):
        if self.path == '/speak':
            self._handle_speak_toggle()
        elif self.path == '/stop':
            # Hard stop: abort playback and clear pause so the next /speak starts
            # fresh on new clipboard text.
            self.server.stop_event.set()
            self.server.pause_event.clear()
            self._respond(200, 'stopped')
        elif self.path == '/seek/back':
            self._handle_seek(-1)
        elif self.path == '/seek/forward':
            self._handle_seek(+1)
        else:
            self._respond(404, 'not found')

    def _handle_seek(self, direction):
        """Rewind (direction<0) or fast-forward (direction>0) the current
        utterance by tts_seek_seconds. No-op when nothing is playing."""
        if not self.server.playing.is_set():
            self._respond(200, 'idle')
            return
        samples = int(direction * self.server.tts.seek_seconds * SAMPLE_RATE)
        self.server.seek.request(samples)
        # A seek while paused implies the user wants to keep listening from the
        # new spot, so lift the pause too.
        self.server.pause_event.clear()
        print(f"[seek {'+' if direction > 0 else '-'}{self.server.tts.seek_seconds}s]",
              flush=True)
        self._respond(200, 'seeked')

    def _handle_speak_toggle(self):
        """Option+S is a toggle:
          - speaking & not paused -> pause
          - paused                -> resume
          - idle                  -> speak the clipboard (or POST body)
        """
        if self.server.playing.is_set():
            # Something is already playing: toggle pause/resume instead of
            # starting new speech.
            if self.server.pause_event.is_set():
                self.server.pause_event.clear()
                print("[resume]", flush=True)
                self._respond(200, 'resumed')
            else:
                self.server.pause_event.set()
                print("[pause]", flush=True)
                self._respond(200, 'paused')
            return

        # Idle -> start new speech.
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length > 0:
            text = self.rfile.read(content_length).decode('utf-8')
        else:
            text = subprocess.run(
                ['pbpaste'], capture_output=True, text=True
            ).stdout

        text = text.strip()
        if not text:
            self._respond(400, 'empty')
            return

        with self.server.speak_lock:
            # Pick up any config.yaml edits (voice, speed, etc.) without
            # needing a daemon restart.
            maybe_reload_config(self.server)
            self.server.stop_event.clear()
            self.server.pause_event.clear()
            self.server.seek.pop()  # drop any stale seek from a prior utterance
            self.server.playing.set()
            try:
                self.server.tts.synthesize_and_play(
                    text,
                    stop_event=self.server.stop_event,
                    pause_event=self.server.pause_event,
                    seek=self.server.seek,
                )
                self._respond(200, 'ok')
            except Exception as e:
                print(f"TTS error: {e}", file=sys.stderr)
                self._respond(500, str(e))
            finally:
                self.server.playing.clear()
                self.server.pause_event.clear()

    def do_GET(self):
        if self.path == '/health':
            self._respond(200, 'ok')
        else:
            self._respond(404, 'not found')

    def _respond(self, code, body):
        self.send_response(code)
        self.send_header('Content-Type', 'text/plain')
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, format, *args):
        pass


class ThreadedHTTPServer(HTTPServer):
    """Handle requests in separate threads so /health responds during playback."""

    daemon_threads = True

    def process_request(self, request, client_address):
        t = threading.Thread(target=self._handle, args=(request, client_address))
        t.daemon = True
        t.start()

    def _handle(self, request, client_address):
        try:
            self.finish_request(request, client_address)
        except Exception:
            self.handle_error(request, client_address)
        finally:
            self.shutdown_request(request)


CONFIG_PATH = Path(__file__).parent / 'config.yaml'


def load_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    return {}


def maybe_reload_config(server):
    """Re-apply config.yaml to the live TTS object if the file changed on disk.

    Lets voice/speed/etc. edits take effect without restarting the daemon.
    Only reloads the model if tts_model itself changed (handled in apply_config).
    """
    try:
        mtime = CONFIG_PATH.stat().st_mtime
    except OSError:
        return
    if mtime == server.config_mtime:
        return
    server.config_mtime = mtime
    local_cfg = load_config().get('local', {})
    server.tts.apply_config(local_cfg)
    print(f"Reloaded config (speaker={server.tts.speaker}, speed={server.tts.speed})")


def main():
    config = load_config()
    local_cfg = config.get('local', {})
    port = local_cfg.get('tts_server_port', DEFAULT_PORT)

    tts = LocalTTS(local_cfg)
    print("Loading TTS model...")
    tts._ensure_model()
    print("Model loaded")

    server = ThreadedHTTPServer(('127.0.0.1', port), TTSHandler)
    server.tts = tts
    server.speak_lock = threading.Lock()
    server.stop_event = threading.Event()
    server.pause_event = threading.Event()
    server.seek = SeekControl()  # pending rewind/forward sample deltas
    server.playing = threading.Event()  # set while an utterance is in progress
    try:
        server.config_mtime = CONFIG_PATH.stat().st_mtime
    except OSError:
        server.config_mtime = 0

    PID_FILE.write_text(str(os.getpid()))

    def shutdown(signum, frame):
        print("\nShutting down...")
        PID_FILE.unlink(missing_ok=True)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print(f"TTS daemon ready on http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == '__main__':
    main()
