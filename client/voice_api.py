#!/usr/bin/env python3
"""
LAN voice API — exposes the local MLX STT/TTS as two simple HTTP endpoints so a
remote agent (e.g. Hermes on Origin) can use them over the network.

Endpoints (bound to 0.0.0.0 — LAN only, no auth):

  POST /transcribe   multipart/form-data, field "audio" (ogg/mp3/wav/…)
                     -> JSON {"text": "..."}      (Qwen3-ASR STT)

  POST /synthesize   JSON {"text": "..."}
                     -> WAV bytes, 24kHz mono 16-bit   (Qwen3-TTS)

  GET  /health       -> "ok"

This is independent of tts_daemon.py (the Option+S local-speaker daemon). It
loads its own copy of both models and never touches the Mac's mic or speakers —
audio only flows in/out as bytes.

Run:  python voice_api.py            (port from config.yaml: local.voice_api_port, default 9900)
"""

import io
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import wave
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import numpy as np
import yaml

from tts_client import LocalTTS
from voice_client import LocalTranscriber

DEFAULT_PORT = 9900
TTS_SAMPLE_RATE = 24000
STT_SAMPLE_RATE = 16000
CONFIG_PATH = Path(__file__).parent / 'config.yaml'


def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    return {}


def parse_multipart(content_type: str, body: bytes) -> dict:
    """Parse a multipart/form-data body at the byte level.

    Returns {field_name: (filename, raw_bytes)}. Byte-level (not the email/cgi
    modules) so binary audio payloads pass through untouched. cgi was removed in
    Python 3.13, and email's text-oriented parser can corrupt binary parts.
    """
    m = re.search(r'boundary=(?:"([^"]+)"|([^;]+))', content_type or '')
    if not m:
        return {}
    boundary = (m.group(1) or m.group(2)).strip().encode()
    delimiter = b'--' + boundary

    fields = {}
    for segment in body.split(delimiter):
        # Drop the preamble (b'') and the closing '--\r\n' segment.
        if segment.startswith(b'\r\n'):
            segment = segment[2:]
        if segment.endswith(b'\r\n'):
            segment = segment[:-2]
        if not segment or segment == b'--':
            continue
        header_end = segment.find(b'\r\n\r\n')
        if header_end == -1:
            continue
        raw_headers = segment[:header_end].decode('utf-8', 'replace')
        content = segment[header_end + 4:]

        name = filename = None
        for line in raw_headers.split('\r\n'):
            if line.lower().startswith('content-disposition'):
                nm = re.search(r'name="([^"]*)"', line)
                fn = re.search(r'filename="([^"]*)"', line)
                if nm:
                    name = nm.group(1)
                if fn:
                    filename = fn.group(1)
        if name is not None:
            fields[name] = (filename, content)
    return fields


def decode_to_wav16k(raw: bytes) -> str:
    """Decode arbitrary audio bytes to a 16kHz mono WAV file via ffmpeg.

    Returns the temp WAV path (caller must delete it). Raises on failure rather
    than guessing — a bad upload should surface, not be silently mistranscribed.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix='.upload') as fin:
        fin.write(raw)
        in_path = fin.name
    out_path = in_path + '.16k.wav'
    try:
        proc = subprocess.run(
            ['ffmpeg', '-y', '-i', in_path,
             '-ar', str(STT_SAMPLE_RATE), '-ac', '1', '-f', 'wav', out_path],
            capture_output=True,
        )
        if proc.returncode != 0 or not os.path.exists(out_path):
            tail = proc.stderr.decode('utf-8', 'replace')[-500:]
            raise RuntimeError(f"ffmpeg failed to decode audio: {tail}")
        return out_path
    finally:
        os.unlink(in_path)


def pcm_to_wav_bytes(audio_float32: np.ndarray, sample_rate: int) -> bytes:
    """Pack a mono float32 [-1,1] waveform into 16-bit PCM WAV bytes."""
    clipped = np.clip(audio_float32, -1.0, 1.0)
    int16 = (clipped * 32767.0).astype('<i2')
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(int16.tobytes())
    return buf.getvalue()


class VoiceAPIHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path == '/health':
            self._respond_text(200, 'ok')
        else:
            self._respond_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == '/transcribe':
            self._handle_transcribe()
        elif self.path == '/synthesize':
            self._handle_synthesize()
        else:
            self._respond_json(404, {"error": "not found"})

    def _read_body(self) -> bytes:
        length = int(self.headers.get('Content-Length', 0))
        return self.rfile.read(length) if length > 0 else b''

    def _handle_transcribe(self):
        body = self._read_body()
        fields = parse_multipart(self.headers.get('Content-Type', ''), body)
        # Prefer the documented "audio" field; fall back to the only file part.
        audio = fields.get('audio')
        if audio is None:
            audio = next((v for v in fields.values() if v[0] is not None), None)
        if audio is None or not audio[1]:
            self._respond_json(400, {"error": "no 'audio' file in multipart body"})
            return

        try:
            wav_path = decode_to_wav16k(audio[1])
        except Exception as e:
            self._respond_json(400, {"error": str(e)})
            return

        try:
            with self.server.infer_lock:
                text = self.server.stt.transcribe_file(wav_path)
            self._respond_json(200, {"text": text})
        except Exception as e:
            print(f"[transcribe error] {e}", file=sys.stderr, flush=True)
            self._respond_json(500, {"error": str(e)})
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass

    def _handle_synthesize(self):
        body = self._read_body()
        try:
            data = json.loads(body or b'{}')
            text = (data.get('text') or '').strip()
        except (json.JSONDecodeError, AttributeError):
            self._respond_json(400, {"error": "body must be JSON {\"text\": \"...\"}"})
            return
        if not text:
            self._respond_json(400, {"error": "missing 'text'"})
            return

        try:
            with self.server.infer_lock:
                audio = self.server.tts.synthesize_to_array(text)
        except Exception as e:
            print(f"[synthesize error] {e}", file=sys.stderr, flush=True)
            self._respond_json(500, {"error": str(e)})
            return

        if audio.size == 0:
            self._respond_json(400, {"error": "no speakable text after cleanup"})
            return

        wav = pcm_to_wav_bytes(audio, TTS_SAMPLE_RATE)
        self.send_response(200)
        self.send_header('Content-Type', 'audio/wav')
        self.send_header('Content-Length', str(len(wav)))
        self.end_headers()
        self.wfile.write(wav)

    def _respond_text(self, code: int, body: str):
        payload = body.encode()
        self.send_response(code)
        self.send_header('Content-Type', 'text/plain')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _respond_json(self, code: int, obj: dict):
        payload = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        # Quiet by default; errors are printed explicitly above.
        pass


class ThreadedHTTPServer(HTTPServer):
    """Threaded so /health stays responsive during a long synthesis."""

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


def main():
    config = load_config()
    local_cfg = config.get('local', {})
    port = local_cfg.get('voice_api_port', DEFAULT_PORT)
    stt_model = local_cfg.get('stt_model', 'Qwen/Qwen3-ASR-0.6B')

    print("Loading TTS model (Qwen3)...", flush=True)
    tts = LocalTTS(local_cfg)
    tts._ensure_model()
    print("Loading STT model (Qwen3-ASR)...", flush=True)
    stt = LocalTranscriber(stt_model)
    stt._ensure_model()
    print("Models loaded", flush=True)

    server = ThreadedHTTPServer(('0.0.0.0', port), VoiceAPIHandler)
    server.tts = tts
    server.stt = stt
    # MLX/Metal calls are serialized — one shared GPU, and concurrent generation
    # from multiple requests can clash. A single agent calls these serially anyway.
    server.infer_lock = threading.Lock()

    print(f"Voice API ready on http://0.0.0.0:{port}  "
          f"(POST /transcribe, POST /synthesize, GET /health)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...", flush=True)


if __name__ == '__main__':
    main()
