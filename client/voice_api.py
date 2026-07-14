#!/usr/bin/env python3
"""
LAN voice API — exposes the local MLX STT/TTS as two simple HTTP endpoints so a
remote agent (e.g. Hermes on Origin) can use them over the network.

Endpoints (bound to 0.0.0.0 — LAN only, no auth):

  POST /transcribe   multipart/form-data, field "audio" (ogg/mp3/wav/…)
                     -> JSON {"text": "..."}      (Qwen3-ASR STT)

  POST /v1/audio/transcriptions
                     multipart/form-data, field "file" (OpenAI STT shape;
                     model/language/response_format accepted and ignored)
                     -> JSON {"text": "..."}      (Qwen3-ASR STT)

  POST /synthesize   JSON {"text": "..."}
                     -> WAV bytes, 24kHz mono 16-bit   (Qwen3-TTS, one blob)

  POST /v1/audio/speech
                     JSON {"input", "voice"?, "response_format": "wav"|"pcm",
                     "model"? (ignored)} (OpenAI TTS shape)
                     -> audio streamed progressively as the model generates
                        (unknown-length body, Connection: close), 24kHz mono
                        16-bit; "wav" prepends a streaming WAV header, "pcm" is
                        raw 16-bit LE PCM

  GET  /health       -> "ok"

This is independent of tts_daemon.py (the Option+S local-speaker daemon). It
loads its own copy of both models and never touches the Mac's mic or speakers —
audio only flows in/out as bytes.

Run:  python voice_api.py            (port from config.yaml: local.voice_api_port, default 9900)
"""

import io
import json
import os
import queue
import re
import struct
import subprocess
import sys
import tempfile
import threading
import wave
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
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

# Unknown-length sentinel for the streaming WAV header's RIFF + data size
# fields: a progressive player starts immediately and reads PCM until the
# socket closes; a saved header+chunks blob still parses (players/`wave` read
# the real audio to EOF). 0xFFFFFFFF is the widely-accepted streaming sentinel.
STREAMING_SIZE_SENTINEL = 0xFFFFFFFF
# Seconds of audio per streamed TTS chunk. Smaller than the batch default so the
# first audio byte reaches the client sooner (time-to-first-byte), at the cost
# of slightly more chunks — the right trade for a latency-sensitive endpoint.
SPEECH_STREAMING_INTERVAL = 0.5
# Bounded hand-off queue between the single inference thread (producer) and the
# HTTP handler thread (consumer) for streaming speech. Small on purpose: the
# first chunk must flow straight through (low time-to-first-byte) and we must
# NOT buffer the whole utterance ahead of the client.
STREAM_QUEUE_MAXSIZE = 4


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


def extract_upload(fields: dict, field_name: str) -> tuple[bytes | None, str | None]:
    """Pick the uploaded audio bytes out of parsed multipart fields.

    Prefers the part named ``field_name`` (e.g. "audio" or "file"); if absent,
    falls back to the only file part present. Returns ``(bytes, None)`` on
    success or ``(None, kind)`` where ``kind`` is ``"missing"`` (no file part at
    all) or ``"empty"`` (file part present but zero bytes).
    """
    part = fields.get(field_name)
    if part is None:
        # A file part is any part carrying a filename (v[0] is not None).
        part = next((v for v in fields.values() if v[0] is not None), None)
    if part is None:
        return None, "missing"
    if not part[1]:
        return None, "empty"
    return part[1], None


def error_body(message: str, openai_shape: bool) -> dict:
    """Wrap an error message in the route-appropriate JSON body.

    ``openai_shape=True`` -> ``{"error": {"message": ...}}`` (the OpenAI
    ``/v1/audio/*`` shape); ``openai_shape=False`` -> ``{"error": ...}`` (the
    legacy ``/transcribe`` flat shape). Shared by every endpoint's error paths.
    """
    if openai_shape:
        return {"error": {"message": message}}
    return {"error": message}


def f32_to_pcm16(arr: np.ndarray) -> bytes:
    """Convert a mono float32 [-1, 1] waveform to 16-bit little-endian PCM bytes.

    Samples are clipped to [-1, 1] before scaling so out-of-range values saturate
    at full scale instead of wrapping. This is the single float32->PCM16
    conversion used by both the batch WAV path and the streaming speech path.
    """
    clipped = np.clip(arr, -1.0, 1.0)
    int16 = (clipped * 32767.0).astype('<i2')
    return int16.tobytes()


def wav_streaming_header(sample_rate: int = TTS_SAMPLE_RATE, channels: int = 1,
                         bits_per_sample: int = 16) -> bytes:
    """Build a 44-byte WAV header with unknown-length (streaming) size fields.

    The RIFF chunk size and data chunk size are both set to
    ``STREAMING_SIZE_SENTINEL`` so the header can be written before the total
    audio length is known: progressive players start playing at once, and a
    saved header+PCM blob still parses (readers take the real length from EOF).
    """
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    return b''.join([
        b'RIFF',
        struct.pack('<I', STREAMING_SIZE_SENTINEL),
        b'WAVE',
        b'fmt ',
        struct.pack('<I', 16),              # fmt subchunk size (PCM)
        struct.pack('<H', 1),               # audio format 1 = PCM
        struct.pack('<H', channels),
        struct.pack('<I', sample_rate),
        struct.pack('<I', byte_rate),
        struct.pack('<H', block_align),
        struct.pack('<H', bits_per_sample),
        b'data',
        struct.pack('<I', STREAMING_SIZE_SENTINEL),
    ])


def parse_speech_request(body: bytes) -> tuple[dict | None, str | None]:
    """Validate a ``/v1/audio/speech`` JSON body without doing any synthesis.

    Returns ``(params, None)`` where ``params`` is
    ``{"input", "voice", "response_format"}`` on success, or ``(None, message)``
    with a client-facing error message. ``model`` is accepted and ignored;
    ``voice=None`` means the configured default speaker. Pure/testable so request
    validation runs without loading models or a socket.
    """
    try:
        data = json.loads(body or b'{}')
    except (json.JSONDecodeError, ValueError):
        return None, "body must be valid JSON"
    if not isinstance(data, dict):
        return None, "body must be a JSON object"
    raw_input = data.get('input')
    text = raw_input.strip() if isinstance(raw_input, str) else ''
    if not text:
        return None, "missing required 'input' field"
    response_format = data.get('response_format', 'wav')
    if response_format not in ('wav', 'pcm'):
        return None, "response_format must be 'wav' or 'pcm'"
    voice = data.get('voice')
    if voice is not None and not isinstance(voice, str):
        return None, "voice must be a string"
    return {"input": text, "voice": voice, "response_format": response_format}, None


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
    """Pack a mono float32 [-1,1] waveform into a complete 16-bit PCM WAV blob."""
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(f32_to_pcm16(audio_float32))
    return buf.getvalue()


class _GeneratorError:
    """Wrapper that carries a producer-side exception across the hand-off queue."""

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc


# Distinct end-of-stream marker put on the queue when the producer finishes.
_STREAM_SENTINEL = object()


def run_generator_on(
    executor: ThreadPoolExecutor,
    gen_factory: Callable[[], Iterator[np.ndarray]],
    maxsize: int = STREAM_QUEUE_MAXSIZE,
) -> Iterator[np.ndarray]:
    """Pump ``gen_factory()`` on ``executor``'s thread, yield chunks on the caller's.

    mlx (>=0.31) GPU streams are thread-local, so every model call must run on the
    one dedicated inference thread owned by ``executor``. A streaming generator,
    however, must feed the HTTP handler thread that owns the client socket. This
    bridge runs the generator on the inference thread (the producer) and hands its
    chunks to the consuming caller through a small bounded ``queue.Queue``.

    Guarantees:
      * chunk order is preserved and no chunk is dropped, even for a slow consumer;
      * a producer-side exception is re-raised in the consumer (not swallowed);
      * the queue is bounded (``maxsize``) — the first chunk flows straight through
        and generation never races far ahead of the client;
      * if the consumer stops early (``close()`` / ``GeneratorExit`` on socket
        disconnect) a stop flag is set that the producer checks between chunks, so
        generation aborts instead of running to completion as an orphan.

    Args:
        executor: The single-worker inference executor to run the generator on.
        gen_factory: Zero-arg callable that creates the underlying chunk iterator
            (deferred so its lazy model work also runs on the inference thread).
        maxsize: Bounded hand-off queue depth.

    Yields:
        The chunks produced by ``gen_factory()``, in order.
    """
    handoff: queue.Queue = queue.Queue(maxsize=maxsize)
    stop = threading.Event()

    def _put(item: object) -> bool:
        # Block for space, but wake every 0.1s to honour an early-stop request so
        # a full queue can never wedge the producer after the consumer is gone.
        while not stop.is_set():
            try:
                handoff.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _produce() -> None:
        gen: Iterator[np.ndarray] | None = None
        try:
            gen = gen_factory()
            for chunk in gen:
                if stop.is_set():
                    break
                if not _put(chunk):
                    break
        except BaseException as exc:  # noqa: BLE001 - relayed to the consumer
            _put(_GeneratorError(exc))
        finally:
            # Abort the underlying generator (runs its own cleanup) if we bailed
            # out early, then always signal end-of-stream (unless already stopped).
            if gen is not None:
                close = getattr(gen, "close", None)
                if callable(close):
                    close()
            _put(_STREAM_SENTINEL)

    future = executor.submit(_produce)
    try:
        while True:
            item = handoff.get()
            if item is _STREAM_SENTINEL:
                break
            if isinstance(item, _GeneratorError):
                raise item.exc
            yield item
    finally:
        # Consumer is done or was closed (disconnect): stop the producer and drain
        # so a blocked _put wakes at once, then join so no orphan thread lingers.
        stop.set()
        try:
            while True:
                handoff.get_nowait()
        except queue.Empty:
            pass
        future.result()


class VoiceAPIHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path == '/health':
            self._respond_text(200, 'ok')
        else:
            self._respond_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == '/transcribe':
            self._handle_transcribe()
        elif self.path == '/v1/audio/transcriptions':
            self._handle_openai_transcribe()
        elif self.path == '/synthesize':
            self._handle_synthesize()
        elif self.path == '/v1/audio/speech':
            self._handle_speech()
        else:
            self._respond_json(404, {"error": "not found"})

    def _read_body(self) -> bytes:
        length = int(self.headers.get('Content-Length', 0))
        return self.rfile.read(length) if length > 0 else b''

    def _handle_transcribe(self) -> None:
        # Legacy endpoint: field "audio", flat {"error": "..."} shape. Behavior
        # is unchanged — the shared implementation reproduces it exactly.
        self._transcribe_request(field_name='audio', openai_shape=False)

    def _handle_openai_transcribe(self) -> None:
        # OpenAI-compatible endpoint: field "file", {"error": {"message": ...}}.
        self._transcribe_request(field_name='file', openai_shape=True)

    def _transcribe_request(self, field_name: str, openai_shape: bool) -> None:
        """Shared decode+transcribe flow for both STT routes.

        Parameterized only by the multipart file field name and the error body
        shape (see ``error_body``). Success is identical for both
        routes: HTTP 200, ``{"text": "..."}``.
        """
        body = self._read_body()
        fields = parse_multipart(self.headers.get('Content-Type', ''), body)
        audio_bytes, err = extract_upload(fields, field_name)
        if err is not None:
            if openai_shape:
                message = (f"missing required '{field_name}' field"
                           if err == 'missing' else 'uploaded file is empty')
            else:
                # Legacy collapses missing+empty into one message (unchanged).
                message = f"no '{field_name}' file in multipart body"
            self._respond_json(400, error_body(message, openai_shape))
            return

        # Optional OpenAI form fields are accepted and ignored (logged once).
        if openai_shape:
            ignored = {k: v[1].decode('utf-8', 'replace')
                       for k in ('model', 'language', 'response_format')
                       if (v := fields.get(k)) is not None}
            if ignored:
                print(f"[v1/audio/transcriptions] ignoring params: {ignored}",
                      file=sys.stderr, flush=True)

        try:
            wav_path = decode_to_wav16k(audio_bytes)
        except Exception as e:
            self._respond_json(400, error_body(str(e), openai_shape))
            return

        try:
            # All model inference runs on the one dedicated inference thread
            # (mlx GPU streams are thread-local); the single worker serializes it.
            text = self.server.executor.submit(
                self.server.stt.transcribe_file, wav_path).result()
            self._respond_json(200, {"text": text})
        except Exception as e:
            print(f"[transcribe error] {e}", file=sys.stderr, flush=True)
            self._respond_json(500, error_body(str(e), openai_shape))
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
            audio = self.server.executor.submit(
                self.server.tts.synthesize_to_array, text).result()
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

    def _handle_speech(self) -> None:
        """OpenAI-compatible ``/v1/audio/speech`` — stream TTS audio progressively.

        The response body is delivered with no Content-Length and
        ``Connection: close`` (HTTP/1.0 unknown-length semantics): the first
        bytes hit the wire as soon as the first model chunk exists, and the
        client reads until the socket closes. All validation happens *before* any
        audio byte is sent so those failures can still be a proper 400; once
        bytes are on the wire a mid-stream failure can only close the connection.
        """
        body = self._read_body()
        params, err = parse_speech_request(body)
        if err is not None:
            self._respond_json(400, error_body(err, openai_shape=True))
            return

        text = params['input']
        voice = params['voice']
        fmt = params['response_format']

        # The model runs on the single inference thread; this handler thread owns
        # the socket. run_generator_on bridges the two: chunks are pumped on the
        # inference thread and handed here through a small bounded queue, and
        # closing the returned generator aborts the producer (no orphan) — so we
        # close it in every exit path below.
        stream = run_generator_on(
            self.server.executor,
            lambda: self.server.tts.synthesize_stream(
                text, voice=voice, streaming_interval=SPEECH_STREAMING_INTERVAL),
        )
        try:
            # Pull the first chunk while no bytes are on the wire yet, so an
            # up-front generation failure (e.g. unknown voice) or empty-after-
            # cleanup text still becomes a clean 400 instead of a broken stream.
            first = None
            try:
                for chunk in stream:
                    first = chunk
                    break
            except Exception as e:
                print(f"[speech] generation failed before audio: {e}",
                      file=sys.stderr, flush=True)
                self._respond_json(
                    400, error_body(f"speech generation failed: {e}",
                                    openai_shape=True))
                return
            if first is None:
                self._respond_json(
                    400, error_body("no speakable text after cleanup",
                                    openai_shape=True))
                return

            # Commit to a streaming 200: unknown length, close on completion.
            self.send_response(200)
            self.send_header('Content-Type',
                             'audio/wav' if fmt == 'wav' else 'audio/pcm')
            self.send_header('Connection', 'close')
            self.end_headers()
            self.close_connection = True

            try:
                if fmt == 'wav':
                    self.wfile.write(wav_streaming_header())
                self.wfile.write(f32_to_pcm16(first))
                self.wfile.flush()
                for chunk in stream:
                    self.wfile.write(f32_to_pcm16(chunk))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError) as e:
                # Client hung up mid-stream: not an error, just stop writing.
                print(f"[speech] client disconnected mid-stream: {e}",
                      file=sys.stderr, flush=True)
            except Exception as e:
                # Generation failed after audio was already sent — a 400 is no
                # longer possible; log and let the connection close.
                print(f"[speech] generation error mid-stream: {e}",
                      file=sys.stderr, flush=True)
        finally:
            # Stop the producer (aborts generation if we exited early) and join it.
            stream.close()

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

    # One dedicated inference thread. mlx (>=0.31) GPU streams are thread-local,
    # so EVERY model call — load and generate — must run on this single thread;
    # its single worker also serializes concurrent requests onto the shared GPU.
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="infer")

    tts = LocalTTS(local_cfg)
    stt = LocalTranscriber(stt_model)
    # Warm both models ON the inference thread so their thread-local mlx streams
    # are created where inference will later run (not on the main thread).
    print("Loading TTS model (Qwen3)...", flush=True)
    executor.submit(tts._ensure_model).result()
    print("Loading STT model (Qwen3-ASR)...", flush=True)
    executor.submit(stt._ensure_model).result()
    print("Models loaded", flush=True)

    server = ThreadedHTTPServer(('0.0.0.0', port), VoiceAPIHandler)
    server.tts = tts
    server.stt = stt
    server.executor = executor

    print(f"Voice API ready on http://0.0.0.0:{port}  "
          f"(POST /transcribe, POST /synthesize, GET /health)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...", flush=True)


if __name__ == '__main__':
    main()
