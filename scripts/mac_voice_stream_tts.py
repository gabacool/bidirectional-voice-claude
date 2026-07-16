#!/Users/gabagool/Git/nvidia_parakeet/venv/bin/python
"""
Hermes TTS wrapper — TRUE STREAMING via the voice service's /v1/audio/speech.

One streaming request for the WHOLE text; PCM chunks play as the server
generates them (first audio ~0.2-0.5s regardless of length). Falls back to the
legacy per-sentence /synthesize pipeline if the streaming endpoint can't be
reached. Hermes command-provider contract unchanged: voice paths self-play +
tiny silent WAV; other --output paths get a real WAV, no playback; SIGTERM
stops immediately.

Source of record: ~/Git/nvidia_parakeet/scripts/ (installed copy in ~/bin).
Usage: mac_voice_stream_tts.py --text-file IN.txt --output OUT.wav
"""
import argparse
import io
import json
import os
import queue
import re
import signal
import sys
import threading
import wave
import urllib.request

import numpy as np

BASE = os.environ.get("MAC_VOICE_TTS_BASE", "http://localhost:9900")
LEGACY_ENDPOINT = os.environ.get("MAC_VOICE_TTS_ENDPOINT", f"{BASE}/synthesize")
STREAM_ENDPOINT = f"{BASE}/v1/audio/speech"
SR = 24000
_stop = threading.Event()


def _is_voice_path(path: str) -> bool:
    """Hermes voice/cache dirs mean 'self-play'; anything else is file generation."""
    return any(d in path for d in ("hermes_voice", "/cache/audio/", "/audio_cache/"))


def _split_sentences(text, min_len=20):
    """Split into sentences on . ! ? and the CJK enders 。！？, merging
    fragments shorter than min_len (keeps abbreviations from splitting and
    avoids tiny synth calls)."""
    parts = re.split(r'(?<=[.!?。！？])\s*', text)
    out, buf = [], ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        buf = (buf + " " + p).strip() if buf else p
        if len(buf) >= min_len:
            out.append(buf)
            buf = ""
    if buf:
        out.append(buf)
    return out


def _synth(text):
    """POST one sentence to /synthesize; return float32 mono @24kHz (or None)."""
    body = json.dumps({"text": text}).encode("utf-8")
    req = urllib.request.Request(LEGACY_ENDPOINT, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    with wave.open(io.BytesIO(data), "rb") as w:
        raw = w.readframes(w.getnframes())
    a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return a if a.size else None


def _write_wav(path, audio_f32):
    pcm = np.clip(audio_f32 * 32768.0, -32768, 32767).astype("<i2").tobytes()
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm)


def _stream_and_play(sentences):
    """Producer synthesizes sentences ahead; consumer plays them in order.
    Generation runs several times real-time, so after the first sentence
    playback is continuous."""
    import sounddevice as sd
    q = queue.Queue(maxsize=4)

    def producer():
        for s in sentences:
            if _stop.is_set():
                break
            try:
                a = _synth(s)
            except Exception as e:
                sys.stderr.write(f"synth error: {e}\n")
                a = None
            if a is not None:
                while not _stop.is_set():
                    try:
                        q.put(a, timeout=0.2)
                        break
                    except queue.Full:
                        continue
        q.put(None)

    threading.Thread(target=producer, daemon=True).start()
    with sd.OutputStream(samplerate=SR, channels=1, dtype="float32") as out:
        while not _stop.is_set():
            try:
                a = q.get(timeout=0.2)
            except queue.Empty:
                continue
            if a is None:
                break
            i = 0
            while i < len(a) and not _stop.is_set():
                out.write(a[i:i + 2400])   # 0.1s blocks so a stop is honored fast
                i += 2400


def _stream_play(text):
    """ONE streaming /v1/audio/speech request; play chunks as they arrive.
    Returns False if the CONNECTION failed (caller falls back to legacy);
    a mid-stream error just stops playback (retrying would re-speak audio)."""
    import sounddevice as sd
    body = json.dumps({"input": text, "response_format": "pcm"}).encode()
    req = urllib.request.Request(STREAM_ENDPOINT, body,
                                 {"Content-Type": "application/json"})
    try:
        resp = urllib.request.urlopen(req, timeout=300)
    except Exception as e:
        sys.stderr.write(f"streaming endpoint unavailable ({e}); using legacy\n")
        return False
    with resp, sd.OutputStream(samplerate=SR, channels=1, dtype="int16") as out:
        try:
            while not _stop.is_set():
                b = resp.read(4800)   # 0.1s blocks so SIGTERM stops fast
                if not b:
                    break
                out.write(np.frombuffer(b, dtype="<i2"))
        except Exception as e:
            sys.stderr.write(f"stream aborted mid-play: {e}\n")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?")
    ap.add_argument("--text-file")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    text = ((open(args.text_file, encoding="utf-8").read()
             if args.text_file else args.text) or "").strip()

    def _sig(*_):
        _stop.set()
    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)

    out = args.output or ""
    # Voice playback goes to a Hermes voice/cache dir (TUI: .../hermes_voice/,
    # desktop: .../.hermes/cache/audio/ or audio_cache/). Anything else is an
    # explicit file-generation path — write the real WAV there, don't self-play.
    is_voice = _is_voice_path(out)
    if is_voice:
        # Voice-speak: self-play, and leave a tiny silent WAV so Hermes's
        # "command produced output" check passes and its post-play is a no-op.
        _write_wav(args.output, np.zeros(int(0.05 * SR), dtype=np.float32))
        if text:
            if not _stream_play(text):
                _stream_and_play(_split_sentences(text))   # legacy fallback
    else:
        # File generation: synthesize the whole text in one call, no playback.
        audio = _synth(text) if text else None
        _write_wav(args.output, audio if audio is not None
                   else np.zeros(0, dtype=np.float32))


if __name__ == "__main__":
    main()
