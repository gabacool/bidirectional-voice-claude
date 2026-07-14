#!/usr/bin/env python3
"""
TTS client for Claude Code voice output.
Reads text from clipboard, synthesizes speech, and plays audio.
Supports two backends: "origin" (GPU server via WebSocket) or "local" (Qwen3-TTS on Mac via MLX).
"""

import asyncio
import json
import queue
import subprocess
import sys
import io
import re
import threading
import time
import wave
from pathlib import Path
from typing import Iterator

import numpy as np
import sounddevice as sd
import yaml


def _manual_cleanup(text: str) -> str:
    """Clean up text for speech when no LLM summarizer is available."""
    # Remove code blocks
    text = re.sub(r'```[\s\S]*?```', '[code block removed]', text)

    # Remove inline code
    text = re.sub(r'`[^`]+`', '', text)

    # Remove markdown formatting
    text = text.replace("**", "").replace("*", "").replace("#", "")

    # Remove ASCII art characters
    text = re.sub(r'[│├└┌┐┘─┬┴┼═║╔╗╚╝╠╣╦╩╬]+', '', text)

    # Remove table formatting
    text = re.sub(r'\|[^\n]+\|', '', text)

    # Collapse whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)

    return text.strip()


def _prepare_for_speech(text: str) -> str:
    """Prepare text for TTS - strip markdown/code artifacts, pass plain text through."""
    if "```" not in text and not any(c in text for c in ['|', '─', '│', '┌', '└']):
        return text.replace("**", "").replace("*", "").replace("`", "").strip()
    return _manual_cleanup(text)


def _squeeze_silence(audio: np.ndarray, sr: int = 24000, max_gap_s: float = 0.2,
                     thresh: float = 0.01, win_s: float = 0.02) -> np.ndarray:
    """Shorten the long silences the TTS model inserts between sentences so
    speech doesn't drag. Any run of near-silence longer than max_gap_s is
    clipped down to max_gap_s. Silence is detected on a short RMS window (a
    per-sample test would trip on every zero-crossing inside normal speech).
    max_gap_s <= 0 disables squeezing and returns the audio untouched.
    """
    n = audio.size
    if n == 0 or max_gap_s <= 0:
        return audio
    win = max(1, int(win_s * sr))
    nwin = n // win
    if nwin < 2:
        return audio
    body = audio[:nwin * win].reshape(nwin, win)
    silent = np.sqrt((body ** 2).mean(axis=1)) < thresh
    max_run = max(1, round(max_gap_s / win_s))
    keep = np.ones(nwin, dtype=bool)
    i = 0
    while i < nwin:
        if silent[i]:
            j = i
            while j < nwin and silent[j]:
                j += 1
            if j - i > max_run:
                keep[i + max_run:j] = False   # drop the excess pause windows
            i = j
        else:
            i += 1
    kept = body[keep].reshape(-1)
    return np.concatenate([kept, audio[nwin * win:]]).astype(np.float32)


class SeekControl:
    """Thread-safe accumulator for rewind/forward requests.

    The HTTP handler thread adds a signed sample delta (negative = rewind,
    positive = forward); the playback thread drains it once per loop with
    pop() and applies it to its position pointer.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._pending = 0

    def request(self, samples: int):
        with self._lock:
            self._pending += samples

    def pop(self) -> int:
        with self._lock:
            d = self._pending
            self._pending = 0
            return d


class AudioTape:
    """A growable PCM buffer the consumer can seek within.

    The producer appends every generated chunk here instead of discarding it
    after playback, so we retain the full utterance and can rewind to any
    earlier point. Forward seeks are bounded by `length` (what's been generated
    so far) — you can't jump past audio that doesn't exist yet. Capacity grows
    by doubling, so appends are amortized O(1) and reads copy only a sub-chunk.
    """

    def __init__(self, initial=48000):
        self._lock = threading.Lock()
        self._buf = np.zeros(initial, dtype=np.float32)
        self.length = 0      # samples generated so far
        self.done = False    # producer finished (no more audio coming)

    def append(self, arr: np.ndarray):
        with self._lock:
            need = self.length + len(arr)
            if need > len(self._buf):
                cap = len(self._buf)
                while cap < need:
                    cap *= 2
                grown = np.zeros(cap, dtype=np.float32)
                grown[:self.length] = self._buf[:self.length]
                self._buf = grown
            self._buf[self.length:need] = arr
            self.length = need

    def read(self, pos: int, n: int):
        """Return (samples, total_len, done) for up to n samples from pos."""
        with self._lock:
            if pos >= self.length:
                return np.empty(0, dtype=np.float32), self.length, self.done
            end = min(pos + n, self.length)
            return self._buf[pos:end].copy(), self.length, self.done

    def finish(self):
        with self._lock:
            self.done = True


class LocalTTS:
    """Synthesize speech locally using Qwen3-TTS via mlx-audio."""

    def __init__(self, config: dict):
        self._model = None
        self.model_name = None
        self.apply_config(config)

    def apply_config(self, config: dict):
        """(Re)load all generation settings from a config dict.

        Safe to call on a live daemon: only the cheap params are swapped in
        place. The model is reloaded ONLY if tts_model changed, so changing
        voice/speed/etc. takes effect instantly with the model staying resident.
        """
        new_model = config.get('tts_model', 'mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit')
        if new_model != self.model_name:
            # Model identity changed — drop the loaded model so it reloads lazily.
            self.model_name = new_model
            self._model = None
        self.speaker = config.get('tts_speaker', 'aiden')
        self.language = config.get('tts_language', 'english')
        self.instruct = config.get('tts_instruct')  # None means no instruction
        self.temperature = config.get('tts_temperature', 0.9)
        self.top_k = config.get('tts_top_k', 50)
        self.top_p = config.get('tts_top_p', 1.0)
        self.repetition_penalty = config.get('tts_repetition_penalty', 1.05)
        self.max_tokens = config.get('tts_max_tokens', 4096)
        self.streaming_interval = config.get('tts_streaming_interval', 2.0)
        self.speed = config.get('tts_speed', 1.0)  # >1 faster, <1 slower (pitch preserved)
        self.seek_seconds = config.get('tts_seek_seconds', 15)  # rewind/forward step
        # Cap the model's inter-sentence silences to this many seconds (0 = off).
        self.max_pause = config.get('tts_max_pause_seconds', 0.2)

    def _ensure_model(self):
        """Lazy-load the TTS model on first use."""
        if self._model is not None:
            return
        from mlx_audio.tts.utils import load_model
        print(f"Loading TTS model: {self.model_name}...")
        self._model = load_model(self.model_name)
        print("TTS model loaded")

    def synthesize_stream(self, text: str, voice: str | None = None,
                          streaming_interval: float | None = None
                          ) -> Iterator[np.ndarray]:
        """Yield mono float32 24kHz audio chunks AS THE MODEL GENERATES THEM.

        This is the single generation code path: the model is driven in
        streaming mode and each non-empty chunk is handed back immediately, with
        no buffering and no concatenation, so the first audio is available long
        before the full utterance finishes.

        NOTE: the whole-array batch post-processing — inter-sentence silence
        squeezing (``_squeeze_silence``) and the ``tts_speed`` time-stretch —
        needs the complete waveform and is therefore applied ONLY by the batch
        path (``synthesize_to_array``). This streaming path yields the RAW model
        chunks unmodified.

        Args:
            text: Text to speak. Cleaned via ``_prepare_for_speech`` first; an
                empty result yields no chunks.
            voice: Speaker name to pass to the model. ``None`` uses the
                configured default speaker (``tts_speaker``).
            streaming_interval: Seconds of audio per chunk. ``None`` uses the
                configured ``tts_streaming_interval``.

        Yields:
            1-D float32 numpy arrays of 24kHz mono audio, one per model chunk.
        """
        self._ensure_model()

        speech_text = _prepare_for_speech(text)
        if not speech_text:
            return

        kwargs = dict(
            text=speech_text,
            speaker=self.speaker if voice is None else voice,
            language=self.language,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            max_tokens=self.max_tokens,
            stream=True,
            streaming_interval=(self.streaming_interval
                                if streaming_interval is None
                                else streaming_interval),
        )
        if self.instruct:
            kwargs['instruct'] = self.instruct

        for chunk in self._model.generate_custom_voice(**kwargs):
            audio_np = np.array(chunk.audio, dtype=np.float32)
            if audio_np.size == 0:
                continue
            yield audio_np

    def synthesize_to_array(self, text: str) -> np.ndarray:
        """Synthesize text and return the full mono float32 waveform at 24kHz.

        No playback — used by the LAN voice API to hand raw audio back to a
        remote caller. Reuses the same voice/temperature/speed config as
        playback, just collected into one array instead of streamed to speakers.

        Consumes ``synthesize_stream`` (the one generation code path) and then
        applies the whole-array batch post-processing that only makes sense on
        the complete waveform: the per-chunk ``tts_speed`` time-stretch and the
        inter-sentence ``_squeeze_silence``. The streaming generator itself
        yields raw chunks without either.
        """
        chunks = []
        for audio_np in self.synthesize_stream(text):
            if self.speed != 1.0:
                import librosa
                audio_np = librosa.effects.time_stretch(
                    audio_np, rate=self.speed
                ).astype(np.float32)
            chunks.append(audio_np)

        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return _squeeze_silence(np.concatenate(chunks), max_gap_s=self.max_pause)

    def synthesize_and_play(self, text: str, stop_event=None, pause_event=None,
                            seek=None):
        """Synthesize text to speech and play it, streaming chunks as they generate.

        pause_event (optional threading.Event): when SET, playback pauses (audio
        device stopped) and holds; when CLEARED, playback resumes from where it
        left off. stop_event aborts entirely.

        seek (optional SeekControl): pending signed sample deltas to move the
        playback position — negative rewinds, positive fast-forwards. Rewinds
        reach back to the start (full audio is retained); forwards are capped at
        whatever has been generated so far.
        """
        self._ensure_model()

        speech_text = _prepare_for_speech(text)
        if not speech_text:
            print("No text to speak after cleanup")
            return

        print(f"Speaking: {speech_text[:100]}...")

        kwargs = dict(
            text=speech_text,
            speaker=self.speaker,
            language=self.language,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            max_tokens=self.max_tokens,
            stream=True,
            streaming_interval=self.streaming_interval,
        )
        if self.instruct:
            kwargs['instruct'] = self.instruct

        tape = AudioTape()
        gen_error = [None]

        def producer():
            try:
                for chunk in self._model.generate_custom_voice(**kwargs):
                    if stop_event is not None and stop_event.is_set():
                        break
                    # While paused, block here so the lazy generator is NOT
                    # advanced — no model/GPU work happens during a pause.
                    while (pause_event is not None and pause_event.is_set()
                           and not (stop_event is not None and stop_event.is_set())):
                        time.sleep(0.1)
                    if stop_event is not None and stop_event.is_set():
                        break
                    audio_np = np.array(chunk.audio, dtype=np.float32)
                    if audio_np.size == 0:
                        continue
                    # Pitch-preserving speed change. Done here on the producer
                    # thread so the playback thread keeps its buffer headroom.
                    if self.speed != 1.0:
                        import librosa
                        audio_np = librosa.effects.time_stretch(
                            audio_np, rate=self.speed
                        ).astype(np.float32)
                    # Trim the model's long inter-sentence pauses so playback
                    # doesn't drag. Generation already runs ahead of playback,
                    # so this directly shortens the gap the listener hears.
                    audio_np = _squeeze_silence(audio_np, max_gap_s=self.max_pause)
                    tape.append(audio_np)
            except Exception as e:
                gen_error[0] = e
            finally:
                tape.finish()

        t = threading.Thread(target=producer, daemon=True)
        t.start()

        # 0.1s at 24kHz. We write in small sub-chunks and poll stop_event
        # between them so an interrupt is honored within ~100ms WITHOUT ever
        # touching the PortAudio stream from another thread (doing so segfaults
        # CoreAudio). The stream's stop()/close() happen here on this same
        # thread via the context-manager exit, which is the only safe place.
        SUBCHUNK = 2400

        def stopped():
            return stop_event is not None and stop_event.is_set()

        def paused():
            return pause_event is not None and pause_event.is_set()

        def handle_pause(output):
            # If paused, stop the audio device (safe — same thread) and hold
            # until resumed or stopped, then restart so playback continues from
            # exactly where it left off.
            if not paused():
                return
            output.stop()
            while paused() and not stopped():
                sd.sleep(100)
            if not stopped():
                output.start()

        pos = 0          # absolute playback position in samples (into the tape)
        written = 0      # total samples sent to the device (counts replays too)
        outcome = "ok"
        completed = False
        with sd.OutputStream(samplerate=24000, channels=1, dtype='float32') as output:
            try:
                while not stopped():
                    handle_pause(output)
                    if stopped():
                        break

                    # Apply any pending rewind/forward. Rewind clamps at 0;
                    # forward clamps at the generation frontier (can't skip past
                    # audio that doesn't exist yet).
                    if seek is not None:
                        delta = seek.pop()
                        if delta:
                            total = tape.read(0, 0)[1]
                            pos = max(0, min(pos + delta, total))

                    sub, total, done = tape.read(pos, SUBCHUNK)
                    if sub.size == 0:
                        if done and pos >= total:
                            completed = True
                            break
                        # At the live edge: generation hasn't caught up yet.
                        sd.sleep(50)
                        continue
                    output.write(sub)
                    pos += len(sub)
                    written += len(sub)
                # Drain: closing the stream discards audio still in PortAudio's
                # buffer, clipping the final ~second of speech. On normal
                # completion, append a generous silence pad and wait long enough
                # for the real tail to actually play out before the stream closes.
                if completed and not stopped():
                    output.write(np.zeros(12000, dtype=np.float32))  # 0.5s silence
                    sd.sleep(800)
                elif stopped():
                    outcome = "interrupted"
            except sd.PortAudioError as e:
                outcome = f"PortAudioError:{e}"
                if not stopped():
                    raise

        t.join(timeout=5)
        if gen_error[0] is not None and not stopped():
            raise gen_error[0]
        print(f"Done [{written/24000:.1f}s played, {outcome}]", flush=True)


class TTSClient:
    """Manages TTS via origin server or local Piper."""

    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.backend = self.config.get('backend', 'origin')

        if self.backend == 'origin':
            import websockets  # noqa: F401 - verify available
            origin_cfg = self.config.get('origin', {})
            self.server_url = origin_cfg.get('tts_server_url', 'ws://localhost:8088')
            self.websocket = None
        elif self.backend == 'local':
            local_cfg = self.config.get('local', {})
            self.local_tts = LocalTTS(local_cfg)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _load_config(self, config_path: str = None) -> dict:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = Path(__file__).parent / 'config.yaml'

        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        return {}

    def get_clipboard_text(self) -> str:
        """Get text from clipboard."""
        result = subprocess.run(['pbpaste'], capture_output=True, text=True)
        return result.stdout.strip()

    # --- Origin backend ---

    async def _origin_connect(self):
        import websockets
        print(f"Connecting to {self.server_url}...")
        self.websocket = await websockets.connect(
            self.server_url, max_size=50*1024*1024
        )
        print("Connected to TTS server")

    async def _origin_disconnect(self):
        if self.websocket:
            await self.websocket.close()
            self.websocket = None

    async def _origin_speak(self, text: str, skip_summary: bool = False):
        print(f"Sending text ({len(text)} chars)...")
        request = {"text": text, "skip_summary": skip_summary}
        await self.websocket.send(json.dumps(request))

        audio_data = None

        async for message in self.websocket:
            if isinstance(message, bytes):
                audio_data = message
            else:
                data = json.loads(message)
                if data.get("type") == "audio_start":
                    speech_text = data.get("text", "")
                    print(f"Speaking: {speech_text[:100]}...")
                elif data.get("type") == "audio_complete":
                    break
                elif "error" in data:
                    print(f"Error: {data['error']}")
                    return

        if audio_data:
            _play_wav(audio_data)

    # --- Public API ---

    async def connect(self):
        if self.backend == 'origin':
            await self._origin_connect()

    async def disconnect(self):
        if self.backend == 'origin':
            await self._origin_disconnect()

    async def speak(self, text: str, skip_summary: bool = False):
        if self.backend == 'origin':
            await self._origin_speak(text, skip_summary)
        else:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self.local_tts.synthesize_and_play, text)


def _play_wav(wav_data: bytes):
    """Play WAV audio data through speakers."""
    try:
        wav_buffer = io.BytesIO(wav_data)
        with wave.open(wav_buffer, 'rb') as wav_file:
            sample_rate = wav_file.getframerate()
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            audio_bytes = wav_file.readframes(wav_file.getnframes())

        if sample_width == 2:
            audio = np.frombuffer(audio_bytes, dtype=np.int16)
        else:
            audio = np.frombuffer(audio_bytes, dtype=np.int8)

        audio = audio.astype(np.float32) / 32768.0

        if n_channels > 1:
            audio = audio.reshape(-1, n_channels)

        print(f"Playing audio ({len(audio)/sample_rate:.1f}s)...")
        sd.play(audio, sample_rate)
        sd.wait()
        print("Done")

    except Exception as e:
        print(f"Audio playback error: {e}")


async def main():
    """Main entry point for TTS client."""
    import argparse
    parser = argparse.ArgumentParser(description="TTS client")
    parser.add_argument("--config", help="Path to config.yaml")
    parser.add_argument("--server", help="Override server URL (origin backend)")
    parser.add_argument("--backend", choices=["origin", "local"], help="Override backend")
    parser.add_argument("--text", help="Text to speak (default: clipboard)")
    parser.add_argument("--raw", action="store_true", help="Skip summarization")
    args = parser.parse_args()

    client = TTSClient(config_path=args.config)
    if args.backend:
        client.backend = args.backend
    if args.server and client.backend == 'origin':
        client.server_url = args.server

    print(f"Backend: {client.backend}")

    if args.text:
        text = args.text
    else:
        text = client.get_clipboard_text()
        if not text:
            print("Clipboard is empty")
            sys.exit(1)

    try:
        await client.connect()
        await client.speak(text, skip_summary=args.raw)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
