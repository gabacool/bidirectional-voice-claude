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
import wave
from pathlib import Path

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


class LocalTTS:
    """Synthesize speech locally using Qwen3-TTS via mlx-audio."""

    def __init__(self, config: dict):
        self.model_name = config.get('tts_model', 'mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit')
        self.speaker = config.get('tts_speaker', 'aiden')
        self.language = config.get('tts_language', 'english')
        self.instruct = config.get('tts_instruct')  # None means no instruction
        self.temperature = config.get('tts_temperature', 0.9)
        self.top_k = config.get('tts_top_k', 50)
        self.top_p = config.get('tts_top_p', 1.0)
        self.repetition_penalty = config.get('tts_repetition_penalty', 1.05)
        self.max_tokens = config.get('tts_max_tokens', 4096)
        self.streaming_interval = config.get('tts_streaming_interval', 2.0)
        self._model = None
        self._stream = None

    def _ensure_model(self):
        """Lazy-load the TTS model on first use."""
        if self._model is not None:
            return
        from mlx_audio.tts.utils import load_model
        print(f"Loading TTS model: {self.model_name}...")
        self._model = load_model(self.model_name)
        print("TTS model loaded")

    def synthesize_and_play(self, text: str, stop_event=None):
        """Synthesize text to speech and play it, streaming chunks as they generate."""
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

        audio_queue = queue.Queue()
        gen_error = [None]

        def producer():
            try:
                for chunk in self._model.generate_custom_voice(**kwargs):
                    if stop_event is not None and stop_event.is_set():
                        break
                    audio_np = np.array(chunk.audio, dtype=np.float32)
                    if audio_np.size == 0:
                        continue
                    audio_queue.put(audio_np)
            except Exception as e:
                gen_error[0] = e
            finally:
                audio_queue.put(None)

        t = threading.Thread(target=producer, daemon=True)
        t.start()

        with sd.OutputStream(samplerate=24000, channels=1, dtype='float32') as output:
            self._stream = output
            try:
                while True:
                    chunk = audio_queue.get()
                    if chunk is None:
                        break
                    if stop_event is not None and stop_event.is_set():
                        break
                    output.write(chunk)
            except sd.PortAudioError:
                if stop_event is None or not stop_event.is_set():
                    raise
            finally:
                self._stream = None

        t.join(timeout=5)
        if gen_error[0] is not None and (stop_event is None or not stop_event.is_set()):
            raise gen_error[0]
        print("Done")

    def interrupt(self):
        stream = self._stream
        if stream is not None:
            try:
                stream.abort()
            except Exception:
                pass


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
