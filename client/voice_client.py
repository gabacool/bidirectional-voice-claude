#!/usr/bin/env python3
"""
Voice input client for Claude Code.
Captures microphone audio, transcribes via Parakeet ASR, and pastes result into terminal.
Supports two backends: "origin" (GPU server via WebSocket) or "local" (Apple Silicon via MLX).
"""

import asyncio
import json
import subprocess
import sys
import signal
import tempfile
from pathlib import Path

import numpy as np
import sounddevice as sd
import yaml


class LocalTranscriber:
    """Transcribe audio locally using parakeet-mlx on Apple Silicon."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def _ensure_model(self):
        """Lazy-load the model on first use."""
        if self._model is not None:
            return
        from parakeet_mlx import from_pretrained
        print(f"Loading STT model: {self.model_name}...")
        self._model = from_pretrained(self.model_name)
        print("STT model loaded")

    def transcribe(self, audio_float32: np.ndarray, sample_rate: int) -> str:
        """Transcribe audio buffer to text."""
        self._ensure_model()

        # Write audio to temp WAV file (parakeet-mlx expects a file path)
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
            sf.write(temp_path, audio_float32, sample_rate)

        try:
            result = self._model.transcribe(temp_path)
            # AlignedResult.text is already correctly spaced — parakeet tokens
            # carry their own leading space, so the library joins them with "".
            # Joining tokens with " " ourselves inserts spurious mid-word spaces
            # ("H ello", "he ar"), so prefer .text and never space-join tokens.
            if hasattr(result, 'text') and result.text:
                return result.text.strip()
            if hasattr(result, 'sentences') and result.sentences:
                return ''.join(
                    ''.join(t.text for t in s.tokens if hasattr(t, 'text'))
                    for s in result.sentences
                ).strip()
            if isinstance(result, str):
                return result.strip()
            return str(result).strip()
        finally:
            import os
            os.unlink(temp_path)


class VoiceClient:
    """Manages voice recording, transcription, and terminal paste."""

    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.sample_rate = self.config.get('sample_rate', 16000)
        self.channels = 1
        self.chunk_duration = self.config.get('chunk_duration', 0.1)
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        self.backend = self.config.get('backend', 'origin')
        self.recording = False
        self.transcription = ""

        # Backend-specific init
        if self.backend == 'origin':
            import websockets  # noqa: F401 - verify available
            origin_cfg = self.config.get('origin', {})
            self.server_url = origin_cfg.get('server_url', 'ws://localhost:8087')
            self.websocket = None
        elif self.backend == 'local':
            local_cfg = self.config.get('local', {})
            model_name = local_cfg.get('stt_model', 'mlx-community/parakeet-tdt-0.6b-v3')
            self.local_transcriber = LocalTranscriber(model_name)
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

    # --- Origin backend methods ---

    async def _origin_connect(self):
        """Connect to the remote ASR server."""
        import websockets
        print(f"Connecting to {self.server_url}...")
        self.websocket = await websockets.connect(
            self.server_url,
            max_size=10*1024*1024
        )
        print("Connected to ASR server")

    async def _origin_disconnect(self):
        """Disconnect from the remote ASR server."""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None

    async def _origin_record_and_transcribe(self):
        """Record audio and stream to remote ASR server."""
        self.recording = True
        self.transcription = ""
        audio_queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        chunks_sent = [0]

        def audio_callback(indata, frames, time, status):
            if status:
                print(f"Audio status: {status}", file=sys.stderr)
            if self.recording:
                # Stream ALL audio contiguously. Per-chunk energy gating drops
                # quiet speech and breaks the server's continuous-stream
                # end-of-utterance detection.
                audio_int16 = (indata[:, 0] * 32767).astype('<i2')
                loop.call_soon_threadsafe(
                    audio_queue.put_nowait, audio_int16.tobytes()
                )
                chunks_sent[0] += 1
                if chunks_sent[0] % 10 == 0:
                    print(f"\r[Audio chunks sent: {chunks_sent[0]}]", end='', flush=True)

        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype='float32',
            blocksize=self.chunk_size,
            callback=audio_callback
        )

        receive_task = asyncio.create_task(self._origin_receive_transcriptions())

        try:
            with stream:
                print("Recording... (press Ctrl+C or send SIGUSR1 to stop)")
                while self.recording:
                    try:
                        audio_data = await asyncio.wait_for(
                            audio_queue.get(), timeout=0.5
                        )
                        await self.websocket.send(audio_data)
                    except asyncio.TimeoutError:
                        continue
        finally:
            self.recording = False

        print(f"\n[Finalizing... sent {chunks_sent[0]} chunks total]")
        await self.websocket.send(json.dumps({"command": "finalize"}))

        print("[Waiting for transcription...]")
        try:
            for _ in range(20):
                await asyncio.sleep(0.1)
                if self.transcription:
                    break
        finally:
            receive_task.cancel()
            try:
                await receive_task
            except asyncio.CancelledError:
                pass

        print()
        return self.transcription

    async def _origin_receive_transcriptions(self):
        """Receive transcriptions from the remote server."""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                if data.get('type') == 'transcription':
                    text = data.get('text', '')
                    is_final = data.get('is_final', False)
                    if text:
                        self.transcription = text
                        print(f"\rTranscription: {text}", end='', flush=True)
                    if is_final:
                        print()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"\nError receiving transcription: {e}", file=sys.stderr)

    # --- Local backend methods ---

    async def _local_record_and_transcribe(self):
        """Record audio locally and transcribe via parakeet-mlx."""
        self.recording = True
        audio_buffer = []
        loop = asyncio.get_running_loop()
        chunks_captured = [0]

        def audio_callback(indata, frames, time, status):
            if status:
                print(f"Audio status: {status}", file=sys.stderr)
            if self.recording:
                # Capture ALL audio contiguously. Do NOT gate per-chunk by energy —
                # that drops quiet consonants/word boundaries and splices
                # non-adjacent audio together, which wrecks recognition. Silence
                # is handled once, on the whole buffer, after recording stops.
                loop.call_soon_threadsafe(
                    audio_buffer.append, indata[:, 0].copy()
                )
                chunks_captured[0] += 1
                if chunks_captured[0] % 10 == 0:
                    print(f"\r[Audio chunks captured: {chunks_captured[0]}]", end='', flush=True)

        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype='float32',
            blocksize=self.chunk_size,
            callback=audio_callback
        )

        try:
            with stream:
                print("Recording... (press Ctrl+C or send SIGUSR1 to stop)")
                while self.recording:
                    await asyncio.sleep(0.1)
        finally:
            self.recording = False

        if not audio_buffer:
            print("\n[No audio captured]")
            return ""

        audio_float32 = np.concatenate(audio_buffer)

        # Anti-hallucination guard: Parakeet invents text on pure silence.
        # Check energy over the WHOLE recording (not per-chunk) so real speech
        # with quiet passages is preserved.
        overall_rms = float(np.sqrt(np.mean(audio_float32 ** 2)))
        if overall_rms < 0.003:
            print(f"\n[Silence detected (RMS {overall_rms:.4f}) - nothing to transcribe]")
            return ""

        print(f"\n[Transcribing {chunks_captured[0]} chunks locally...]")

        # Run transcription in executor to avoid blocking the event loop
        transcription = await loop.run_in_executor(
            None, self.local_transcriber.transcribe, audio_float32, self.sample_rate
        )

        if transcription:
            print(f"Transcription: {transcription}")
        return transcription

    # --- Public API ---

    async def connect(self):
        """Connect to backend (only needed for origin)."""
        if self.backend == 'origin':
            await self._origin_connect()

    async def disconnect(self):
        """Disconnect from backend (only needed for origin)."""
        if self.backend == 'origin':
            await self._origin_disconnect()

    async def record_and_transcribe(self):
        """Record audio and transcribe using configured backend."""
        if self.backend == 'origin':
            return await self._origin_record_and_transcribe()
        else:
            return await self._local_record_and_transcribe()

    def stop_recording(self, signum=None, frame=None):
        """Stop recording (signal handler)."""
        self.recording = False

    @staticmethod
    def paste_to_terminal(text: str):
        """Copy text to clipboard for manual paste."""
        if not text:
            return
        subprocess.run(['pbcopy'], input=text.encode(), check=True)
        print(f"Copied to clipboard: {text}")
        print("Press Cmd+V to paste")


async def main():
    """Main entry point for voice client."""
    import argparse
    parser = argparse.ArgumentParser(description="Voice input client")
    parser.add_argument("--config", help="Path to config.yaml")
    parser.add_argument("--server", help="Override server URL (origin backend)")
    parser.add_argument("--backend", choices=["origin", "local"], help="Override backend")
    parser.add_argument("--no-paste", action="store_true", help="Don't paste, just print")
    args = parser.parse_args()

    # Set signal handlers early — before model loading which can take time.
    # SIGUSR1 with default handler terminates the process, so we must
    # install our own handler before any slow init.
    stop_flag = [False]

    def early_stop(signum, frame):
        stop_flag[0] = True

    signal.signal(signal.SIGINT, early_stop)
    signal.signal(signal.SIGUSR1, early_stop)

    client = VoiceClient(config_path=args.config)
    if args.backend:
        client.backend = args.backend
    if args.server and client.backend == 'origin':
        client.server_url = args.server

    print(f"Backend: {client.backend}")

    # Now that client exists, point signal handlers at it
    signal.signal(signal.SIGINT, client.stop_recording)
    signal.signal(signal.SIGUSR1, client.stop_recording)

    # If stop was requested during init, exit cleanly
    if stop_flag[0]:
        print("Stop requested during startup")
        return

    # Load model before recording so transcription is fast after stop
    if client.backend == 'local':
        client.local_transcriber._ensure_model()

    if stop_flag[0]:
        print("Stop requested during model load")
        return

    try:
        await client.connect()
        transcription = await client.record_and_transcribe()

        if transcription and not args.no_paste:
            client.paste_to_terminal(transcription)
        elif transcription:
            print(f"Transcription: {transcription}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
