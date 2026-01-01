#!/usr/bin/env python3
"""
Voice input client for Claude Code.
Captures microphone audio, streams to Parakeet ASR server, and pastes result into terminal.
"""

import asyncio
import json
import subprocess
import sys
import signal
import os
from pathlib import Path

import numpy as np
import sounddevice as sd
import websockets
import yaml


class VoiceClient:
    """Manages voice recording, streaming, and terminal paste."""

    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.sample_rate = self.config.get('sample_rate', 16000)
        self.channels = 1
        self.chunk_duration = self.config.get('chunk_duration', 0.1)  # 100ms chunks
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        self.server_url = self.config.get('server_url', 'ws://localhost:8087')
        self.recording = False
        self.websocket = None
        self.transcription = ""

    def _load_config(self, config_path: str = None) -> dict:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = Path(__file__).parent / 'config.yaml'

        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        return {}

    async def connect(self):
        """Connect to the ASR server."""
        print(f"Connecting to {self.server_url}...")
        self.websocket = await websockets.connect(
            self.server_url,
            max_size=10*1024*1024
        )
        print("Connected to ASR server")

    async def disconnect(self):
        """Disconnect from the ASR server."""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None

    async def record_and_transcribe(self):
        """Record audio and stream to ASR server."""
        self.recording = True
        self.transcription = ""
        audio_queue = asyncio.Queue()

        # Capture event loop reference for use in callback thread
        loop = asyncio.get_running_loop()

        chunks_sent = [0]  # Use list for mutable in closure

        def audio_callback(indata, frames, time, status):
            """Called by sounddevice for each audio chunk."""
            if status:
                print(f"Audio status: {status}", file=sys.stderr)
            if self.recording:
                # Simple energy-based VAD: only send if audio energy exceeds threshold
                rms = np.sqrt(np.mean(indata ** 2))
                if rms > 0.005:  # Lower threshold for voice activity
                    # Convert float32 to int16
                    audio_int16 = (indata[:, 0] * 32767).astype('<i2')
                    loop.call_soon_threadsafe(
                        audio_queue.put_nowait, audio_int16.tobytes()
                    )
                    chunks_sent[0] += 1
                    if chunks_sent[0] % 10 == 0:  # Print every 10 chunks
                        print(f"\r[Audio chunks sent: {chunks_sent[0]}]", end='', flush=True)

        # Start audio stream
        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype='float32',
            blocksize=self.chunk_size,
            callback=audio_callback
        )

        # Receive transcriptions in background
        receive_task = asyncio.create_task(self._receive_transcriptions())

        try:
            with stream:
                print("Recording... (press Ctrl+C or send SIGUSR1 to stop)")
                while self.recording:
                    try:
                        audio_data = await asyncio.wait_for(
                            audio_queue.get(),
                            timeout=0.5
                        )
                        await self.websocket.send(audio_data)
                    except asyncio.TimeoutError:
                        continue

        finally:
            self.recording = False

        # Send finalize command
        print(f"\n[Finalizing... sent {chunks_sent[0]} chunks total]")
        await self.websocket.send(json.dumps({"command": "finalize"}))

        # Wait for final transcription
        print("[Waiting for transcription...]")
        try:
            # Wait for the transcription response with timeout
            for _ in range(20):  # 2 second timeout
                await asyncio.sleep(0.1)
                if self.transcription:
                    break
        finally:
            receive_task.cancel()
            try:
                await receive_task
            except asyncio.CancelledError:
                pass

        print()  # Clean newline
        return self.transcription

    async def _receive_transcriptions(self):
        """Receive transcriptions from the server."""
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
                        print()  # New line after final transcription
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"\nError receiving transcription: {e}", file=sys.stderr)

    def stop_recording(self, signum=None, frame=None):
        """Stop recording (signal handler)."""
        self.recording = False

    @staticmethod
    def paste_to_terminal(text: str):
        """Copy text to clipboard for manual paste."""
        if not text:
            return

        # Copy to clipboard - user can paste with Cmd+V
        subprocess.run(['pbcopy'], input=text.encode(), check=True)
        print(f"Copied to clipboard: {text}")
        print("Press Cmd+V to paste")


async def main():
    """Main entry point for voice client."""
    import argparse
    parser = argparse.ArgumentParser(description="Voice input client")
    parser.add_argument("--config", help="Path to config.yaml")
    parser.add_argument("--server", help="Override server URL")
    parser.add_argument("--no-paste", action="store_true", help="Don't paste, just print")
    args = parser.parse_args()

    client = VoiceClient(config_path=args.config)
    if args.server:
        client.server_url = args.server

    # Setup signal handlers
    signal.signal(signal.SIGINT, client.stop_recording)
    signal.signal(signal.SIGUSR1, client.stop_recording)

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
