#!/usr/bin/env python3
"""
TTS client for Claude Code voice output.
Reads text from clipboard, sends to TTS server, plays audio response.
"""

import asyncio
import json
import subprocess
import sys
import io
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
import websockets
import yaml


class TTSClient:
    """Manages TTS request and audio playback."""

    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.server_url = self.config.get('tts_server_url', 'ws://localhost:8088')
        self.websocket = None

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

    async def connect(self):
        """Connect to the TTS server."""
        print(f"Connecting to {self.server_url}...")
        self.websocket = await websockets.connect(
            self.server_url,
            max_size=50*1024*1024  # 50MB for audio
        )
        print("Connected to TTS server")

    async def disconnect(self):
        """Disconnect from the TTS server."""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None

    async def speak(self, text: str, skip_summary: bool = False):
        """Send text to TTS server and play the audio response."""
        if not text:
            print("No text to speak")
            return

        print(f"Sending text ({len(text)} chars)...")

        # Send text request
        request = {
            "text": text,
            "skip_summary": skip_summary
        }
        await self.websocket.send(json.dumps(request))

        # Wait for response
        audio_data = None
        speech_text = None

        async for message in self.websocket:
            if isinstance(message, bytes):
                # Audio data
                audio_data = message
            else:
                # JSON message
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
            self._play_audio(audio_data)

    def _play_audio(self, wav_data: bytes):
        """Play WAV audio data."""
        try:
            # Parse WAV data
            wav_buffer = io.BytesIO(wav_data)
            with wave.open(wav_buffer, 'rb') as wav_file:
                sample_rate = wav_file.getframerate()
                n_channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                n_frames = wav_file.getnframes()

                # Read audio data
                audio_bytes = wav_file.readframes(n_frames)

            # Convert to numpy array
            if sample_width == 2:
                audio = np.frombuffer(audio_bytes, dtype=np.int16)
            else:
                audio = np.frombuffer(audio_bytes, dtype=np.int8)

            # Convert to float for sounddevice
            audio = audio.astype(np.float32) / 32768.0

            # Reshape for channels
            if n_channels > 1:
                audio = audio.reshape(-1, n_channels)

            # Play audio (blocking)
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
    parser.add_argument("--server", help="Override server URL")
    parser.add_argument("--text", help="Text to speak (default: clipboard)")
    parser.add_argument("--raw", action="store_true", help="Skip summarization")
    args = parser.parse_args()

    client = TTSClient(config_path=args.config)
    if args.server:
        client.server_url = args.server

    # Get text
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
