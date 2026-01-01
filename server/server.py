#!/usr/bin/env python3
"""
WebSocket ASR Server using NVIDIA Parakeet realtime EOU model.
Accepts 16kHz mono audio chunks and returns transcriptions.
"""

import asyncio
import json
import logging
import struct
import numpy as np
import torch
import websockets
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global model instance (loaded once at startup)
asr_model = None


def load_model():
    """Load the Parakeet realtime EOU model."""
    global asr_model
    import nemo.collections.asr as nemo_asr

    logger.info("Loading nvidia/parakeet_realtime_eou_120m-v1...")
    asr_model = nemo_asr.models.ASRModel.from_pretrained(
        model_name="nvidia/parakeet_realtime_eou_120m-v1"
    )
    asr_model.eval()

    if torch.cuda.is_available():
        asr_model = asr_model.cuda()
        logger.info(f"Model loaded on CUDA: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("CUDA not available, using CPU")

    return asr_model


class StreamingSession:
    """Manages a single streaming ASR session."""

    def __init__(self, model):
        self.model = model
        self.audio_buffer = []
        self.sample_rate = 16000
        self.min_chunk_samples = int(0.16 * self.sample_rate)  # 160ms minimum

    def add_audio(self, audio_bytes: bytes) -> Optional[str]:
        """
        Add audio data to buffer. Returns None - only transcribe on finalize.
        Audio format: 16-bit PCM, 16kHz, mono
        """
        # Convert bytes to numpy array
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0

        self.audio_buffer.extend(audio_float32.tolist())

        # Don't transcribe incrementally - wait for finalize
        # This gives the model full context for better accuracy
        return None

    def _transcribe(self) -> str:
        """Transcribe accumulated audio buffer."""
        if not self.audio_buffer:
            return ""

        import tempfile
        import soundfile as sf

        # Save audio buffer to temporary WAV file
        audio_array = np.array(self.audio_buffer, dtype=np.float32)

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
            sf.write(temp_path, audio_array, self.sample_rate)

        try:
            # Transcribe using file path (NeMo preferred method)
            with torch.inference_mode():
                output = self.model.transcribe([temp_path])

            if output and len(output) > 0:
                # Handle different return formats
                if hasattr(output[0], 'text'):
                    return output[0].text
                elif isinstance(output[0], str):
                    return output[0]
            return ""
        finally:
            import os
            os.unlink(temp_path)

    def finalize(self) -> str:
        """Get final transcription and reset buffer."""
        result = self._transcribe()
        self.audio_buffer = []
        return result

    def reset(self):
        """Reset the session state."""
        self.audio_buffer = []


async def handle_client(websocket):
    """Handle a single WebSocket client connection."""
    client_addr = websocket.remote_address
    logger.info(f"Client connected: {client_addr}")

    session = StreamingSession(asr_model)

    try:
        async for message in websocket:
            if isinstance(message, bytes):
                # Audio data
                text = session.add_audio(message)
                if text:
                    # Check for EOU token
                    is_final = "<EOU>" in text
                    clean_text = text.replace("<EOU>", "").strip()

                    response = {
                        "type": "transcription",
                        "text": clean_text,
                        "is_final": is_final
                    }
                    await websocket.send(json.dumps(response))

                    if is_final:
                        session.reset()

            elif isinstance(message, str):
                # Control message
                data = json.loads(message)
                cmd = data.get("command")

                if cmd == "finalize":
                    # Force finalization
                    text = session.finalize()
                    clean_text = text.replace("<EOU>", "").strip()
                    response = {
                        "type": "transcription",
                        "text": clean_text,
                        "is_final": True
                    }
                    await websocket.send(json.dumps(response))

                elif cmd == "reset":
                    session.reset()
                    await websocket.send(json.dumps({"type": "reset", "status": "ok"}))

                elif cmd == "ping":
                    await websocket.send(json.dumps({"type": "pong"}))

    except websockets.exceptions.ConnectionClosed:
        logger.info(f"Client disconnected: {client_addr}")
    except Exception as e:
        logger.error(f"Error handling client {client_addr}: {e}")
    finally:
        session.reset()


async def main(host: str = "0.0.0.0", port: int = 8087):
    """Start the WebSocket ASR server."""
    load_model()

    logger.info(f"Starting WebSocket ASR server on ws://{host}:{port}")
    async with websockets.serve(handle_client, host, port, max_size=10*1024*1024):
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="WebSocket ASR Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8087, help="Port to listen on")
    args = parser.parse_args()

    asyncio.run(main(args.host, args.port))
