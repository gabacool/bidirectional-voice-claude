#!/usr/bin/env python3
"""
WebSocket TTS Server using Piper TTS with vLLM summarization.
Receives text, summarizes via vLLM, generates speech via Piper, streams audio back.
"""

import asyncio
import json
import logging
import io
import os
import wave
import tempfile
from pathlib import Path
from typing import Optional

import aiohttp
import websockets

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8086/v1/chat/completions")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "")  # Will use default model on server
PIPER_VOICE = os.environ.get("PIPER_VOICE", "/home/YOUR_USER/parakeet-asr/voices/en-us-lessac-medium.onnx")

# Global Piper instance
piper_voice = None


def load_piper():
    """Load Piper TTS voice model."""
    global piper_voice
    try:
        from piper import PiperVoice

        # Piper will download the voice model automatically if not present
        logger.info(f"Loading Piper voice: {PIPER_VOICE}")
        piper_voice = PiperVoice.load(PIPER_VOICE)
        logger.info("Piper TTS loaded successfully")
    except ImportError:
        logger.error("piper-tts not installed. Run: pip install piper-tts")
        raise
    except Exception as e:
        logger.error(f"Failed to load Piper voice: {e}")
        raise


SUMMARIZE_PROMPT = """Summarize this Claude Code response for spoken output.
Rules:
- Convert code blocks to brief descriptions like "I wrote a Python function that does X"
- Skip ASCII diagrams, just describe what they show in one sentence
- Keep it conversational, 2-4 sentences max
- No markdown formatting, code syntax, or special characters in output
- Speak naturally as if explaining to someone verbally

Response to summarize:
{text}

Spoken summary:"""


async def summarize_for_speech(text: str) -> str:
    """Use vLLM to summarize text for speech output."""
    # If text is short and simple (no code blocks), skip summarization
    if len(text) < 200 and "```" not in text and not any(c in text for c in ['|', '─', '│', '┌', '└']):
        # Clean up markdown for simple responses
        clean = text.replace("**", "").replace("*", "").replace("`", "").strip()
        return clean

    try:
        payload = {
            "messages": [
                {"role": "user", "content": SUMMARIZE_PROMPT.format(text=text)}
            ],
            "max_tokens": 200,
            "temperature": 0.3
        }

        if VLLM_MODEL:
            payload["model"] = VLLM_MODEL

        async with aiohttp.ClientSession() as session:
            async with session.post(VLLM_URL, json=payload, timeout=30) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    summary = data["choices"][0]["message"]["content"].strip()
                    logger.info(f"Summarized: {len(text)} chars -> {len(summary)} chars")
                    return summary
                else:
                    error = await resp.text()
                    logger.warning(f"vLLM returned {resp.status}: {error}")
                    # Fallback: strip markdown manually
                    return _manual_cleanup(text)
    except asyncio.TimeoutError:
        logger.warning("vLLM timeout, using manual cleanup")
        return _manual_cleanup(text)
    except Exception as e:
        logger.warning(f"vLLM error: {e}, using manual cleanup")
        return _manual_cleanup(text)


def _manual_cleanup(text: str) -> str:
    """Manual cleanup when vLLM is unavailable."""
    import re

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

    # Truncate if too long
    if len(text) > 500:
        text = text[:500] + "... and more."

    return text.strip()


def generate_audio(text: str) -> bytes:
    """Generate WAV audio from text using Piper."""
    if not piper_voice:
        raise RuntimeError("Piper not loaded")

    # Generate audio to in-memory buffer
    audio_buffer = io.BytesIO()

    with wave.open(audio_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(piper_voice.config.sample_rate)

        # Synthesize speech - piper returns AudioChunk objects
        for audio_chunk in piper_voice.synthesize(text):
            # Get int16 bytes from the chunk
            audio_bytes = audio_chunk.audio_int16_bytes
            wav_file.writeframes(audio_bytes)

    audio_buffer.seek(0)
    return audio_buffer.read()


async def handle_client(websocket):
    """Handle a single WebSocket client connection."""
    client_addr = websocket.remote_address
    logger.info(f"TTS client connected: {client_addr}")

    try:
        async for message in websocket:
            if isinstance(message, str):
                try:
                    data = json.loads(message)
                    text = data.get("text", "")
                    skip_summary = data.get("skip_summary", False)
                except json.JSONDecodeError:
                    # Plain text message
                    text = message
                    skip_summary = False

                if not text:
                    await websocket.send(json.dumps({"error": "No text provided"}))
                    continue

                logger.info(f"Received text: {len(text)} chars")

                # Summarize for speech (unless skipped)
                if skip_summary:
                    speech_text = text
                else:
                    speech_text = await summarize_for_speech(text)

                logger.info(f"Generating audio for: {speech_text[:100]}...")

                # Generate audio
                try:
                    audio_data = await asyncio.get_event_loop().run_in_executor(
                        None, generate_audio, speech_text
                    )

                    # Send metadata first
                    await websocket.send(json.dumps({
                        "type": "audio_start",
                        "format": "wav",
                        "size": len(audio_data),
                        "text": speech_text
                    }))

                    # Send audio data
                    await websocket.send(audio_data)

                    # Send completion
                    await websocket.send(json.dumps({"type": "audio_complete"}))

                    logger.info(f"Sent audio: {len(audio_data)} bytes")

                except Exception as e:
                    logger.error(f"Audio generation error: {e}")
                    await websocket.send(json.dumps({"error": str(e)}))

    except websockets.exceptions.ConnectionClosed:
        logger.info(f"TTS client disconnected: {client_addr}")
    except Exception as e:
        logger.error(f"Error handling TTS client {client_addr}: {e}")


async def main(host: str = "0.0.0.0", port: int = 8088):
    """Start the WebSocket TTS server."""
    load_piper()

    logger.info(f"Starting WebSocket TTS server on ws://{host}:{port}")
    logger.info(f"vLLM endpoint: {VLLM_URL}")
    logger.info(f"Piper voice: {PIPER_VOICE}")

    async with websockets.serve(handle_client, host, port, max_size=10*1024*1024):
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="WebSocket TTS Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8088, help="Port to listen on")
    parser.add_argument("--vllm-url", default=None, help="vLLM API URL")
    parser.add_argument("--voice", default=None, help="Piper voice model")
    args = parser.parse_args()

    if args.vllm_url:
        VLLM_URL = args.vllm_url
    if args.voice:
        PIPER_VOICE = args.voice

    asyncio.run(main(args.host, args.port))
