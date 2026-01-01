# Parakeet Voice Input for Claude Code

Voice-to-text input for terminal sessions using NVIDIA Parakeet ASR on a local GPU server.

---

## Quick Start Guide

### Usage (3 Steps)

1. **Option+V** - Start recording (REC appears in menubar)
2. **Speak** your message
3. **Option+V** - Stop recording → text copied to clipboard
4. **Cmd+V** - Paste into terminal

### Manual Mode (without hotkey)

```bash
cd ~/Git/nvidia_parakeet
source venv/bin/activate
python client/voice_client.py
# Speak, then Ctrl+C to stop
# Result copied to clipboard, Cmd+V to paste
```

### Server Commands

```bash
# Check status (SSH to your GPU server)
ssh YOUR_SERVER "systemctl --user status parakeet-asr.service"

# Restart
ssh YOUR_SERVER "systemctl --user restart parakeet-asr.service"

# View logs
ssh YOUR_SERVER "journalctl --user -u parakeet-asr.service -f"
```

---

## Architecture Overview

```
┌─────────────────┐        ┌─────────────────────────────────┐
│  Mac (Client)   │        │   GPU Server (YOUR_SERVER_IP)   │
├─────────────────┤        ├─────────────────────────────────┤
│ Hammerspoon     │        │ Parakeet ASR Model              │
│ (Option+V)      │        │ nvidia/parakeet_realtime_eou    │
│       ↓         │        │ 120M params, GPU accelerated    │
│ voice_client.py │───────▶│ WebSocket server :8087          │
│ - Mic capture   │ audio  │ - Receives audio chunks         │
│ - VAD filter    │ stream │ - Transcribes on finalize       │
│ - WebSocket     │◀───────│ - Returns text                  │
│       ↓         │  text  │                                 │
│ Clipboard       │        │ systemd service (auto-start)    │
└─────────────────┘        └─────────────────────────────────┘
```

---

## Server Code (`server/server.py`)

The server runs on a Linux machine with NVIDIA GPU.

### Key Components

**1. Model Loading**
```python
asr_model = nemo_asr.models.ASRModel.from_pretrained(
    model_name="nvidia/parakeet_realtime_eou_120m-v1"
)
asr_model.eval().cuda()
```

**2. WebSocket Handler**
- Accepts connections on port 8087
- Receives binary audio data (16-bit PCM, 16kHz mono)
- Accumulates audio in buffer
- Transcribes on "finalize" command

**3. Transcription**
- Saves audio buffer to temp WAV file
- Calls NeMo `model.transcribe()`
- Returns text with `<EOU>` (end-of-utterance) token stripped

### Why Utterance-Based (Not Streaming)

Initially tried streaming (transcribe every 160ms), but:
- Model lacks context → fragmented output
- "hello how are you" became "hello" ... "are" ... "how"

Solution: Accumulate all audio, transcribe once at end. Model gets full context → accurate transcription.

---

## Client Code (`client/voice_client.py`)

The client runs on Mac, captures audio, streams to server.

### Key Components

**1. Audio Capture**
```python
stream = sd.InputStream(
    samplerate=16000,
    channels=1,
    dtype='float32',
    blocksize=chunk_size,
    callback=audio_callback
)
```

**2. Voice Activity Detection (VAD)**
```python
rms = np.sqrt(np.mean(indata ** 2))
if rms > 0.005:  # Only send if energy exceeds threshold
    # Send audio chunk
```

Why VAD? Without it, silent audio causes model to hallucinate ("yeah", "four", etc).

**3. WebSocket Streaming**
- Sends audio chunks as binary data
- Sends `{"command": "finalize"}` on stop
- Receives transcription JSON

**4. Clipboard Output**
```python
subprocess.run(['pbcopy'], input=text.encode())
```

Safe approach - user manually pastes with Cmd+V. Prevents accidental command execution.

---

## Development Saga: Fixing Bad Recognition

### Problem 1: Event Loop Error

```
RuntimeError: There is no current event loop in thread 'Dummy-1'
```

**Cause:** Sounddevice callback runs in separate thread without asyncio loop.

**Fix:** Capture loop reference before callback:
```python
loop = asyncio.get_running_loop()  # Main thread
# In callback:
loop.call_soon_threadsafe(queue.put_nowait, data)
```

---

### Problem 2: Hallucination on Silence

```
Transcription: yeah
Transcription: yeah
Transcription: yeah
```

**Cause:** ASR models hallucinate on low-energy/silent input.

**Fix:** Energy-based VAD filter:
```python
rms = np.sqrt(np.mean(indata ** 2))
if rms > 0.005:  # Skip silent chunks
    send_audio()
```

---

### Problem 3: Fragmented Output

```
Transcription: hello
Transcription: are
Transcription: hello
Transcription: how
```

**Cause:** Transcribing small 160ms chunks independently. No context.

**Fix:** Utterance-based approach:
- Accumulate ALL audio during session
- Only transcribe on finalize (user stops speaking)
- Model gets full context → coherent output

```python
# Before (bad):
if len(buffer) >= 160ms:
    transcribe_and_return()

# After (good):
def add_audio():
    buffer.extend(audio)
    return None  # Don't transcribe yet

def finalize():
    return transcribe(buffer)  # Full context
```

---

### Problem 4: Garbled Terminal Output

Text auto-typed into terminal, accidentally executed commands.

**Fix:** Clipboard-only mode:
```python
subprocess.run(['pbcopy'], input=text.encode())
print("Copied to clipboard, Cmd+V to paste")
```

User controls when/where to paste.

---

## File Structure

```
nvidia_parakeet/
├── README.md
├── PLAN.md
├── venv/                    # Mac Python venv
├── client/
│   ├── voice_client.py     # Audio capture + WebSocket
│   ├── config.yaml         # Server URL
│   ├── requirements.txt
│   └── scripts/
│       ├── start_voice.sh  # Hammerspoon calls this
│       └── stop_voice.sh
└── server/
    ├── server.py           # WebSocket ASR server
    ├── requirements.txt
    └── parakeet-asr.service

# On GPU Server:
~/parakeet-asr/
├── venv/
├── server.py
└── scripts/
    └── start_server.sh
```

---

## Configuration

### Client (`client/config.yaml`)
```yaml
# Copy config.yaml.example to config.yaml and update
server_url: "ws://YOUR_SERVER_IP:8087"
sample_rate: 16000
chunk_duration: 0.1
```

### Hammerspoon (`~/.hammerspoon/init.lua`)
- Hotkey: Option+V (toggle recording)
- Shows "REC" in menubar
- Reload: Cmd+Ctrl+R

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No audio captured | Check mic permissions in System Preferences |
| Server not responding | `ssh YOUR_SERVER "systemctl --user restart parakeet-asr.service"` |
| Poor recognition | Speak clearly, reduce background noise, use full sentences |
| VAD too sensitive | Adjust threshold in `voice_client.py` (default: 0.005) |

---

## Model Info

**nvidia/parakeet_realtime_eou_120m-v1**
- 120M parameters
- FastConformer-RNNT architecture
- Optimized for real-time voice agents
- Native end-of-utterance detection
- ~9% word error rate
- Requires 16kHz mono audio
