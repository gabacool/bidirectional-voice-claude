# Bidirectional Voice for Claude Code

## Overview
Real-time voice input AND output for iTerm2/Claude Code sessions using:
- **STT:** NVIDIA Parakeet ASR (speech-to-text)
- **TTS:** Piper + vLLM summarization (text-to-speech)

Both running on local GPU server.

---

# Part 1: Voice Input (STT)

## Selected Model
**nvidia/parakeet_realtime_eou_120m-v1**
- 120M params, 80-160ms latency
- Native streaming + EOU detection
- 9.3% average WER

## Architecture

```
[Mac: iTerm2] <-- paste text
      ^
      |
[Mac: Voice Client] -- audio stream --> [Origin: ASR Server :8087]
      |                                        |
  (Option+V hotkey)                     (parakeet model)
```

---

## Phase 1.1: ASR Server Setup (Origin)

**Location:** `~/parakeet-asr/`

### Directory Structure
```
~/parakeet-asr/
├── venv/
├── server.py          # WebSocket ASR server
├── requirements.txt
└── scripts/
    └── start_server.sh
```

### Dependencies
- nemo_toolkit[asr] (includes PyTorch, CUDA support)
- websockets
- soundfile

### Implementation
- Load `parakeet_realtime_eou_120m-v1` model on startup
- Accept WebSocket connections on port 8087
- Receive 16kHz mono audio chunks
- Accumulate audio, transcribe on "finalize" command
- Return text with `<EOU>` token stripped

### Systemd Service
- File: `~/.config/systemd/user/parakeet-asr.service`
- Auto-restart on failure

---

## Phase 1.2: Voice Client Setup (Mac)

**Location:** `~/nvidia_parakeet/client/`

### Directory Structure
```
client/
├── voice_client.py    # Audio capture + WebSocket
├── config.yaml        # Server URL
├── requirements.txt
└── scripts/
    ├── start_voice.sh
    └── stop_voice.sh
```

### Dependencies
- sounddevice
- websockets
- pyyaml
- numpy

### Implementation
- Capture microphone audio (16kHz, mono)
- Energy-based VAD filter (threshold: 0.005)
- Stream to server via WebSocket
- Copy transcription to clipboard on finalize

### Hammerspoon Hotkey
- **Option+V:** Toggle recording
- Shows "REC" in menubar while active

---

## Phase 1.3: Challenges Solved

| Problem | Cause | Solution |
|---------|-------|----------|
| Event loop error | Callback in separate thread | Capture `asyncio.get_running_loop()` before callback |
| Hallucination on silence | ASR hallucinates on low energy | VAD filter: `rms > 0.005` |
| Fragmented output | Small chunks lack context | Utterance-based: transcribe all at once |
| Garbled terminal | Auto-typing caused issues | Clipboard-only output |

---

# Part 2: Voice Output (TTS)

## Selected Models
- **Summarizer:** vLLM (existing instance on port 8086)
- **TTS:** Piper `en_US-lessac-medium` (63MB, ~100ms latency)

## Architecture

```
[Mac: Terminal] -- capture text --> [speak_clipboard.sh]
                                           |
                                           v
                                    [tts_client.py]
                                           |
                                           v
[Mac: Speaker] <-- audio stream -- [Origin: TTS Server :8088]
                                           |
                                    ┌──────┴──────┐
                                    v             v
                              [vLLM :8086]  [Piper TTS]
                              (summarize)   (synthesize)
```

---

## Phase 2.1: TTS Server Setup (Origin)

**Location:** `~/parakeet-asr/`

### New Files
```
~/parakeet-asr/
├── tts_server.py      # WebSocket TTS server
├── tts-server.service # Systemd service
└── voices/
    └── en-us-lessac-medium.onnx  # Piper voice model
```

### Dependencies (added to requirements.txt)
- piper-tts>=1.2.0
- aiohttp>=3.9.0

### Implementation
- Accept WebSocket connections on port 8088
- Receive text from client
- If text > 200 chars or contains code: call vLLM to summarize
- Pass summary to Piper TTS
- Stream WAV audio back to client

### Summarization Prompt
```
Summarize this Claude Code response for spoken output.
Rules:
- Convert code blocks to brief descriptions
- Skip ASCII diagrams, describe what they show
- Keep it conversational, 2-4 sentences max
- No markdown formatting in output
```

### Systemd Service
- File: `~/.config/systemd/user/tts-server.service`
- Port 8088, runs alongside ASR service

---

## Phase 2.2: TTS Client Setup (Mac)

**Location:** `~/nvidia_parakeet/client/`

### New Files
```
client/
├── tts_client.py           # Audio playback client
└── scripts/
    └── speak_clipboard.sh  # Terminal capture + TTS trigger
```

### Implementation

**speak_clipboard.sh:**
1. Capture iTerm2 terminal content via AppleScript
2. Extract last Claude response (lines starting with ⏺)
3. Send to TTS server via tts_client.py
4. Play received audio

**tts_client.py:**
1. Read text from clipboard
2. Connect to TTS server via WebSocket
3. Send text, receive WAV audio
4. Play audio using sounddevice

### Hammerspoon Hotkey
- **Option+S:** Speak Claude's last response

---

## Phase 2.3: Challenges Solved

| Problem | Cause | Solution |
|---------|-------|----------|
| Piper API mismatch | `synthesize_stream_raw` doesn't exist | Use `synthesize()` returning AudioChunk |
| Voice model not found | Piper doesn't auto-download | Manual download to `voices/` directory |
| Terminal capture empty | Unicode ⏺ not matching | Set `LANG=en_US.UTF-8`, use `\u23fa` |
| Multiple responses captured | Grabbing too many lines | Only take `response_lines[-1]` |
| Technical jargon unlistenable | Code/formulas hard to speak | vLLM summarizes before TTS |

---

# Files Summary

| File | Location | Purpose |
|------|----------|---------|
| `server.py` | server/ | ASR WebSocket server |
| `tts_server.py` | server/ | TTS WebSocket server |
| `parakeet-asr.service` | server/ | ASR systemd service |
| `tts-server.service` | server/ | TTS systemd service |
| `voice_client.py` | client/ | STT: Audio capture |
| `tts_client.py` | client/ | TTS: Audio playback |
| `speak_clipboard.sh` | client/scripts/ | Terminal capture + TTS trigger |
| `config.yaml` | client/ | Server URLs |

---

# Hotkeys

| Hotkey | Action |
|--------|--------|
| **Option+V** | Voice Input - toggle recording |
| **Option+S** | Voice Output - speak last response |
| **Cmd+Ctrl+R** | Reload Hammerspoon config |

---

# Success Criteria

## Voice Input (STT)
1. Press Option+V, see "REC" in menubar
2. Speak naturally
3. Press Option+V again
4. Text copied to clipboard
5. Latency < 500ms

## Voice Output (TTS)
1. Claude responds in terminal
2. Press Option+S
3. Hear spoken summary of response
4. Technical content converted to natural speech
5. Latency < 2 seconds
