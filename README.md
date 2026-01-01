# Bidirectional Voice for Claude Code

Voice input AND output for terminal sessions using NVIDIA Parakeet ASR + Piper TTS on a local GPU server.

**Talk to Claude, hear Claude talk back.**

---

## Quick Start

| Hotkey | Action |
|--------|--------|
| **Option+V** | Voice Input - speak to type |
| **Option+S** | Voice Output - hear Claude's response |

### Voice Input (Option+V)
1. Press **Option+V** - "REC" appears in menubar
2. Speak your message
3. Press **Option+V** again - text copied to clipboard
4. **Cmd+V** to paste into terminal

### Voice Output (Option+S)
1. Claude responds in terminal
2. Press **Option+S**
3. Hear a spoken summary of the response

**Note:** Auto-capture works in iTerm2 only. For other terminals (VS Code, Terminal.app), copy the response first (Cmd+C), then press Option+S.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              Mac (Client)                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Option+V (Voice Input)              Option+S (Voice Output)           │
│         │                                    │                          │
│         ▼                                    ▼                          │
│   voice_client.py                    speak_clipboard.sh                 │
│   - Mic capture                      - Capture terminal text            │
│   - VAD filter                       - Send to TTS server               │
│   - Stream audio                     - Play audio response              │
│         │                                    │                          │
└─────────┼────────────────────────────────────┼──────────────────────────┘
          │                                    │
          ▼                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         GPU Server (Origin)                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ASR Server :8087                    TTS Server :8088                  │
│   ┌─────────────────┐                 ┌─────────────────┐               │
│   │ Parakeet Model  │                 │ vLLM → Piper    │               │
│   │ (120M params)   │                 │ Summarize → TTS │               │
│   │ Audio → Text    │                 │ Text → Audio    │               │
│   └─────────────────┘                 └─────────────────┘               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Components

### Voice Input (STT)
- **Model:** nvidia/parakeet_realtime_eou_120m-v1
- **Server:** `server/server.py` on port 8087
- **Client:** `client/voice_client.py`
- **Features:** VAD filtering, utterance-based transcription

### Voice Output (TTS)
- **Summarizer:** vLLM (your existing instance on port 8086)
- **TTS:** Piper (en_US-lessac-medium voice)
- **Server:** `server/tts_server.py` on port 8088
- **Client:** `client/tts_client.py`
- **Features:** Auto-captures Claude response from terminal, summarizes technical content

---

## Server Commands

```bash
# ASR Server (Voice Input)
ssh YOUR_SERVER "systemctl --user status parakeet-asr.service"
ssh YOUR_SERVER "systemctl --user restart parakeet-asr.service"

# TTS Server (Voice Output)
ssh YOUR_SERVER "systemctl --user status tts-server.service"
ssh YOUR_SERVER "systemctl --user restart tts-server.service"

# View logs
ssh YOUR_SERVER "journalctl --user -u parakeet-asr.service -f"
ssh YOUR_SERVER "journalctl --user -u tts-server.service -f"
```

---

## File Structure

```
nvidia_parakeet/
├── README.md
├── PLAN.md
├── client/
│   ├── voice_client.py      # STT: Audio capture → server
│   ├── tts_client.py        # TTS: Receive audio → playback
│   ├── config.yaml          # Server URLs
│   └── scripts/
│       ├── start_voice.sh   # Option+V start
│       ├── stop_voice.sh    # Option+V stop
│       └── speak_clipboard.sh  # Option+S handler
└── server/
    ├── server.py            # ASR WebSocket server
    ├── tts_server.py        # TTS WebSocket server
    ├── parakeet-asr.service # ASR systemd service
    └── tts-server.service   # TTS systemd service
```

---

## Configuration

### Client (`client/config.yaml`)
```yaml
# Voice Input (STT)
server_url: "ws://YOUR_SERVER_IP:8087"
sample_rate: 16000
chunk_duration: 0.1

# Voice Output (TTS)
tts_server_url: "ws://YOUR_SERVER_IP:8088"
```

### Hammerspoon (`~/.hammerspoon/init.lua`)
- Option+V: Toggle voice recording
- Option+S: Speak Claude's response
- Cmd+Ctrl+R: Reload config

**Security Note:** Hammerspoon requires Accessibility permissions to capture hotkeys. This config only listens for specific hotkeys (not all keystrokes) and only communicates with your local server. Recommendations:
- Don't install untrusted Hammerspoon plugins ("Spoons")
- Periodically verify `~/.hammerspoon/init.lua` hasn't been modified
- Hammerspoon is open-source and widely trusted in the Mac community

---

## TTS Summarization

The TTS server uses vLLM to convert technical Claude responses into natural speech:

**Input (technical):**
```
The backpropagation algorithm computes dL/dw = dL/da * da/dz * dz/dw
using the chain rule. ReLU(Wx + b) activations flow forward...
```

**Output (spoken):**
> "The network learns by calculating how wrong each prediction was
> and adjusting weights to reduce errors over time."

This makes complex responses listenable without losing meaning.

---

## Development Notes

### STT Challenges Solved
1. **Event loop errors** - Captured asyncio loop before thread callback
2. **Hallucination on silence** - Added energy-based VAD filter
3. **Fragmented output** - Switched to utterance-based transcription
4. **Garbled terminal** - Clipboard-only output for safety

### TTS Challenges Solved
1. **Terminal capture** - AppleScript to read iTerm2 content
2. **Unicode encoding** - Explicit UTF-8 for special characters
3. **Response isolation** - Extract only latest Claude response
4. **Technical jargon** - vLLM summarizes before TTS

---

## Models

| Component | Model | Size | Latency |
|-----------|-------|------|---------|
| STT | nvidia/parakeet_realtime_eou_120m-v1 | 120M | ~200ms |
| Summarizer | vLLM (your model) | varies | ~1s |
| TTS | Piper lessac-medium | 63MB | ~100ms |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No audio captured | Check mic permissions in System Preferences |
| ASR server down | `systemctl --user restart parakeet-asr.service` |
| TTS server down | `systemctl --user restart tts-server.service` |
| Option+S no sound | Check speaker volume, verify TTS service running |
| Wrong response spoken | Terminal capture may include old text, scroll down first |
