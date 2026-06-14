# Bidirectional Voice for Claude Code

Voice input AND output for terminal sessions using NVIDIA Parakeet ASR + Piper TTS.

**Talk to Claude, hear Claude talk back.**

Two backends: run everything on your Mac (Apple Silicon), or offload to a GPU server.

---

## Quick Start

| Hotkey | Action |
|--------|--------|
| **Option+V** | Voice Input — speak, auto-types where your cursor is |
| **Option+S** | Voice Output — speak / pause / resume the clipboard |
| **Option+S Option+S** | Stop voice output (two quick taps) |
| **Option+<** | Rewind ~15s while speaking |
| **Option+>** | Fast-forward ~15s while speaking |

### Voice Input (Option+V)
1. Press **Option+V** — "REC" appears in menubar
2. Speak your message
3. Press **Option+V** again — the transcription is auto-typed at your cursor

No manual paste needed. Your clipboard is preserved (it pastes via a
save/restore).

### Voice Output (Option+S)
1. Copy any text to clipboard (**Cmd+C**)
2. Press **Option+S** to hear it
3. Press **Option+S** again to **pause**, again to **resume**
4. **Double-tap Option+S** (two quick presses) to **stop** — the next press
   starts fresh on whatever is on your clipboard
5. While it's speaking, **Option+<** rewinds ~15s and **Option+>** jumps ~15s
   ahead. The `<`/`>` keys are Shift+comma/period, so the bindings are on the
   plain comma/period keys (no Shift needed). Rewind reaches all the way back to
   the start; fast-forward is capped at how much audio has been generated so far
   (you can't skip past speech that doesn't exist yet). Tune the step with
   `tts_seek_seconds` in `config.yaml`.

Works from any app — terminal, browser, editor, etc.

**Performance:** First press starts a background daemon that loads the TTS model (~30s). All subsequent presses are near-instant since the model stays in memory. To stop the daemon: `kill $(cat /tmp/tts_daemon.pid)`

---

## LAN Voice API (for remote agents)

`voice_api.py` exposes the same local MLX STT/TTS as two plain HTTP endpoints so a
remote agent (e.g. Hermes on the Origin GPU box) can use them over the LAN. It
runs independently of the Option+S daemon, loads its own copy of both models, and
never touches the Mac's mic or speakers — audio only flows in/out as bytes.

Bound to `0.0.0.0:9900` (set `local.voice_api_port` in `config.yaml`). **LAN only,
no auth** — don't expose it to the public internet.

| Method | Path | Input | Output |
|--------|------|-------|--------|
| POST | `/transcribe` | `multipart/form-data`, field `audio` (ogg/mp3/wav/…) | JSON `{"text": "..."}` |
| POST | `/synthesize` | JSON `{"text": "..."}` | WAV bytes, 24kHz mono 16-bit |
| GET | `/health` | — | `ok` |

```bash
# Speech-to-text
curl -X POST http://192.168.1.162:9900/transcribe -F "audio=@clip.ogg"
# -> {"text": "transcribed text"}

# Text-to-speech (save the WAV)
curl -X POST http://192.168.1.162:9900/synthesize \
     -H 'Content-Type: application/json' \
     -d '{"text":"hello from the agent"}' -o out.wav
```

Uploads are decoded with `ffmpeg`, so any common container works. STT uses
parakeet-mlx; TTS uses the configured Qwen3 voice (`tts_speaker`, etc.). GPU calls
are serialized internally, so concurrent requests are safe (they queue).

**Run it as an always-on service** (LaunchAgent — auto-starts at login, restarts on
crash, runs in the user session so Metal/GPU is available):

```bash
cp client/launchd/com.gabagool.voiceapi.plist ~/Library/LaunchAgents/
launchctl load -w ~/Library/LaunchAgents/com.gabagool.voiceapi.plist
# logs: tail -f /tmp/voice_api.log
# stop: launchctl unload -w ~/Library/LaunchAgents/com.gabagool.voiceapi.plist
```

Or run it by hand: `client/scripts/start_voice_api.sh`.

---

## Backends

### Local (Apple Silicon)

Runs STT and TTS directly on your Mac using MLX. No server dependency.

**Requirements:** Apple Silicon Mac (M1+), Python 3.14+, ffmpeg, ~1.5GB disk for models

**Setup:**
```bash
# Create/activate venv (Python 3.14+)
cd ~/git/nvidia_parakeet
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r client/requirements.txt

# Set backend to local in config.yaml
# (already the default)
```

On first use, models download automatically:
- STT: `parakeet-tdt-0.6b-v3` (~1.2GB from HuggingFace)
- TTS: `en_US-lessac-medium` (~63MB from HuggingFace)

### Origin (GPU Server)

Offloads to a remote GPU server via WebSocket. Lower latency for heavy workloads, requires Origin to be running.

**Setup:**
```bash
# Edit client/config.yaml
backend: origin

origin:
  server_url: "ws://YOUR_SERVER_IP:8087"
  tts_server_url: "ws://YOUR_SERVER_IP:8088"
```

See [Server Commands](#server-commands) for managing the remote services.

### Switching Backends

Edit `client/config.yaml`:

```yaml
# Use local Mac (Apple Silicon)
backend: local

# Or use remote GPU server
backend: origin
```

No other changes needed - the hotkeys and scripts work with either backend.

---

## Architecture

### Local Backend
```
Option+V / Option+S
        |
        v
  voice_client.py / tts_client.py
        |
        v
  parakeet-mlx (STT)    Qwen3-TTS (TTS)
  In-process on Mac      In-process on Mac
```

### Origin Backend
```
  Mac (Client)                     GPU Server (Origin)
  ─────────────                    ────────────────────
  voice_client.py  ──WebSocket──>  ASR Server :8087
                                   (Parakeet 120M, CUDA)

  tts_client.py    ──WebSocket──>  TTS Server :8088
                                   (vLLM + Piper)
```

---

## Configuration

### Client (`client/config.yaml`)
```yaml
backend: local

sample_rate: 16000
chunk_duration: 0.1

origin:
  server_url: "ws://YOUR_SERVER_IP:8087"
  tts_server_url: "ws://YOUR_SERVER_IP:8088"

local:
  stt_model: "mlx-community/parakeet-tdt-0.6b-v3"
  tts_model: "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
  tts_speaker: "aiden"
  tts_language: "english"
  tts_instruct: null
  tts_temperature: 0.9
  tts_top_k: 50
  tts_top_p: 1.0
  tts_repetition_penalty: 1.05
  tts_max_tokens: 4096
  tts_streaming_interval: 2.0
```

**After changing config:** The TTS daemon watches `config.yaml` and auto-reloads
voice/speed/etc. on the next Option+S — no restart needed. The model stays loaded
unless you change `tts_model` itself (then it reloads on the next press).

**Alternative STT models:**

| Model | Params | Notes |
|-------|--------|-------|
| `mlx-community/parakeet-tdt-0.6b-v3` | 600M | Best accuracy, #1 on OpenASR (default) |
| `mlx-community/parakeet-tdt_ctc-110m` | 110M | Lightweight, faster startup |
| `mlx-community/parakeet-tdt-1.1b` | 1.1B | Maximum accuracy |

### Hammerspoon (`~/.hammerspoon/init.lua`)
- Option+V: Toggle voice recording (auto-types the transcription)
- Option+S: Speak the clipboard; press again to pause/resume
- Option+S Option+S: Stop voice output (two quick taps)
- Option+< / Option+>: Rewind / fast-forward ~15s while speaking
- Cmd+Ctrl+R: Reload config

After editing `~/.hammerspoon/init.lua`, reload it (Cmd+Ctrl+R) for changes
to take effect.

**Security Note:** Hammerspoon requires Accessibility permissions to capture hotkeys. This config only listens for specific hotkeys (not all keystrokes) and only communicates with your local server. Recommendations:
- Don't install untrusted Hammerspoon plugins ("Spoons")
- Periodically verify `~/.hammerspoon/init.lua` hasn't been modified
- Hammerspoon is open-source and widely trusted in the Mac community

---

## TTS Customization (Local Backend)

The local backend uses [Qwen3-TTS](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit) via mlx-audio, which supports voice selection, emotion control, and generation tuning.

All parameters are set in `client/config.yaml` under `local:`.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tts_speaker` | `aiden` | Voice identity |
| `tts_language` | `english` | Output language |
| `tts_instruct` | `null` | Emotion/style instruction (free-form text) |
| `tts_temperature` | `0.9` | Randomness. **Keep 0.7–1.0** — below ~0.5 this model emits the first word then collapses to silence |
| `tts_top_k` | `50` | Token sampling pool size |
| `tts_top_p` | `1.0` | Nucleus sampling threshold (0.0-1.0) |
| `tts_repetition_penalty` | `1.05` | Penalize repeated tokens (1.0 = off) |
| `tts_max_tokens` | `4096` | Maximum generation length |
| `tts_streaming_interval` | `2.0` | Seconds of audio per streaming chunk |
| `tts_speed` | `1.0` | Playback speed, pitch preserved (1.5 faster, 0.8 slower) |

### Alternative TTS Models

If speech has gaps or stuttering, try a smaller/faster model in `config.yaml`:

| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit` | 0.8GB | Baseline | Best (default) |
| `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit` | 0.6GB | ~2x faster | Good |
| `mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit` | 0.5GB | ~3x faster | Good |
| `mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit` | 0.4GB | ~4x faster | Acceptable |

Models download automatically on first use. After changing `tts_model`, restart the daemon:
```bash
kill $(cat /tmp/tts_daemon.pid)
```

### Available Speakers

| Speaker | Language | Notes |
|---------|----------|-------|
| `aiden` | English | Male, default |
| `ryan` | English | Male |
| `serena` | Chinese | Female |
| `vivian` | Chinese | Female |
| `dylan` | Chinese (Beijing) | Male |
| `eric` | Chinese (Sichuan) | Male |
| `uncle_fu` | Chinese | Male |
| `ono_anna` | Japanese | Female |
| `sohee` | Korean | Female |

### Examples

**Calm, professional narrator:**
```yaml
local:
  tts_speaker: "aiden"
  tts_instruct: "calm and professional, like a podcast host"
  tts_temperature: 0.7
```

**Excited, energetic delivery:**
```yaml
local:
  tts_speaker: "ryan"
  tts_instruct: "excited and enthusiastic"
  tts_temperature: 1.0
```

**Whispering:**
```yaml
local:
  tts_speaker: "aiden"
  tts_instruct: "whispering softly"
  tts_temperature: 0.8
```

**No emotion control (neutral):**
```yaml
local:
  tts_instruct: null
```

---

## Components

### Voice Input (STT)

| | Local | Origin |
|-|-------|--------|
| **Model** | parakeet-tdt-0.6b-v3 (MLX) | parakeet_realtime_eou_120m-v1 (NeMo) |
| **Params** | 600M | 120M |
| **Inference** | Apple Silicon (MLX) | CUDA (RTX) |
| **Client** | `client/voice_client.py` | `client/voice_client.py` |
| **Server** | N/A (in-process) | `server/server.py` on port 8087 |

### Voice Output (TTS)

| | Local | Origin |
|-|-------|--------|
| **TTS** | Qwen3-TTS 1.7B (MLX) | Piper (on server) |
| **Summarizer** | Text cleanup (regex) | vLLM + text cleanup fallback |
| **Client** | `client/tts_client.py` | `client/tts_client.py` |
| **Server** | N/A (in-process) | `server/tts_server.py` on port 8088 |

---

## Server Commands

For origin backend only:

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
│   ├── voice_client.py      # STT: Audio capture + transcription
│   ├── tts_client.py        # TTS: Text cleanup + synthesis + playback
│   ├── tts_daemon.py        # TTS: Persistent daemon (model stays loaded)
│   ├── voice_api.py         # LAN HTTP API: /transcribe + /synthesize for remote agents
│   ├── config.yaml          # Backend selection + settings
│   ├── hammerspoon/
│   │   └── init.lua         # Hotkey bindings (backup of ~/.hammerspoon/init.lua)
│   ├── launchd/
│   │   └── com.gabagool.voiceapi.plist  # LaunchAgent for the LAN voice API
│   └── scripts/
│       ├── start_voice.sh   # Option+V start
│       ├── stop_voice.sh    # Option+V stop
│       ├── speak_clipboard.sh  # Option+S handler (speak/pause/resume)
│       ├── stop_tts.sh      # Double Option+S handler (stop)
│       ├── seek_back.sh     # Option+< handler (rewind ~15s)
│       ├── seek_forward.sh  # Option+> handler (forward ~15s)
│       └── start_voice_api.sh  # Launches voice_api.py (used by the LaunchAgent)
└── server/
    ├── server.py            # ASR WebSocket server (origin)
    ├── tts_server.py        # TTS WebSocket server (origin)
    ├── parakeet-asr.service # ASR systemd service
    └── tts-server.service   # TTS systemd service
```

---

## TTS Summarization

On origin backend, the TTS server uses vLLM to convert technical Claude responses into natural speech:

**Input (technical):**
```
The backpropagation algorithm computes dL/dw = dL/da * da/dz * dz/dw
using the chain rule. ReLU(Wx + b) activations flow forward...
```

**Output (spoken):**
> "The network learns by calculating how wrong each prediction was
> and adjusting weights to reduce errors over time."

On local backend, text is cleaned up with regex (code blocks removed, markdown stripped) before synthesis.

---

## Models

| Component | Local (MLX) | Origin (CUDA) |
|-----------|-------------|---------------|
| STT | parakeet-tdt-0.6b-v3 (600M) | parakeet_realtime_eou_120m-v1 (120M) |
| Summarizer | Regex cleanup | vLLM |
| TTS | Qwen3-TTS 1.7B 8-bit (~3GB) | Piper lessac-medium (63MB) |

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

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No audio captured | Check mic permissions in System Preferences |
| STT model download slow | First run downloads ~1.2GB model, subsequent runs are instant |
| Piper voice not found | Delete `client/voices/` and retry (re-downloads) |
| ASR server down (origin) | `systemctl --user restart parakeet-asr.service` |
| TTS server down (origin) | `systemctl --user restart tts-server.service` |
| Option+S no sound | Check speaker volume, verify backend is configured |
| TTS only speaks the first word then stops | `tts_temperature` is too low — set it to 0.7–1.0 (below ~0.5 the model collapses to silence) |
| Hotkey change has no effect | Reload Hammerspoon: Cmd+Ctrl+R |
