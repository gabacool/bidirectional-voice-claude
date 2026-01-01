# Voice Input for Claude Code via NVIDIA Parakeet

## Overview
Real-time voice-to-text input for iTerm2/Claude Code sessions using NVIDIA Parakeet ASR running on Origin server.

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
[Mac: Voice Client] -- audio stream --> [Origin: ASR Server]
      |                                        |
  (keyboard shortcut)                   (parakeet model)
```

**Components:**
1. **GPU Server**: ASR service with WebSocket API
2. **Mac Client**: Audio capture + hotkey daemon

---

## Implementation Plan

### Phase 1: Origin Server Setup

**Location:** `YOUR_USER@YOUR_SERVER:~/parakeet-asr/`

#### 1.1 Create Python venv and install dependencies
```
~/parakeet-asr/
├── venv/
├── server.py          # WebSocket ASR server
├── requirements.txt
└── scripts/
    └── start_server.sh
```

**Dependencies:**
- nemo_toolkit[asr] (includes PyTorch, CUDA support)
- websockets
- soundfile

#### 1.2 Implement WebSocket ASR server
- Load `parakeet_realtime_eou_120m-v1` model on startup
- Accept WebSocket connections on port 8087
- Receive 16kHz mono audio chunks
- Stream transcriptions back, emit final on `<EOU>`
- Handle multiple concurrent connections

#### 1.3 Create systemd service for auto-start
- Service file: `/etc/systemd/user/parakeet-asr.service`
- Auto-restart on failure

---

### Phase 2: Mac Client Setup

**Location:** `~/nvidia_parakeet/` (your local clone)

#### 2.1 Create Python venv and install dependencies
```
nvidia_parakeet/
├── venv/
├── voice_client.py    # Audio capture + WebSocket client
├── requirements.txt
├── config.yaml        # Server address, hotkey, audio settings
└── scripts/
    ├── install.sh
    └── voice-input.sh  # Entry script for hotkey
```

**Dependencies:**
- sounddevice (or pyaudio)
- websockets
- pyyaml

#### 2.2 Implement voice client
- Capture microphone audio (16kHz, mono)
- Stream to Origin server via WebSocket
- Buffer incoming transcriptions
- On `<EOU>`: paste final text into active terminal via AppleScript

#### 2.3 Global hotkey setup via Hammerspoon
- Install Hammerspoon via Homebrew (new installation)
- Configure toggle hotkey: `Option+V` (⌥V)
- First press: start recording, show menubar indicator
- Second press: stop recording, paste result, hide indicator

**Why Hammerspoon over Karabiner:**
- Karabiner = key remapping (what you have for MS apps)
- Hammerspoon = automation/scripting (run Python, show UI, toggle states)
- Both can coexist, no conflicts

**Hammerspoon config:**
```lua
-- ~/.hammerspoon/init.lua
local voiceRecording = false
local menubarItem = nil

hs.hotkey.bind({"alt"}, "v", function()
    if voiceRecording then
        -- Stop recording
        hs.task.new("/path/to/stop_voice.sh", nil):start()
        menubarItem:delete()
        voiceRecording = false
    else
        -- Start recording
        hs.task.new("/path/to/start_voice.sh", nil):start()
        menubarItem = hs.menubar.new():setTitle("🎤")
        voiceRecording = true
    end
end)
```

---

### Phase 3: Integration & Testing

#### 3.1 End-to-end test
- Start server on Origin
- Run client on Mac
- Press hotkey, speak, verify text appears in terminal

#### 3.2 Error handling
- Server unavailable: show notification, fail gracefully
- Audio device issues: prompt user
- Network latency: buffer handling

---

## Files to Create

| File | Location | Purpose |
|------|----------|---------|
| `server.py` | Origin: `~/parakeet-asr/` | WebSocket ASR server |
| `requirements.txt` | Origin: `~/parakeet-asr/` | Server dependencies |
| `parakeet-asr.service` | Origin: `/etc/systemd/user/` | Systemd service |
| `voice_client.py` | Mac: `nvidia_parakeet/` | Audio capture client |
| `requirements.txt` | Mac: `nvidia_parakeet/` | Client dependencies |
| `config.yaml` | Mac: `nvidia_parakeet/` | Configuration |
| `init.lua` (edit) | Mac: `~/.hammerspoon/` | Hotkey binding |

---

## Hotkey: Option+V (⌥V)

Currently types `√` symbol - will be intercepted by Hammerspoon before reaching any app.

---

## Success Criteria

1. Press `Option+V` in any terminal
2. See microphone indicator in menubar
3. Speak naturally
4. Press `Option+V` again to stop
5. Transcribed text appears at cursor position
6. Latency < 500ms end-to-end
