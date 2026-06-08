-- Hammerspoon configuration
-- Voice input toggle for Claude Code

-- Paths
local voiceScriptsPath = "/Users/gabagool/Git/nvidia_parakeet/client/scripts"

-- State tracking
local voiceRecording = false
local menubarItem = nil
local recordingTask = nil

-- Toggle voice recording with Option+V
hs.hotkey.bind({"alt"}, "v", function()
    if voiceRecording then
        -- Stop recording
        print("Stopping voice recording...")

        -- Send stop signal. stdout is exactly the transcription; paste it.
        local stopScript = voiceScriptsPath .. "/stop_voice.sh"
        hs.task.new("/bin/bash", function(exitCode, stdOut, stdErr)
            local text = (stdOut or ""):gsub("^%s+", ""):gsub("%s+$", "")
            if text ~= "" then
                -- Paste via clipboard but preserve the user's existing clipboard
                local saved = hs.pasteboard.getContents()
                hs.pasteboard.setContents(text)
                hs.eventtap.keyStroke({"cmd"}, "v")
                hs.timer.doAfter(0.3, function()
                    if saved ~= nil then hs.pasteboard.setContents(saved) end
                end)
            end
        end, {stopScript}):start()

        -- Remove menubar indicator
        if menubarItem then
            menubarItem:delete()
            menubarItem = nil
        end

        voiceRecording = false
        hs.notify.new({title="Voice Input", informativeText="Transcribing & pasting..."}):send()
    else
        -- Start recording
        print("Starting voice recording...")

        -- Create menubar indicator
        menubarItem = hs.menubar.new()
        if menubarItem then
            menubarItem:setTitle("REC")
            menubarItem:setTooltip("Voice recording in progress - Option+V to stop")
        end

        -- Start recording script
        local startScript = voiceScriptsPath .. "/start_voice.sh"
        recordingTask = hs.task.new("/bin/bash", function(exitCode, stdOut, stdErr)
            print("Start script finished: " .. (stdOut or "") .. (stdErr or ""))
        end, {startScript})
        recordingTask:start()

        voiceRecording = true
        hs.notify.new({title="Voice Input", informativeText="Recording... Option+V to stop"}):send()
    end
end)

-- TTS with Option+S. Copy text (Cmd+C), then press Option+S to hear it.
--   Single press : speak (when idle) -> pause -> resume -> pause ...
--   Double press : (two quick taps within 0.4s) STOP, so the next press
--                  starts fresh on new clipboard text.
-- A single press fires immediately (no latency); a second quick press
-- overrides with a stop.
local lastSPress = 0
local function runTTSScript(name)
    hs.task.new("/bin/bash", function(exitCode, stdOut, stdErr)
        print("TTS script finished: " .. (stdOut or "") .. (stdErr or ""))
    end, {voiceScriptsPath .. "/" .. name}):start()
end

hs.hotkey.bind({"alt"}, "s", function()
    local now = hs.timer.secondsSinceEpoch()
    if now - lastSPress < 0.4 then
        -- Double tap -> stop. Reset so a 3rd press starts a fresh sequence.
        lastSPress = 0
        print("TTS double Option+S -> stop")
        runTTSScript("stop_tts.sh")
        hs.notify.new({title="TTS", informativeText="Stopped"}):send()
    else
        lastSPress = now
        print("TTS Option+S (speak/pause/resume)...")
        runTTSScript("speak_clipboard.sh")
    end
end)

-- Reload config with Cmd+Ctrl+R
hs.hotkey.bind({"cmd", "ctrl"}, "r", function()
    hs.reload()
end)

hs.notify.new({title="Hammerspoon", informativeText="Config loaded. Option+V=voice, Option+S=speak/pause, double Option+S=stop"}):send()
print("Hammerspoon config loaded. Option+V=voice input, Option+S=speak/pause, double Option+S=stop")
