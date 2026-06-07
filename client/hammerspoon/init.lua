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

-- Speak SELECTED text with Option+S (TTS) — without polluting the clipboard
local function speakText(text)
    local args = {voiceScriptsPath .. "/speak_clipboard.sh"}
    if text and text ~= "" then
        -- Passed via the task args array (execve), so no shell escaping needed;
        -- newlines/quotes/unicode in the selection are preserved verbatim.
        table.insert(args, text)
    end
    hs.task.new("/bin/bash", function(exitCode, stdOut, stdErr)
        print("Speak script finished: " .. (stdOut or "") .. (stdErr or ""))
    end, args):start()
    hs.notify.new({title="TTS", informativeText="Speaking selection..."}):send()
end

hs.hotkey.bind({"alt"}, "s", function()
    print("Speaking selection via TTS...")

    -- 1. Preferred: read the selection directly via the Accessibility API.
    --    This never touches the clipboard at all.
    local elem = hs.uielement.focusedElement()
    local sel = elem and elem:selectedText()
    if sel and sel ~= "" then
        speakText(sel)
        return
    end

    -- 2. Fallback (terminals etc. that don't expose AXSelectedText): copy the
    --    selection with Cmd+C, but save and restore the clipboard around it so
    --    the user's existing clipboard contents are preserved.
    local saved = hs.pasteboard.getContents()
    hs.pasteboard.clearContents()
    hs.eventtap.keyStroke({"cmd"}, "c")
    hs.timer.doAfter(0.15, function()
        local copied = hs.pasteboard.getContents()
        if saved ~= nil then
            hs.pasteboard.setContents(saved)
        else
            hs.pasteboard.clearContents()
        end
        speakText(copied)
    end)
end)

-- Reload config with Cmd+Ctrl+R
hs.hotkey.bind({"cmd", "ctrl"}, "r", function()
    hs.reload()
end)

hs.notify.new({title="Hammerspoon", informativeText="Config loaded. Option+V=voice input, Option+S=speak"}):send()
print("Hammerspoon config loaded. Voice input: Option+V, TTS: Option+S")
