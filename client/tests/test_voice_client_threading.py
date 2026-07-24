"""Option+V recorder: all MLX work must happen on ONE thread.

mlx (>=0.31) GPU streams are thread-local. Loading the STT model on one thread
and running inference on another raises

    RuntimeError: There is no Stream(gpu, 1) in current thread.

which `main()` swallowed into "Error: ..." + exit(1), leaving the transcription
file empty — so Option+V announced "Transcribing & pasting..." and pasted
nothing. `voice_api.py` already pins load+inference to a single "infer" worker;
these tests pin the recorder to the same invariant, without a mic or a model.
"""

import asyncio
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

import voice_client


class FakeTranscriber:
    """Stands in for LocalTranscriber, recording which thread called it."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.load_thread: int | None = None
        self.infer_thread: int | None = None

    def _ensure_model(self) -> None:
        self.load_thread = threading.get_ident()

    def transcribe(self, audio_float32, sample_rate: int) -> str:
        self.infer_thread = threading.get_ident()
        return "hello from the fake model"


class FakeInputStream:
    """sounddevice.InputStream stand-in: feeds a few loud chunks, then stops."""

    def __init__(self, *, callback, blocksize, **kwargs):
        self._callback = callback
        self._blocksize = blocksize

    def __enter__(self):
        # Loud enough to clear the RMS silence guard in the recorder.
        chunk = np.full((self._blocksize, 1), 0.5, dtype=np.float32)
        for _ in range(3):
            self._callback(chunk, self._blocksize, None, None)
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def recorder(monkeypatch, tmp_path):
    """Run voice_client.main() with a fake mic and a fake model."""
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({
        "backend": "local",
        "sample_rate": 16000,
        "chunk_duration": 0.1,
        "max_record_seconds": 1,          # recorder self-stops; no signal needed
        "local": {"stt_model": "fake/model"},
    }))
    paste_file = tmp_path / "transcription.txt"
    built = []

    monkeypatch.setattr(voice_client, "LocalTranscriber",
                        lambda model_name: built.append(FakeTranscriber(model_name)) or built[-1])
    monkeypatch.setattr(voice_client.sd, "InputStream", FakeInputStream)
    monkeypatch.setattr(voice_client, "TRANSCRIPTION_FILE", str(paste_file))
    monkeypatch.setattr("sys.argv", ["voice_client.py", "--config", str(config)])

    def run():
        asyncio.run(voice_client.main())
        return SimpleNamespace(transcriber=built[-1], paste_file=paste_file)

    return run


def test_model_loads_and_infers_on_the_same_thread(recorder):
    """The regression: load thread != inference thread killed transcription."""
    result = recorder()

    assert result.transcriber.load_thread is not None, "model was never loaded"
    assert result.transcriber.infer_thread is not None, "transcription never ran"
    assert result.transcriber.load_thread == result.transcriber.infer_thread, (
        "MLX model loaded on one thread and used on another — "
        "this is the Stream(gpu, N) failure that broke Option+V"
    )


def test_inference_runs_off_the_event_loop_thread(recorder):
    """Same-thread must come from a dedicated worker, not from blocking the
    asyncio loop, which also serves the stop signal and the max-record timer."""
    result = recorder()

    assert result.transcriber.infer_thread != threading.main_thread().ident


def test_transcription_reaches_the_paste_file(recorder):
    """End of the chain: stop_voice.sh cats this file and Hammerspoon pastes it."""
    result = recorder()

    assert result.paste_file.read_text() == "hello from the fake model"
