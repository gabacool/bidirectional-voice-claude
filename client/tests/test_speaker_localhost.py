"""Localhost-only speaker endpoints on the LAN voice API."""

import http.client
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

import voice_api
from voice_api import is_loopback_host, SPEAKER_PATHS, tts_download_path


def test_loopback_ipv4():
    assert is_loopback_host("127.0.0.1")
    assert is_loopback_host("localhost")
    assert is_loopback_host("LOCALHOST")


def test_loopback_ipv6():
    assert is_loopback_host("::1")
    assert is_loopback_host("::ffff:127.0.0.1")


def test_lan_hosts_are_not_loopback():
    assert not is_loopback_host("192.168.1.50")
    assert not is_loopback_host("10.0.0.2")
    assert not is_loopback_host("0.0.0.0")
    assert not is_loopback_host("")


def test_speaker_paths_are_the_option_s_set():
    assert SPEAKER_PATHS == (
        '/speak', '/stop', '/seek/back', '/seek/forward', '/save',
    )


def test_tts_download_path_names_engine_and_timestamp(tmp_path):
    p = tts_download_path(
        "breeze",
        now=datetime(2026, 9, 20, 11, 3, 5),
        downloads=tmp_path,
    )
    assert p == tmp_path / "breeze-20260920-110305.wav"


def test_tts_download_path_avoids_collision(tmp_path):
    first = tts_download_path(
        "vireo",
        now=datetime(2026, 9, 20, 11, 3, 5),
        downloads=tmp_path,
    )
    first.write_bytes(b"x")
    second = tts_download_path(
        "vireo",
        now=datetime(2026, 9, 20, 11, 3, 5),
        downloads=tmp_path,
    )
    assert second == tmp_path / "vireo-20260920-110305-2.wav"


class _SaveFakeTTS:
    engine = "breeze"
    speaker = "lock"

    def synthesize_to_array(self, text):
        assert "hello" in text.lower()
        return np.full(2400, 0.25, dtype=np.float32)


def test_save_writes_wav_under_downloads(tmp_path):
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts-infer")
    server = voice_api.ThreadedHTTPServer(("127.0.0.1", 0), voice_api.VoiceAPIHandler)
    server.tts = _SaveFakeTTS()
    server.tts_executor = executor
    server.speak_lock = threading.Lock()
    server.tts_downloads = tmp_path
    server.config_mtime = Path(voice_api.CONFIG_PATH).stat().st_mtime
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        port = server.server_address[1]
        body = b"Hello from the clipboard."
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("POST", "/save", body=body, headers={"Content-Length": str(len(body))})
        resp = conn.getresponse()
        payload = resp.read().decode()
        assert resp.status == 200, payload
        out = Path(payload.strip())
        assert out.parent == tmp_path
        assert out.name.startswith("breeze-")
        assert out.suffix == ".wav"
        assert out.stat().st_size > 44
        with open(out, "rb") as f:
            assert f.read(4) == b"RIFF"
        import wave
        with wave.open(str(out), "rb") as w:
            assert w.getframerate() == 24000
            assert w.getnchannels() == 1
            assert w.getsampwidth() == 2
            assert w.getnframes() == 2400
    finally:
        server.shutdown()
        executor.shutdown(wait=False)
