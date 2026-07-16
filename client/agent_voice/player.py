"""Sequential streaming TTS playback engine.

Groups are enqueued as sentences stream in and are spoken strictly in order.
Each group's PCM is STREAMED from the synthesizer while it plays (chunk in,
chunk out), so first audio per group is the server's TTFB (~0.25 s) rather
than the group's full generation time. Timing-sensitive collaborators
(``fetch_pcm``, ``write_pcm``, ``sleep``) are injected, so the queue / token /
error logic tests with hand-rolled fakes.

A single long-lived worker thread owns playback. It blocks on a Condition
while idle and is woken by ``enqueue`` / ``stop_all``; ``enqueue`` is therefore
a pure append + notify with no blocking thread start, so a synchronous burst of
enqueues reliably lands in the queue as one drain (one ``on_idle``) instead of
racing a freshly spawned pump.
"""

import threading
import time
from collections.abc import Callable, Iterator


class Player:
    def __init__(
        self,
        fetch_pcm: Callable[[str], Iterator[bytes]],
        write_pcm: Callable[[bytes], None],
        pause_ms: Callable[[], int] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self._fetch_pcm = fetch_pcm
        self._write_pcm = write_pcm
        self._pause_ms = pause_ms or (lambda: 0)
        self._sleep = sleep or time.sleep
        self._cond = threading.Condition()
        self._queue: list[str] = []
        self._running = False
        # Monotonic run token: stop_all() bumps it so a pump parked in a fetch
        # or write bails at the next chunk boundary without touching new state.
        self._token = 0
        self.on_idle: Callable[[], None] | None = None
        self.on_error: Callable[[Exception, str], None] | None = None
        # One persistent worker, parked on the Condition until there is work.
        self._worker = threading.Thread(target=self._pump, daemon=True)
        self._worker.start()

    @property
    def busy(self) -> bool:
        with self._cond:
            return self._running

    def enqueue(self, text: str) -> None:
        with self._cond:
            self._queue.append(text)
            self._running = True
            self._cond.notify()

    def stop_all(self) -> None:
        """Discard the queue, invalidate the pump, fire on_idle if a drain was
        active — an aborted drain still 'ends' so the loop can re-arm the mic."""
        with self._cond:
            was_active = self._running
            self._token += 1
            self._queue = []
            self._running = False
            self._cond.notify()
        if was_active and self.on_idle:
            self.on_idle()

    def _stale(self, token: int) -> bool:
        with self._cond:
            return token != self._token

    def _pump(self) -> None:
        while True:
            with self._cond:
                while not self._queue:
                    self._cond.wait()
                token = self._token
                text = self._queue.pop(0)

            aborted = False
            try:
                for chunk in self._fetch_pcm(text):
                    if self._stale(token):
                        aborted = True
                        break
                    self._write_pcm(chunk)
                    if self._stale(token):
                        aborted = True
                        break
            except Exception as err:   # noqa: BLE001 — skip group, keep draining
                if self._stale(token):
                    aborted = True
                elif self.on_error:
                    self.on_error(err, text)

            if aborted:
                # stop_all() invalidated this run — it owns on_idle / _running.
                continue
            fire = self._after_group(token)
            if fire:
                fire()

    def _after_group(self, token: int) -> Callable[[], None] | None:
        """Ran after each group finishes (cleanly or after a skipped error).
        Returns the on_idle callback to invoke now, or None. Clears _running
        and captures on_idle atomically so a racing stop_all can never drop the
        one on_idle a drain owes. Inter-clip pause runs only between groups
        (never before the first, never after the last)."""
        with self._cond:
            if token != self._token:
                return None            # aborted; stop_all owns on_idle
            if self._queue:
                do_pause = True
                fire: Callable[[], None] | None = None
            else:
                self._running = False
                do_pause = False
                fire = self.on_idle
        if do_pause:
            pause = self._pause_ms()
            if pause > 0:
                self._sleep(pause / 1000.0)
        return fire
