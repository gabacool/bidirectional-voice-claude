"""Pure tests for ``run_generator_on`` — the inference-thread -> handler-thread bridge.

mlx (>=0.31) GPU streams are thread-local, so the voice API runs every model call
on one dedicated inference thread. For streaming speech the generator therefore
runs on that thread (the producer) while the HTTP handler thread (the consumer)
owns the socket; ``run_generator_on`` pumps chunks across a small bounded queue.

These tests exercise the real bridge with a *fake* generator — no MLX model, no
socket — so they run in the default suite. They pin down the four properties the
streaming path depends on: order preservation, producer-exception propagation,
early-consumer-close aborting the producer, and lossless delivery to a slow
consumer (backpressure), plus clean end-of-stream handling.
"""

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import voice_api


class FakeGenerator:
    """A model-free chunk source that records how far it actually ran.

    ``produced`` counts chunks yielded; ``closed`` flips True when the generator's
    ``finally`` runs (i.e. it was exhausted OR ``close()``d via ``GeneratorExit``).
    Together they let a test prove the producer stopped early instead of running
    the whole sequence to completion.
    """

    def __init__(self, n: int, per_item_sleep: float = 0.0,
                 raise_after: int | None = None) -> None:
        self.n = n
        self.per_item_sleep = per_item_sleep
        self.raise_after = raise_after
        self.produced = 0
        self.closed = False

    def __iter__(self):
        try:
            for i in range(self.n):
                if self.raise_after is not None and i == self.raise_after:
                    raise ValueError(f"boom at chunk {i}")
                if self.per_item_sleep:
                    time.sleep(self.per_item_sleep)
                self.produced += 1
                yield np.full(4, float(i), dtype=np.float32)
        finally:
            self.closed = True


def test_chunk_order_preserved() -> None:
    """Every chunk arrives, in order, exactly once (fast consumer)."""
    fake = FakeGenerator(n=10)
    with ThreadPoolExecutor(max_workers=1) as ex:
        got = list(voice_api.run_generator_on(ex, lambda: iter(fake)))
    assert len(got) == 10
    for i, chunk in enumerate(got):
        assert np.array_equal(chunk, np.full(4, float(i), dtype=np.float32))
    # Clean end-of-stream: the whole sequence ran and the generator closed.
    assert fake.produced == 10
    assert fake.closed is True


def test_slow_consumer_loses_no_chunks() -> None:
    """A consumer slower than the producer still receives every chunk in order.

    The bounded queue applies backpressure (the producer blocks on a full queue)
    rather than dropping or overwriting chunks.
    """
    fake = FakeGenerator(n=20)
    got = []
    with ThreadPoolExecutor(max_workers=1) as ex:
        for chunk in voice_api.run_generator_on(ex, lambda: iter(fake), maxsize=2):
            time.sleep(0.005)  # consumer lags well behind a would-be fast producer
            got.append(chunk)
    assert [int(c[0]) for c in got] == list(range(20))
    assert fake.produced == 20
    assert fake.closed is True


def test_producer_exception_surfaces_to_consumer() -> None:
    """An exception raised inside the generator is re-raised in the consumer."""
    fake = FakeGenerator(n=10, raise_after=3)
    got = []
    with ThreadPoolExecutor(max_workers=1) as ex:
        with pytest.raises(ValueError, match="boom at chunk 3"):
            for chunk in voice_api.run_generator_on(ex, lambda: iter(fake)):
                got.append(chunk)
    # The chunks before the failure were delivered intact and in order.
    assert [int(c[0]) for c in got] == [0, 1, 2]


def test_early_consumer_close_stops_producer() -> None:
    """Closing the consumer early aborts the producer (no orphan generation).

    The fake would yield 10_000 chunks (with a per-chunk sleep, so running it to
    completion would take seconds); we read a few then close. By the time close()
    returns the producer has been joined, so ``produced`` is a stable, tiny count
    — proof the producer stopped between chunks rather than finishing.
    """
    fake = FakeGenerator(n=10_000, per_item_sleep=0.001)
    with ThreadPoolExecutor(max_workers=1) as ex:
        stream = voice_api.run_generator_on(ex, lambda: iter(fake), maxsize=4)
        seen = 0
        for _chunk in stream:
            seen += 1
            if seen == 3:
                break
        stream.close()  # GeneratorExit -> stop flag -> producer aborts + joins
    assert seen == 3
    assert fake.closed is True, "underlying generator should have been closed"
    assert fake.produced < 100, (
        f"producer kept running after consumer closed: produced={fake.produced} "
        "of 10000 — it should have stopped between chunks"
    )


def test_gen_factory_exception_surfaces() -> None:
    """A failure while *creating* the generator also reaches the consumer."""
    def boom() -> "iter":
        raise RuntimeError("factory failed")

    with ThreadPoolExecutor(max_workers=1) as ex:
        with pytest.raises(RuntimeError, match="factory failed"):
            list(voice_api.run_generator_on(ex, boom))
