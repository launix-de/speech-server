"""Delivery must preserve order without blocking the upstream pipeline."""
from __future__ import annotations

import threading
from unittest.mock import Mock

import pytest

from speech_pipeline import WebhookSink
from speech_pipeline.base import AudioFormat, Stage


class _Source(Stage):
    output_format = AudioFormat(0, "ndjson")

    def __init__(self, chunks):
        super().__init__()
        self.chunks = chunks

    def stream_pcm24k(self):
        for chunk in self.chunks:
            if self.cancelled:
                return
            yield chunk


def _finish(sink):
    if sink._worker:
        sink._worker.join(timeout=2)
        assert not sink._worker.is_alive(), "delivery worker leaked"


@pytest.mark.parametrize("entry_point", ["run", "stream_pcm24k"])
def test_slow_delivery_does_not_block_upstream(monkeypatch, entry_point):
    started, release, consumed = (threading.Event() for _ in range(3))
    seen = []

    def post(*args, **kwargs):
        started.set()
        assert release.wait(2)
        seen.append(kwargs["data"])
        return Mock()

    def chunks():
        yield b'{"i":1}\n'
        assert started.wait(2)
        yield b'{"i":2}\n'
        consumed.set()

    sink = _Source(chunks()).pipe(WebhookSink("https://example.invalid/wh"))
    monkeypatch.setattr("requests.post", post)

    def consume():
        result = getattr(sink, entry_point)()
        if result is not None:
            assert list(result) == []

    driver = threading.Thread(target=consume, daemon=True)
    driver.start()
    try:
        assert consumed.wait(1), "upstream blocked on webhook delivery"
        driver.join(1)
        assert not driver.is_alive()
    finally:
        release.set()
        driver.join(2)
        _finish(sink)
    assert seen == ['{"i":1}', '{"i":2}']


def test_fragmented_unicode_lines_and_extra_fields(monkeypatch):
    response = Mock()
    post = Mock(return_value=response)
    monkeypatch.setattr("requests.post", post)
    data = '{"text":"Grüße"}\n\n{"text":"end"}'.encode()
    split = data.index(b'\xc3') + 1
    sink = _Source([data[:split], data[split:]]).pipe(WebhookSink(
        "https://example.invalid/wh", bearer_token="token",
        extra_fields={"call_id": "call-1"}, timeout=3,
    ))
    sink.run()
    _finish(sink)
    import json
    assert [json.loads(c.kwargs["data"]) for c in post.call_args_list] == [
        {"text": "Grüße", "call_id": "call-1"},
        {"text": "end", "call_id": "call-1"},
    ]
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer token"
    assert post.call_args.kwargs["timeout"] == 3
    assert response.close.call_count == 2


def test_delivery_continues_after_network_and_http_errors(monkeypatch, caplog):
    bad_response, good_response = Mock(), Mock()
    bad_response.raise_for_status.side_effect = RuntimeError("HTTP 500")
    post = Mock(side_effect=[RuntimeError("network down"), bad_response, good_response])
    monkeypatch.setattr("requests.post", post)
    sink = _Source([b'1\n2\n3\n']).pipe(WebhookSink("https://example.invalid/wh"))
    sink.run()
    _finish(sink)
    assert [c.kwargs["data"] for c in post.call_args_list] == ["1", "2", "3"]
    assert "network down" in caplog.text and "HTTP 500" in caplog.text
    bad_response.close.assert_called_once()
    good_response.close.assert_called_once()


@pytest.mark.parametrize("cancel", [False, True])
def test_full_queue_never_blocks_and_cancel_discards_pending(monkeypatch, caplog, cancel):
    started, release = threading.Event(), threading.Event()
    seen = []

    def post(*args, **kwargs):
        started.set()
        assert release.wait(2)
        seen.append(kwargs["data"])
        return Mock()

    def chunks():
        yield b'1\n'
        assert started.wait(2)
        yield b'2\n3\n'

    monkeypatch.setattr("requests.post", post)
    sink = _Source(chunks()).pipe(WebhookSink("https://example.invalid/wh", queue_maxsize=1))
    driver = threading.Thread(target=sink.run, daemon=True)
    driver.start()
    try:
        driver.join(1)
        assert not driver.is_alive(), "full queue blocked pipeline/shutdown"
        assert "queue full" in caplog.text
        if cancel:
            sink.cancel()
            assert sink.upstream.cancelled
    finally:
        release.set()
        driver.join(2)
        _finish(sink)
    assert seen == (["1"] if cancel else ["1", "2"])


def test_upstream_error_finishes_worker(monkeypatch):
    post = Mock(return_value=Mock())
    monkeypatch.setattr("requests.post", post)

    def chunks():
        yield b'1\n'
        raise RuntimeError("upstream failed")

    sink = _Source(chunks()).pipe(WebhookSink("https://example.invalid/wh"))
    with pytest.raises(RuntimeError, match="upstream failed"):
        sink.run()
    _finish(sink)
    assert post.call_count == 1


def test_cancel_before_start_does_not_consume_or_start_worker(monkeypatch):
    post = Mock()
    monkeypatch.setattr("requests.post", post)
    sink = _Source([b'1\n']).pipe(WebhookSink("https://example.invalid/wh"))
    sink.cancel()
    assert list(sink.stream_pcm24k()) == []
    assert sink._worker is None
    post.assert_not_called()
