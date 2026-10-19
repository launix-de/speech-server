"""Deliver upstream NDJSON lines asynchronously to an HTTP endpoint.

A single worker preserves delivery order without blocking the audio path.
When the pending queue is full, new lines are dropped with a warning.
Orderly completion drains pending deliveries; cancellation discards them.
"""
from __future__ import annotations

import json
import logging
import queue
import threading
from typing import Iterator, Optional

import requests as http_requests

from .base import AudioFormat, Stage

_LOGGER = logging.getLogger("webhook-sink")


class WebhookSink(Stage):
    """POST NDJSON lines in order, with bounded pending delivery storage.

    ``queue_maxsize=0`` explicitly enables an unbounded queue. Delivery
    is best effort: HTTP errors are logged without retrying. An in-flight
    request can run until ``timeout`` after cancellation.
    """

    def __init__(self, url: str, bearer_token: str = "",
                 extra_fields: Optional[dict] = None,
                 timeout: float = 10.0,
                 queue_maxsize: int = 256) -> None:
        super().__init__()
        self.url = url
        self.bearer_token = bearer_token
        self.extra_fields = extra_fields or {}
        self.timeout = timeout
        self.input_format = AudioFormat(0, "ndjson")
        self._queue: queue.Queue = queue.Queue(maxsize=queue_maxsize)
        self._finished = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._worker_lock = threading.Lock()

    def _ensure_worker(self) -> None:
        with self._worker_lock:
            if self._worker is not None:
                return
            self._worker = threading.Thread(
                target=self._sender_loop, daemon=True,
                name=f"webhook-sink-{self.id}",
            )
            self._worker.start()

    def _sender_loop(self) -> None:
        headers = {"Content-Type": "application/json"}
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        while not self.cancelled:
            try:
                payload = self._queue.get(timeout=0.1)
            except queue.Empty:
                if self._finished.is_set():
                    return
                continue
            if self.cancelled:
                return
            try:
                response = http_requests.post(
                    self.url, data=payload, headers=headers,
                    timeout=self.timeout,
                )
                try:
                    response.raise_for_status()
                finally:
                    response.close()
            except Exception as e:
                _LOGGER.warning("WebhookSink POST failed: %s", e)

    def _enqueue(self, line: bytes) -> None:
        if self.cancelled:
            return
        text = line.decode("utf-8", errors="replace").strip()
        if not text:
            return
        if self.extra_fields:
            try:
                obj = json.loads(text)
                obj.update(self.extra_fields)
                text = json.dumps(obj)
            except (ValueError, TypeError, AttributeError) as e:
                _LOGGER.warning("WebhookSink invalid JSON object: %s", e)
                return
        try:
            self._queue.put_nowait(text)
        except queue.Full:
            _LOGGER.warning("WebhookSink queue full; dropping NDJSON line")

    def stream_pcm24k(self) -> Iterator[bytes]:
        """Consume upstream without waiting for HTTP delivery; yields no audio."""
        if not self.upstream or self.cancelled or self._finished.is_set():
            return
        self._ensure_worker()
        pending = b""
        try:
            for chunk in self.upstream.stream_pcm24k():
                if self.cancelled:
                    break
                if not chunk:
                    continue
                pending += chunk.encode("utf-8") if isinstance(chunk, str) else chunk
                while b"\n" in pending and not self.cancelled:
                    line, pending = pending.split(b"\n", 1)
                    self._enqueue(line)
            if not self.cancelled:
                self._enqueue(pending)
        finally:
            self._finished.set()
        if False:
            yield b""

    def run(self) -> None:
        """Compatibility entry point for existing terminal/sidechain runners."""
        for _ in self.stream_pcm24k():
            if self.cancelled:
                break

    def _on_close(self) -> None:
        self._finished.set()

    def cancel(self) -> None:
        super().cancel()
        self._finished.set()
