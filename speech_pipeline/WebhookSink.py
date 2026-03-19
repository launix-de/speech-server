"""Sink stage: POST NDJSON lines to an HTTP endpoint.

Each line from the upstream (expected: NDJSON text) is sent as the
body of a POST request to the configured URL.  A Bearer token can
be provided for authentication.

Designed for forwarding STT transcription segments to a subscriber
callback endpoint.
"""
from __future__ import annotations

import logging
from typing import Iterator, Optional

import requests as http_requests

import json

from .base import AudioFormat, Stage

_LOGGER = logging.getLogger("webhook-sink")


class WebhookSink(Stage):
    """Sink: reads NDJSON text from upstream, POSTs each line."""

    def __init__(self, url: str, bearer_token: str = "",
                 extra_fields: Optional[dict] = None,
                 timeout: float = 10.0) -> None:
        super().__init__()
        self.url = url
        self.bearer_token = bearer_token
        self.extra_fields = extra_fields or {}
        self.timeout = timeout
        self.input_format = AudioFormat(0, "ndjson")

    def run(self) -> None:
        """Drive the pipeline — read upstream, POST each NDJSON line."""
        if not self.upstream:
            return

        headers = {"Content-Type": "application/json"}
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"

        _LOGGER.info("WebhookSink: streaming to %s", self.url)

        for chunk in self.upstream.stream_pcm24k():
            if self.cancelled:
                break
            if not chunk:
                continue
            text = chunk if isinstance(chunk, str) else chunk.decode("utf-8", errors="replace")
            for line in text.strip().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    # Merge extra fields into the JSON payload
                    if self.extra_fields:
                        obj = json.loads(line)
                        obj.update(self.extra_fields)
                        line = json.dumps(obj)
                    http_requests.post(
                        self.url, data=line, headers=headers,
                        timeout=self.timeout)
                except Exception as e:
                    _LOGGER.warning("WebhookSink POST failed: %s", e)

        _LOGGER.info("WebhookSink: done")
