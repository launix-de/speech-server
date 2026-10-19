"""Audit guard: every outbound webhook must be fire-and-forget.

Why this is a hard rule
-----------------------
The CRM is typically a PHP-FPM tenant with a per-session lock — a slow
or hung subscriber endpoint can pin a worker for the timeout window.
If the speech server blocks any of its own threads on such a webhook,
the CRM's follow-up API calls (the bridge POSTs that connect inbound
calls) queue behind the wait and the user perceives "phone rings minutes
late".

Every site that POSTs to a subscriber-controlled URL must therefore
return control to its caller immediately and do the actual HTTP work in
its own thread.  This test pins that contract.
"""
from __future__ import annotations

import threading
import time
from unittest.mock import patch

import pytest

from speech_pipeline.telephony import _shared, dispatcher, leg as leg_mod


@pytest.fixture
def workers(monkeypatch):
    original = threading.Thread
    threads = []

    def create(*args, **kwargs):
        thread = original(*args, **kwargs)
        threads.append(thread)
        return thread

    monkeypatch.setattr(threading, "Thread", create)
    return threads


def _join(workers):
    for worker in workers:
        worker.join(2)
        assert not worker.is_alive(), "webhook worker leaked"


class _FakeSubscriber(dict):
    """Just enough of the subscriber dict for fire_subscriber_event."""


def _make_subscriber():
    return _FakeSubscriber({
        "id": "sub-1",
        "account_id": "acct-1",
        "base_url": "https://example.invalid/crm",
        "bearer_token": "tok",
        "events": {"incoming": "/state=incoming"},
    })


def test_post_webhook_returns_immediately_on_slow_endpoint(workers):
    """``_shared.post_webhook`` must not block its caller."""
    blocker = threading.Event()
    started = threading.Event()

    def _slow(*args, **kwargs):
        started.set()
        blocker.wait(timeout=10.0)

        class _R:
            status_code = 200

        return _R()

    with patch("speech_pipeline.telephony._shared.http_requests.post",
               side_effect=_slow):
        t0 = time.monotonic()
        _shared.post_webhook("https://example.invalid/wh", {"x": 1}, "tok")
        elapsed = time.monotonic() - t0

        try:
            assert elapsed < 1, f"post_webhook blocked for {elapsed:.3f}s"
            assert started.wait(timeout=2), "background thread never ran"
        finally:
            blocker.set()
            _join(workers)


def test_dispatcher_fire_subscriber_event_returns_immediately(workers):
    """``dispatcher.fire_subscriber_event`` must not block its caller."""
    blocker = threading.Event()
    started = threading.Event()

    def _slow(*args, **kwargs):
        started.set()
        blocker.wait(timeout=10.0)

        class _R:
            status_code = 200

        return _R()

    with patch("speech_pipeline.telephony.dispatcher.http_requests.request",
               side_effect=_slow):
        t0 = time.monotonic()
        dispatcher.fire_subscriber_event(
            _make_subscriber(), "incoming", {"caller": "+49"},
        )
        elapsed = time.monotonic() - t0

        try:
            assert elapsed < 1, f"fire_subscriber_event blocked for {elapsed:.3f}s"
            assert started.wait(timeout=2)
        finally:
            blocker.set()
            _join(workers)


def test_leg_fire_callback_returns_immediately(workers):
    """``leg.fire_callback`` (used for ringing/answered/completed) is async."""
    from speech_pipeline.telephony import subscriber as sub_mod

    sub_mod._subscribers["sub-cb"] = {
        "id": "sub-cb",
        "account_id": "acct-cb",
        "base_url": "https://example.invalid/crm",
        "bearer_token": "tok",
        "events": {},
    }
    try:
        leg = leg_mod.create_leg(
            "outbound", "+49", "TestPBX", "sub-cb", voip_call=None,
        )
        leg.callbacks = {"ringing": "/ring"}

        blocker = threading.Event()
        started = threading.Event()

        def _slow(*args, **kwargs):
            started.set()
            blocker.wait(timeout=10.0)

            class _R:
                status_code = 200

            return _R()

        with patch("speech_pipeline.telephony._shared.http_requests.post",
                   side_effect=_slow):
            t0 = time.monotonic()
            leg_mod.fire_callback(leg, "ringing")
            elapsed = time.monotonic() - t0

            try:
                assert elapsed < 1, f"fire_callback blocked for {elapsed:.3f}s"
                assert started.wait(timeout=2)
            finally:
                blocker.set()
                _join(workers)
    finally:
        leg_mod.delete_leg(leg.leg_id)
        sub_mod._subscribers.pop("sub-cb", None)

