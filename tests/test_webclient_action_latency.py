"""Regression: POST /api/pipelines with webclient:USER must return fast.

Observed in production: CRM calls joinWebclientAction → POSTs
``webclient:USER{..., call_id}`` → server synchronously builds the
STT/codec pipes (Whisper model load, ConferenceLeg wiring) → CRM's
front-proxy times out with 504 before the server responds.

The pipe build must run in a background thread; the HTTP response
only needs the nonce + iframe_url, which are ready immediately.
"""
from __future__ import annotations

import json
import threading
import time

import pytest

from conftest import ADMIN_TOKEN, ACCOUNT_ID, ACCOUNT_TOKEN, SUBSCRIBER_ID


@pytest.fixture
def webclient_account(client, admin):
    """Provision an account with the webclient feature enabled."""
    headers = {"Authorization": f"Bearer {ACCOUNT_TOKEN}",
               "Content-Type": "application/json"}
    client.put("/api/pbx/TestPBX",
               data=json.dumps({"sip_proxy": "", "sip_user": "", "sip_password": ""}),
               headers=admin)
    client.put(f"/api/accounts/{ACCOUNT_ID}",
               data=json.dumps({
                   "token": ACCOUNT_TOKEN,
                   "pbx": "TestPBX",
                   "features": ["webclient"],
               }), headers=admin)
    client.put(f"/api/subscribe/{SUBSCRIBER_ID}",
               data=json.dumps({
                   "base_url": "https://example.com/crm",
                   "bearer_token": "sub-token-xyz",
               }), headers=headers)
    return headers


class TestWebclientActionDoesNotBlockOnPipeBuild:

    def test_post_returns_even_if_pipe_build_blocks(
            self, client, webclient_account, monkeypatch):
        """CRM timeout scenario: if stage construction blocks (e.g.
        Whisper model load), the POST response MUST still return within
        the CRM's proxy budget. We simulate a slow build with a
        blocking event and assert the HTTP roundtrip is fast."""
        # Create a conference
        from speech_pipeline.telephony import call_state
        c = call_state.create_call(SUBSCRIBER_ID, ACCOUNT_ID, "TestPBX")

        # Simulate a slow heavy stage (Whisper model load, ConferenceLeg
        # wiring, etc.) by intercepting _create_stage ONLY for codec/stt
        # elements — the ones reached from the inner webclient pipes.
        # The outer ``webclient:USER`` dispatch itself creates no stage
        # and must stay fast.
        gate = threading.Event()
        from speech_pipeline.telephony.pipe_executor import CallPipeExecutor
        original_create = CallPipeExecutor._create_stage

        def _slow_create(self, typ, elem_id, params, **kw):
            if typ in ("codec", "stt", "tee"):
                gate.wait(timeout=10)
            return original_create(self, typ, elem_id, params, **kw)
        monkeypatch.setattr(CallPipeExecutor, "_create_stage", _slow_create)

        dsl_params = {
            "callback": "/cb/wc",
            "base_url": "https://srv.example.com",
            "call_id": c.call_id,
            "stt_callback": "/stt",
        }
        dsl = 'webclient:u1' + json.dumps(dsl_params)

        t0 = time.monotonic()
        resp = client.post("/api/pipelines",
                           data=json.dumps({"dsl": dsl}),
                           headers=webclient_account)
        elapsed = time.monotonic() - t0

        # Release the blocked background build so teardown is clean.
        gate.set()

        assert resp.status_code in (200, 201), resp.get_data(as_text=True)
        # Generous budget (1s) — real CRM proxy is ~30s, but blocking
        # build would keep the response pinned until gate.wait(10s).
        assert elapsed < 1.0, (
            f"POST /api/pipelines {{webclient:}} took {elapsed:.2f}s — "
            "pipe build must run in background, not inline"
        )

    def test_async_build_thread_is_spawned(
            self, client, webclient_account):
        """Smoke: the webclient action hands the pipe build off to a
        named background thread (``wc-build-<sid>``) instead of blocking
        the request."""
        from speech_pipeline.telephony import call_state
        c = call_state.create_call(SUBSCRIBER_ID, ACCOUNT_ID, "TestPBX")

        dsl_params = {
            "callback": "/cb/wc",
            "base_url": "https://srv.example.com",
            "call_id": c.call_id,
        }
        resp = client.post("/api/pipelines",
                           data=json.dumps({"dsl":
                               'webclient:u2' + json.dumps(dsl_params)}),
                           headers=webclient_account)
        assert resp.status_code in (200, 201)

        # Within a short window, at least one wc-build-* thread must
        # have existed.  Since it may finish quickly we also accept
        # "already-joined" by checking enumeration at any point during
        # the poll.
        seen = False
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            if any(t.name.startswith("wc-build-")
                   for t in threading.enumerate()):
                seen = True
                break
            time.sleep(0.02)
        assert seen, "No wc-build-* background thread was spawned"
