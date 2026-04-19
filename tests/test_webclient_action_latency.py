"""Regression: ``webclient:USER`` must stay slot-only and fast.

The action may create a nonce + session id, but it must not build any
``codec -> call`` or ``tee -> stt -> webhook`` pipes itself. The CRM
attaches those explicitly after the answered callback.
"""
from __future__ import annotations

import json
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


class TestWebclientActionStaysSlotOnly:

    def test_post_does_not_instantiate_media_stages(
            self, client, webclient_account, monkeypatch):
        """The slot action must not create codec/tee/stt stages at all."""
        from speech_pipeline.telephony import call_state
        c = call_state.create_call(SUBSCRIBER_ID, ACCOUNT_ID, "TestPBX")

        from speech_pipeline.telephony.pipe_executor import CallPipeExecutor
        original_create = CallPipeExecutor._create_stage
        seen = []

        def _slow_create(self, typ, elem_id, params, **kw):
            seen.append(typ)
            return original_create(self, typ, elem_id, params, **kw)
        monkeypatch.setattr(CallPipeExecutor, "_create_stage", _slow_create)

        dsl_params = {
            "callback": "/cb/wc",
            "base_url": "https://srv.example.com",
            "call_id": c.call_id,
        }
        dsl = 'webclient:u1' + json.dumps(dsl_params)

        t0 = time.monotonic()
        resp = client.post("/api/pipelines",
                           data=json.dumps({"dsl": dsl}),
                           headers=webclient_account)
        elapsed = time.monotonic() - t0

        assert resp.status_code in (200, 201), resp.get_data(as_text=True)
        assert elapsed < 1.0, (
            f"POST /api/pipelines {{webclient:}} took {elapsed:.2f}s — "
            "slot creation must stay lightweight"
        )
        assert seen == [], (
            "webclient slot creation instantiated media stages itself; "
            f"expected slot-only behavior, got {seen!r}"
        )

    def test_no_wc_build_thread_is_spawned(
            self, client, webclient_account):
        """Slot-only webclient creation must not spawn build threads."""
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

        time.sleep(0.1)
        assert not any(t.name.startswith("wc-build-")
                       for t in __import__("threading").enumerate()), (
            "webclient slot creation must not spawn wc-build-* threads"
        )
