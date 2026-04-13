"""Regression: a webhook URL on the subscriber's own origin must NOT be
blocked by url_safety, even if that origin resolves to a private IP.

Prod scenario (2026-04-13): subscriber.base_url = ``https://srv.launix.de/tts``,
the speech-server fires transient webhooks to that origin without issue
(no url_safety check in ``post_webhook``).  But ``WebhookSink`` built via
``webhook:<url>`` DSL applies ``require_safe_url`` — and rejects the very
same host, so the STT transcript never reaches the CRM.

Contract: if the webhook URL has the same scheme+host+port as the
subscriber's ``base_url``, trust it.  Any other URL still passes through
the private-range filter.
"""
from __future__ import annotations

import json

import pytest

from conftest import ADMIN_TOKEN, ACCOUNT_TOKEN, SUBSCRIBER_ID


class TestWebhookOnSubscriberOrigin:

    def test_loopback_webhook_allowed_when_matches_subscriber(
            self, client, admin, monkeypatch):
        """Subscriber has base_url http://127.0.0.1:8090/crm (loopback).
        A webhook on the same origin must build — url_safety alone would
        reject it, but the subscriber whitelist must override."""
        headers = {"Authorization": f"Bearer {ACCOUNT_TOKEN}",
                   "Content-Type": "application/json"}
        client.put("/api/pbx/WhPBX",
                   data=json.dumps({"sip_proxy": "", "sip_user": "",
                                    "sip_password": ""}), headers=admin)
        client.put("/api/accounts/WhAcc",
                   data=json.dumps({"token": ACCOUNT_TOKEN, "pbx": "WhPBX"}),
                   headers=admin)
        client.put(f"/api/subscribe/{SUBSCRIBER_ID}",
                   data=json.dumps({
                       "base_url": "http://127.0.0.1:8090/crm",
                       "bearer_token": "sub-token-xyz",
                   }), headers=headers)

        # Build a webhook stage directly via _create_stage on an executor
        # that knows the subscriber.  Without the whitelist, this raises
        # ValueError from require_safe_url on the loopback URL.
        from speech_pipeline.telephony.pipe_executor import CallPipeExecutor
        from speech_pipeline.telephony import call_state, subscriber as sub_mod
        c = call_state.create_call(SUBSCRIBER_ID, "WhAcc", "WhPBX")
        ex = CallPipeExecutor(call=c, subscriber=sub_mod.get(SUBSCRIBER_ID))
        url = "http://127.0.0.1:8090/crm/sttNote?call=1&participant=2"
        stage = ex._create_stage("webhook", url, {})
        assert stage is not None

    def test_cross_origin_webhook_still_rejected(
            self, client, admin, monkeypatch):
        """Whitelist must ONLY cover the subscriber's own origin — any
        other loopback/private URL must still be rejected."""
        headers = {"Authorization": f"Bearer {ACCOUNT_TOKEN}",
                   "Content-Type": "application/json"}
        client.put("/api/pbx/WhPBX2",
                   data=json.dumps({"sip_proxy": "", "sip_user": "",
                                    "sip_password": ""}), headers=admin)
        client.put("/api/accounts/WhAcc2",
                   data=json.dumps({"token": ACCOUNT_TOKEN, "pbx": "WhPBX2"}),
                   headers=admin)
        client.put(f"/api/subscribe/{SUBSCRIBER_ID}",
                   data=json.dumps({
                       "base_url": "http://127.0.0.1:8090/crm",
                       "bearer_token": "t",
                   }), headers=headers)

        from speech_pipeline.telephony.pipe_executor import CallPipeExecutor
        from speech_pipeline.telephony import call_state, subscriber as sub_mod
        c = call_state.create_call(SUBSCRIBER_ID, "WhAcc2", "WhPBX2")
        ex = CallPipeExecutor(call=c, subscriber=sub_mod.get(SUBSCRIBER_ID))

        # Different host (10.x private) — must still be blocked.
        with pytest.raises(ValueError, match="private or internal network"):
            ex._create_stage("webhook",
                             "http://10.0.0.5/intranet/drop", {})
