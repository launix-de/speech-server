"""Contract: POST /api/pipelines webclient:USER must return session_id + nonce.

The CRM owns participant identity: it creates the Participant row with a
deterministic primary key, bakes ``participant=$id`` into every callback
URL it passes to the server, then calls the webclient DSL action and
stores the returned ``session_id`` as the participant's ``sid``.  Without
that, CRM has to guess which participant a later webhook refers to.
"""
from __future__ import annotations

import json

import pytest

from conftest import ADMIN_TOKEN, ACCOUNT_TOKEN, SUBSCRIBER_ID


@pytest.fixture
def wc_account(client, admin):
    headers = {"Authorization": f"Bearer {ACCOUNT_TOKEN}",
               "Content-Type": "application/json"}
    client.put("/api/pbx/WcPBX",
               data=json.dumps({"sip_proxy": "", "sip_user": "",
                                "sip_password": ""}), headers=admin)
    client.put("/api/accounts/WcAcc",
               data=json.dumps({"token": ACCOUNT_TOKEN, "pbx": "WcPBX",
                                 "features": ["webclient"]}), headers=admin)
    client.put(f"/api/subscribe/{SUBSCRIBER_ID}",
               data=json.dumps({"base_url": "https://crm.example.com/crm",
                                 "bearer_token": "t"}), headers=headers)
    return headers


class TestWebclientActionResponse:

    def test_response_surfaces_session_id_and_nonce(self, client, wc_account):
        from speech_pipeline.telephony import call_state
        c = call_state.create_call(SUBSCRIBER_ID, "WcAcc", "WcPBX")

        dsl_params = {
            "callback": f"/cb?call=1&participant=42",
            "base_url": "https://crm.example.com",
            "call_id": c.call_id.split(":", 1)[1],
        }
        resp = client.post("/api/pipelines",
                           data=json.dumps({"dsl":
                               'webclient:u1' + json.dumps(dsl_params)}),
                           headers=wc_account)
        assert resp.status_code == 201, resp.get_data(as_text=True)
        body = resp.get_json()
        assert body.get("session_id", "").startswith("wc-"), (
            f"POST /api/pipelines webclient:... did not return session_id "
            f"in response body — CRM has no way to store it on the "
            f"Participant row. Body: {body}"
        )
        assert body.get("nonce", "").startswith("n-"), (
            f"response missing nonce (browser iframe auth token): {body}"
        )
        assert "iframe_url" in body, (
            f"response missing iframe_url for the browser popup: {body}"
        )
        assert "?" not in body["iframe_url"], (
            f"iframe_url must not expose browser-controlled query params: "
            f"{body['iframe_url']}"
        )
        assert "/phone/" in body["iframe_url"], body["iframe_url"]

    def test_scheme_less_base_url_is_rejected(self, client, wc_account):
        from speech_pipeline.telephony import call_state
        c = call_state.create_call(SUBSCRIBER_ID, "WcAcc", "WcPBX")

        dsl_params = {
            "callback": "/cb?call=1&participant=42",
            "base_url": "srv.launix.de/tts",
            "call_id": c.call_id.split(":", 1)[1],
        }
        resp = client.post("/api/pipelines",
                           data=json.dumps({"dsl":
                               'webclient:u1' + json.dumps(dsl_params)}),
                           headers=wc_account)
        assert resp.status_code == 400, resp.get_data(as_text=True)
        assert "base_url must be absolute" in resp.get_data(as_text=True)

    def test_embedded_media_params_are_rejected(self, client, wc_account):
        from speech_pipeline.telephony import call_state
        c = call_state.create_call(SUBSCRIBER_ID, "WcAcc", "WcPBX")

        dsl_params = {
            "callback": "/cb?call=1&participant=42",
            "base_url": "https://srv.example.com/tts",
            "call_id": c.call_id.split(":", 1)[1],
            "stt_callback": "https://crm.example.com/sttNote?call=1&participant=42",
        }
        resp = client.post("/api/pipelines",
                           data=json.dumps({"dsl":
                               'webclient:u1' + json.dumps(dsl_params)}),
                           headers=wc_account)
        assert resp.status_code == 400, resp.get_data(as_text=True)
        assert "build codec/webhook pipes separately" in resp.get_data(as_text=True)
