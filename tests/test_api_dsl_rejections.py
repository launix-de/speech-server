"""HTTP API guards for invalid or dangerous DSL payloads."""
from __future__ import annotations

import json

from conftest import create_call


class TestApiDslRejections:
    def test_webclient_rejects_embedded_media_params(self, client, account):
        call_id = create_call(client, account)
        resp = client.post(
            "/api/pipelines",
            data=json.dumps({
                "dsl": 'webclient:u1{"base_url":"https://speech.example.com/tts",'
                       f'"call_id":"{call_id}",'
                       '"stt_callback":"https://crm.example.com/stt"}'
            }),
            headers=account,
        )
        assert resp.status_code == 400
        assert "webclient only creates a slot" in resp.get_data(as_text=True)

    def test_tee_sidechain_requires_existing_primary_tee(self, client, account):
        resp = client.post(
            "/api/pipelines",
            data=json.dumps({
                "dsl": "tee:missing_tap -> stt:de -> webhook:https://crm.example.com/stt",
            }),
            headers=account,
        )
        assert resp.status_code == 400
        assert "tee may only start a pipe" in resp.get_data(as_text=True)

    def test_play_requires_json_url(self, client, account):
        call_id = create_call(client, account)
        resp = client.post(
            "/api/pipelines",
            data=json.dumps({"dsl": f"play:hold_music -> call:{call_id}"}),
            headers=account,
        )
        assert resp.status_code == 400
        assert "play requires a url parameter" in resp.get_data(as_text=True)
