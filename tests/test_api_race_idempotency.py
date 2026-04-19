"""API-level idempotency and event ordering tests."""
from __future__ import annotations

import json
import re
from urllib.parse import urlparse

from conftest import create_call


def _create_webclient_slot(client, headers, call_id):
    dsl = (
        'webclient:u1{"base_url":"https://speech.example.com/tts",'
        f'"call_id":"{call_id}"'
        "}"
    )
    resp = client.post(
        "/api/pipelines",
        data=json.dumps({"dsl": dsl}),
        headers=headers,
    )
    assert resp.status_code == 201, resp.get_data(as_text=True)
    return resp.get_json()


def _browser_bindings(client, iframe_url: str) -> tuple[str, str]:
    parsed = urlparse(iframe_url)
    phone_path = parsed.path
    resp = client.get(phone_path)
    if resp.status_code == 404 and "/phone/" in phone_path:
        phone_path = phone_path[phone_path.index("/phone/"):]
        resp = client.get(phone_path)
    assert resp.status_code == 200, resp.get_data(as_text=True)
    html = resp.get_data(as_text=True)
    m = re.search(r"var sessionId = ([^;]+);", html)
    assert m, html
    return phone_path, json.loads(m.group(1))


class TestApiWebclientEventIdempotency:
    def test_answered_is_idempotent(self, client, account):
        call_id = create_call(client, account)
        slot = _create_webclient_slot(client, account, call_id)
        phone_path, session_id = _browser_bindings(client, slot["iframe_url"])

        for _ in range(2):
            resp = client.post(
                phone_path + "/event",
                data=json.dumps({"session": session_id, "event": "answered"}),
                headers=account,
            )
            assert resp.status_code == 200

    def test_completed_then_answered_is_rejected(self, client, account):
        call_id = create_call(client, account)
        slot = _create_webclient_slot(client, account, call_id)
        phone_path, session_id = _browser_bindings(client, slot["iframe_url"])

        resp = client.post(
            phone_path + "/event",
            data=json.dumps({"session": session_id, "event": "completed"}),
            headers=account,
        )
        assert resp.status_code == 200

        resp = client.post(
            phone_path + "/event",
            data=json.dumps({"session": session_id, "event": "answered"}),
            headers=account,
        )
        assert resp.status_code == 403

    def test_event_with_wrong_session_is_rejected(self, client, account):
        call_id = create_call(client, account)
        slot = _create_webclient_slot(client, account, call_id)
        phone_path, _ = _browser_bindings(client, slot["iframe_url"])

        resp = client.post(
            phone_path + "/event",
            data=json.dumps({"session": "wc-attacker", "event": "answered"}),
            headers=account,
        )
        assert resp.status_code == 404
