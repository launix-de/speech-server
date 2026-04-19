"""Black-box API tests for the webclient lifecycle."""
from __future__ import annotations

import json
import re
from urllib.parse import urlparse

from conftest import create_call


def _create_webclient_slot(client, headers, call_id, *, user="u1", callback=None):
    params = {
        "base_url": "https://speech.example.com/tts",
        "call_id": call_id,
    }
    if callback is not None:
        params["callback"] = callback
    dsl = f"webclient:{user}" + json.dumps(params, separators=(",", ":"))
    resp = client.post(
        "/api/pipelines",
        data=json.dumps({"dsl": dsl}),
        headers=headers,
    )
    assert resp.status_code == 201, resp.get_data(as_text=True)
    return resp.get_json()


def _browser_bindings(client, iframe_url: str) -> tuple[str, str]:
    """Return (phone_path, scoped_session_id) from the rendered phone UI."""
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
    session_id = json.loads(m.group(1))
    return phone_path, session_id


class TestApiWebclientLifecycle:
    def test_slot_returns_local_ids_and_clean_phone_url(self, client, account):
        call_id = create_call(client, account)
        slot = _create_webclient_slot(client, account, call_id)

        assert slot["session_id"].startswith("wc-")
        assert ":" not in slot["session_id"]
        assert slot["nonce"].startswith("n-")
        assert ":" not in slot["nonce"]
        assert slot["iframe_url"].startswith("https://speech.example.com/tts/phone/")
        assert "dsl=" not in slot["iframe_url"]
        assert "session=" not in slot["iframe_url"]
        assert "base=" not in slot["iframe_url"]

    def test_phone_nonce_answer_and_complete_lifecycle(self, client, account):
        call_id = create_call(client, account)
        slot = _create_webclient_slot(client, account, call_id)
        phone_path, session_id = _browser_bindings(client, slot["iframe_url"])

        resp = client.get(phone_path)
        assert resp.status_code == 200
        resp = client.get(phone_path)
        assert resp.status_code == 200

        resp = client.post(
            phone_path + "/event",
            data=json.dumps({"session": session_id, "event": "answered"}),
            headers=account,
        )
        assert resp.status_code == 200

        resp = client.post(
            phone_path + "/event",
            data=json.dumps({"session": session_id, "event": "completed"}),
            headers=account,
        )
        assert resp.status_code == 200

        resp = client.get(phone_path)
        assert resp.status_code == 403

    def test_completed_burns_nonce_and_rejects_further_events(self, client, account):
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
            data=json.dumps({"session": session_id, "event": "completed"}),
            headers=account,
        )
        assert resp.status_code == 403

        resp = client.post(
            phone_path + "/event",
            data=json.dumps({"session": session_id, "event": "answered"}),
            headers=account,
        )
        assert resp.status_code == 403
