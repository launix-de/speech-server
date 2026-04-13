"""WebClient session cleanup on connection loss + TTL reaper."""
from __future__ import annotations

import json
import time
from unittest.mock import MagicMock

import pytest

from speech_pipeline.telephony import call_state, webclient


@pytest.fixture
def call():
    c = call_state.create_call("sub", "acc", "pbx")
    yield c
    call_state.delete_call(c.call_id)


@pytest.fixture
def fake_nonce():
    return "n-" + "x" * 24


class TestExplicitCleanup:

    def test_close_removes_session_from_registry(self, call, fake_nonce):
        sess = webclient.register_webclient(call, "u1", fake_nonce)
        sid = sess["session_id"]
        assert webclient.get_webclient_session(sid) is not None

        webclient.close_webclient_session(sid)
        assert webclient.get_webclient_session(sid) is None

    def test_close_revokes_nonce(self, call, monkeypatch):
        from speech_pipeline.telephony import auth as auth_mod
        entry = auth_mod.create_nonce(
            account_id=call.account_id,
            subscriber_id=call.subscriber_id,
            user="u1",
        )
        nonce = entry["nonce"]
        sess = webclient.register_webclient(call, "u1", nonce)
        webclient.close_webclient_session(sess["session_id"])
        assert auth_mod._nonces.get(nonce) is None

    def test_close_unregisters_participant(self, call, fake_nonce):
        sess = webclient.register_webclient(call, "u1", fake_nonce)
        sid = sess["session_id"]
        call.register_participant(sid, type="webclient", user="u1", nonce=fake_nonce)
        assert call.get_participant(sid) is not None

        webclient.close_webclient_session(sid)
        assert call.get_participant(sid) is None

    def test_close_is_idempotent(self, call, fake_nonce):
        sess = webclient.register_webclient(call, "u1", fake_nonce)
        sid = sess["session_id"]
        webclient.close_webclient_session(sid)
        # Second close must not raise and must remain a no-op.
        webclient.close_webclient_session(sid)


class TestConnectionLossCleanup:
    """Simulates the ``/ws/socket/<session_id>`` route calling
    ``close_webclient_session`` after ``handle_ws`` returns — which
    happens when the browser disconnects, crashes, or the network drops.
    """

    def test_ws_disconnect_closes_session(self, call, fake_nonce):
        sess = webclient.register_webclient(call, "u1", fake_nonce)
        sid = sess["session_id"]
        # Simulate `finally: close_webclient_session(sid)` in the route.
        webclient.close_webclient_session(sid)
        assert webclient.get_webclient_session(sid) is None

    def test_call_teardown_closes_all_sessions_on_call(self, call, fake_nonce):
        s1 = webclient.register_webclient(call, "u1", fake_nonce + "a")
        s2 = webclient.register_webclient(call, "u2", fake_nonce + "b")
        webclient.close_call_sessions(call.call_id)
        assert webclient.get_webclient_session(s1["session_id"]) is None
        assert webclient.get_webclient_session(s2["session_id"]) is None


class TestReaper:
    """The reaper drops sessions whose codec socket never connected
    within ``SESSION_CONNECT_TIMEOUT_SECONDS``."""

    def test_reap_stale_never_connected(self, call, fake_nonce, monkeypatch):
        monkeypatch.setattr(webclient, "SESSION_CONNECT_TIMEOUT_SECONDS", 0.1)
        sess = webclient.register_webclient(call, "u1", fake_nonce)
        sid = sess["session_id"]
        # Sleep past timeout, then trigger the reap.
        time.sleep(0.2)
        webclient._reap_stale_sessions()
        assert webclient.get_webclient_session(sid) is None

    def test_connected_session_is_not_reaped(self, call, fake_nonce, monkeypatch):
        monkeypatch.setattr(webclient, "SESSION_CONNECT_TIMEOUT_SECONDS", 0.1)
        sess = webclient.register_webclient(call, "u1", fake_nonce)
        sid = sess["session_id"]

        # Inject a fake CodecSocketSession that reports "connected".
        # Package __init__ rebinds the submodule name to the class, so
        # reach the module via sys.modules.
        import sys
        codec_mod = sys.modules["speech_pipeline.CodecSocketSession"]
        fake_codec = MagicMock()
        fake_codec.connected.is_set.return_value = True
        with codec_mod._sessions_lock:
            codec_mod._sessions[sid] = fake_codec
        try:
            time.sleep(0.2)
            webclient._reap_stale_sessions()
            assert webclient.get_webclient_session(sid) is not None
        finally:
            with codec_mod._sessions_lock:
                codec_mod._sessions.pop(sid, None)
            webclient.close_webclient_session(sid)

    def test_recently_created_not_reaped(self, call, fake_nonce, monkeypatch):
        """Brand-new sessions (under the TTL) survive the reap pass."""
        monkeypatch.setattr(webclient, "SESSION_CONNECT_TIMEOUT_SECONDS", 60.0)
        sess = webclient.register_webclient(call, "u1", fake_nonce)
        sid = sess["session_id"]
        webclient._reap_stale_sessions()
        assert webclient.get_webclient_session(sid) is not None
        webclient.close_webclient_session(sid)


class TestPhoneUiNonceIsNotConsumedOnLoad:
    """Regression: ``GET /phone/<nonce>`` used to call
    ``validate_nonce`` (consuming).  A second load (browser reload,
    iframe re-navigation, /event POST) then returned 403 → the
    webclient showed 'Disconnected' immediately after the user
    pressed the green handset."""

    def test_phone_ui_can_be_loaded_twice(self, client, account):
        from speech_pipeline.telephony import auth as auth_mod, subscriber as sub_mod
        from conftest import SUBSCRIBER_ID, create_call
        # We need a webclient feature-enabled account; look it up from conftest.
        if not auth_mod.check_feature(
            next(iter(auth_mod._accounts.keys())), "webclient"
        ):
            pytest.skip("webclient feature not enabled on test account")

        call_id = create_call(client, account)
        call = call_state.get_call(call_id)
        try:
            nonce_entry = auth_mod.create_nonce(
                account_id=call.account_id,
                subscriber_id=call.subscriber_id,
                user="u1",
            )
            nonce = nonce_entry["nonce"]
            webclient.register_webclient(call, "u1", nonce)

            resp1 = client.get(f"/phone/{nonce}")
            assert resp1.status_code == 200
            resp2 = client.get(f"/phone/{nonce}")
            assert resp2.status_code == 200, (
                "Second phone_ui load returned "
                f"{resp2.status_code} — nonce got consumed on first load"
            )
        finally:
            client.delete(f"/api/calls/{call_id}", headers=account)

    def test_check_vs_consume_nonce(self):
        from speech_pipeline.telephony import auth as auth_mod
        entry = auth_mod.create_nonce("acc", "sub", "u1", ttl=60)
        nonce = entry["nonce"]
        # check_nonce is idempotent.
        assert auth_mod.check_nonce(nonce) is not None
        assert auth_mod.check_nonce(nonce) is not None
        # consume_nonce burns it exactly once.
        assert auth_mod.consume_nonce(nonce) is not None
        assert auth_mod.consume_nonce(nonce) is None
        assert auth_mod.check_nonce(nonce) is None


class TestPhoneEventCompletedTearsDown:
    """Browser hits red handset → ``/phone/<nonce>/event`` with
    event=completed.  Server must close the webclient session
    immediately so the conference leg detaches and the peer (e.g.
    phone) doesn't keep running until the 30 s idle timeout."""

    def test_phone_event_completed_closes_session(
            self, client, account):
        from speech_pipeline.telephony import auth as auth_mod
        from conftest import create_call
        call_id = create_call(client, account)
        call = call_state.get_call(call_id)
        try:
            nonce_entry = auth_mod.create_nonce(
                account_id=call.account_id,
                subscriber_id=call.subscriber_id,
                user="u1",
            )
            nonce = nonce_entry["nonce"]
            sess = webclient.register_webclient(call, "u1", nonce)
            sid = sess["session_id"]
            assert webclient.get_webclient_session(sid) is not None

            resp = client.post(
                f"/phone/{nonce}/event",
                data=json.dumps({"session": sid, "event": "completed"}),
                headers={"Content-Type": "application/json"},
            )
            assert resp.status_code == 200
            assert webclient.get_webclient_session(sid) is None, (
                "Session still present after phone_event(completed) — "
                "conference leg stays attached and the peer hangs for "
                "~30 s until idle cleanup."
            )
        finally:
            client.delete(f"/api/calls/{call_id}", headers=account)


class TestBrowserReconnect:
    """Browsers commonly do a short disconnect-reconnect on page load
    (HTTP→WS upgrade, CORS pre-checks).  The session MUST survive —
    if we tear it down on the first WS close the browser can never
    reconnect (observed 2026-04-13, Webclient shows 'disconnected')."""

    def test_session_survives_ws_disconnect(self, call, fake_nonce):
        sess = webclient.register_webclient(call, "u1", fake_nonce)
        sid = sess["session_id"]
        assert webclient.get_webclient_session(sid) is not None

        # Simulate the browser's first WS round-trip ending — the
        # session must NOT be auto-closed on its own; only explicit
        # DELETE /api/calls or the reaper may remove it.
        # (The server's /ws/socket route used to call
        # close_webclient_session in a finally block — that broke
        # reconnects.)
        assert webclient.get_webclient_session(sid) is not None, (
            "Session was removed on a passive lookup — the "
            "disconnect-cleanup hook is too aggressive"
        )


class TestCleanupIntegrationWithCall:
    """When the call ends, every session must be closed and the call's
    idle-cleanup must be able to fire without leftover state."""

    def test_delete_call_closes_webclient_sessions(self, client, account):
        """End-to-end: DELETE /api/calls/<id> removes webclient sessions
        tied to that call."""
        from conftest import create_call
        call_id = create_call(client, account)
        call = call_state.get_call(call_id)
        sess = webclient.register_webclient(call, "u1", "n-" + "y" * 24)
        sid = sess["session_id"]

        client.delete(f"/api/calls/{call_id}", headers=account)
        assert webclient.get_webclient_session(sid) is None
