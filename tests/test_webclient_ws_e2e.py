"""End-to-end WebSocket test for the webclient audio path.

The webclient action is slot-only. After the browser joins, the CRM must
explicitly attach ``codec:<session_id>`` via `/api/pipelines`.
"""
from __future__ import annotations

import argparse
import json
import socket
import struct
import sys
import threading
import time

import pytest

websocket = pytest.importorskip("websocket", reason="websocket-client not installed")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def live_server():
    """Start piper_multi_server in a background thread on a free port."""
    root = str(__import__("pathlib").Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    import piper_multi_server as pms

    port = _free_port()
    args = argparse.Namespace(
        host="127.0.0.1", port=port, model=None, voices_path="voices-piper",
        scan_dir=None, cuda=False, sentence_silence=0.0,
        soundpath="../voices/%s.wav", bearer="", whisper_model="base",
        admin_token="test-admin-token", startup_callback="",
        startup_callback_token="", sip_port=0, debug=False,
    )
    app = pms.create_app(args)

    from werkzeug.serving import make_server
    server = make_server("127.0.0.1", port, app, threaded=True)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    # Wait for server to accept connections.
    for _ in range(50):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                break
        except OSError:
            time.sleep(0.05)

    base = f"http://127.0.0.1:{port}"
    ws_base = f"ws://127.0.0.1:{port}"
    yield {"base": base, "ws_base": ws_base, "port": port,
           "admin_token": "test-admin-token"}
    server.shutdown()


@pytest.fixture
def call_with_webclient(live_server):
    """Create account/subscriber + call + webclient slot, return ids."""
    import requests

    admin = {"Authorization": "Bearer " + live_server["admin_token"],
             "Content-Type": "application/json"}
    base = live_server["base"]

    # Provision account, subscriber, PBX, webclient feature.
    requests.put(base + "/api/pbx/E2EPBX",
                 data=json.dumps({"sip_proxy": "", "sip_user": ""}),
                 headers=admin)
    requests.put(base + "/api/accounts/E2EAcc",
                 data=json.dumps({
                     "token": "e2e-tok", "pbx": "E2EPBX",
                     "features": ["webclient"],
                 }),
                 headers=admin)
    acct = {"Authorization": "Bearer e2e-tok",
            "Content-Type": "application/json"}
    requests.put(base + "/api/subscribe/e2e-sub",
                 data=json.dumps({
                     "base_url": "https://crm.example.com/crm",
                     "bearer_token": "e2e-tok",
                 }),
                 headers=acct)

    resp = requests.post(base + "/api/calls",
                         data=json.dumps({"subscriber_id": "e2e-sub"}),
                         headers=acct)
    assert resp.status_code == 201, resp.text
    call_id = resp.json()["call_id"]

    # Create webclient session directly via DSL action.
    resp = requests.post(
        base + "/api/pipelines",
        data=json.dumps({
            "dsl": 'webclient:e2e_user{"callback":"/cb",'
                   '"base_url":"https://speech.example.com/tts",'
                   f'"call_id":"{call_id}"}}',
        }),
        headers=acct,
    )
    if resp.status_code != 201:
        pytest.skip(f"webclient action failed: {resp.status_code} {resp.text}")

    # Look up session_id via the webclient module directly.
    from speech_pipeline.telephony import webclient as wc_mod
    with wc_mod._sessions_lock:
        for sid, entry in wc_mod._sessions.items():
            if entry.get("call_id") == call_id:
                session_id = sid
                nonce = entry.get("nonce")
                break
        else:
            pytest.skip("webclient session not registered")

    yield {
        "call_id": call_id,
        "session_id": session_id,
        "nonce": nonce,
        "admin": admin,
        "acct": acct,
        "base": base,
        "ws_base": live_server["ws_base"],
    }

    requests.delete(base + f"/api/calls/{call_id}", headers=acct)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _attach_codec_pipe(ctx, *, tee=False, webhook_url=None):
    import requests

    if tee:
        tap = f"{ctx['session_id']}_tap"
        pipes = [
            f"codec:{ctx['session_id']} -> tee:{tap} -> call:{ctx['call_id']} -> codec:{ctx['session_id']}"
        ]
        if webhook_url:
            pipes.append(f"tee:{tap} -> stt:de -> webhook:{webhook_url}")
    else:
        pipes = [f"codec:{ctx['session_id']} -> call:{ctx['call_id']} -> codec:{ctx['session_id']}"]

    for dsl in pipes:
        resp = requests.post(
            ctx["base"] + "/api/pipelines",
            data=json.dumps({"dsl": dsl}),
            headers=ctx["acct"],
        )
        assert resp.status_code == 201, resp.text


class TestWebclientWSHandshake:
    """The exact flow a browser does when the user clicks the green handset."""

    def test_scoped_session_precreates_codec_socket(self, live_server):
        """Regression: scoped ``account:wc-...`` IDs must work on /ws/socket.

        The route used to only pre-create codec sessions for legacy-local
        IDs starting with ``wc-``. After account scoping landed, real browser
        joins hit ``/ws/socket/account:wc-...`` and got an immediate error.
        """
        from speech_pipeline.telephony import auth as auth_mod
        from speech_pipeline.telephony import call_state, webclient as wc_mod

        call = call_state.create_call("ws-sub", "ws-acc", "ws-pbx")
        try:
            nonce = auth_mod.create_nonce(
                account_id=call.account_id,
                subscriber_id=call.subscriber_id,
                user="e2e-user",
            )["nonce"]
            sess = wc_mod.register_webclient(call, "e2e-user", nonce)
            session_id = sess["session_id"]
            assert ":" in session_id, session_id

            codec_ws = websocket.create_connection(
                live_server["ws_base"] + "/ws/socket/" + session_id,
                timeout=5,
            )
            try:
                codec_ws.send(json.dumps({"type": "hello", "profiles": ["medium"]}))
                codec_ws.settimeout(5.0)
                resp = codec_ws.recv()
                assert isinstance(resp, str), resp
                obj = json.loads(resp)
                assert obj.get("type") == "hello", obj
                assert obj.get("session_id") == session_id
                assert "error" not in obj, obj
            finally:
                codec_ws.close()
        finally:
            wc_mod.close_call_sessions(call.call_id)
            call_state.delete_call(call.call_id)

    def test_codec_handshake_returns_hello(self, call_with_webclient):
        ctx = call_with_webclient
        codec_ws = websocket.create_connection(
            ctx["ws_base"] + "/ws/socket/" + ctx["session_id"],
            timeout=5,
        )
        try:
            codec_ws.send(json.dumps({"type": "hello",
                                       "profiles": ["medium"]}))
            codec_ws.settimeout(5.0)
            resp = codec_ws.recv()
            assert isinstance(resp, str), (
                f"expected JSON hello, got binary first: {resp!r}"
            )
            obj = json.loads(resp)
            assert obj.get("type") == "hello", (
                f"server did not send hello — got {obj}"
            )
            assert obj.get("session_id") == ctx["session_id"]
            assert obj.get("profile") in (
                "high", "medium", "low", "lowest"
            )
        finally:
            codec_ws.close()

    def test_codec_session_survives_disconnect_reconnect(
            self, call_with_webclient):
        """Browser does disconnect-reconnect on page load; session must
        persist so the second /ws/socket connection finds it."""
        ctx = call_with_webclient
        ws1 = websocket.create_connection(
            ctx["ws_base"] + "/ws/socket/" + ctx["session_id"], timeout=5,
        )
        ws1.send(json.dumps({"type": "hello", "profiles": ["low"]}))
        ws1.settimeout(5)
        hello1 = json.loads(ws1.recv())
        assert hello1.get("type") == "hello"
        ws1.close()
        time.sleep(0.3)

        ws2 = websocket.create_connection(
            ctx["ws_base"] + "/ws/socket/" + ctx["session_id"], timeout=5,
        )
        try:
            ws2.send(json.dumps({"type": "hello", "profiles": ["low"]}))
            ws2.settimeout(5)
            hello2 = ws2.recv()
            assert isinstance(hello2, str), (
                f"After reconnect the codec WS did NOT send hello — "
                f"got {hello2!r}.  Session was torn down on disconnect."
            )
            obj = json.loads(hello2)
            assert obj.get("type") == "hello", obj
        finally:
            ws2.close()

    def test_codec_ws_opens_before_pipeline_attach(self, call_with_webclient):
        """The slot alone must pre-create the codec session."""
        ctx = call_with_webclient

        # Skip /ws/pipe entirely — just hit /ws/socket directly.
        codec_ws = websocket.create_connection(
            ctx["ws_base"] + "/ws/socket/" + ctx["session_id"], timeout=5,
        )
        try:
            codec_ws.send(json.dumps({"type": "hello", "profiles": ["low"]}))
            codec_ws.settimeout(5)
            resp = codec_ws.recv()
            assert isinstance(resp, str), resp
            obj = json.loads(resp)
            assert "error" not in obj, (
                f"Server returned error on pre-pipe /ws/socket connect: {obj}"
            )
            assert obj.get("type") == "hello", obj
            assert obj.get("session_id") == ctx["session_id"]
        finally:
            codec_ws.close()

    def test_explicit_codec_pipe_can_be_attached_after_slot_creation(
            self, call_with_webclient):
        _attach_codec_pipe(call_with_webclient)

    def test_audio_flows_browser_to_conference(self, call_with_webclient):
        """Browser-encoded frames must end up as audio in the conference
        mixer.  This is the actual thing the user wants to test."""
        ctx = call_with_webclient
        _attach_codec_pipe(ctx)

        # Attach an output sink to the call's mixer so we can measure.
        from speech_pipeline.telephony import call_state
        call = call_state.get_call(ctx["call_id"])
        assert call is not None
        out_q = call.mixer.add_output()
        codec_ws = websocket.create_connection(
            ctx["ws_base"] + "/ws/socket/" + ctx["session_id"], timeout=5,
        )
        try:
            codec_ws.send(json.dumps({
                "type": "hello", "profiles": ["low"],
            }))
            codec_ws.settimeout(5)
            hello = json.loads(codec_ws.recv())
            assert hello.get("type") == "hello"

            from speech_pipeline import fourier_codec as fc
            import numpy as np
            tone = (np.sin(
                2 * np.pi * 440 / fc.SAMPLE_RATE
                * np.arange(fc.FRAME_SAMPLES * 10)
            ) * 0.4).astype(np.float32)
            profile = hello["profile"]
            for i in range(10):
                frame = tone[i * fc.FRAME_SAMPLES
                             : (i + 1) * fc.FRAME_SAMPLES]
                encoded = fc.encode_frame(frame, profile)
                codec_ws.send_binary(encoded)
                time.sleep(0.02)

            import queue as _q
            collected = b""
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline and len(collected) < 48000:
                try:
                    frame = out_q.get(timeout=0.2)
                    if frame:
                        collected += frame
                except _q.Empty:
                    continue
            rms = _rms(collected)
            assert rms > 50, (
                f"Browser audio did not reach the mixer (RMS={rms:.0f})"
            )
        finally:
            codec_ws.close()


def _rms(pcm: bytes) -> float:
    if not pcm:
        return 0.0
    import numpy as np
    arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr ** 2)))
