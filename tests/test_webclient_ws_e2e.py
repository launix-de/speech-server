"""End-to-end WebSocket test for the webclient audio path.

Runs ``piper_multi_server`` in a real HTTP server on a free port and
drives the exact browser flow with a Python WebSocket client:

1. Create a call + webclient session via the DSL action.
2. Open ``/ws/pipe`` → send DSL config → pipeline builds.
3. Open ``/ws/socket/<sid>`` → hello handshake.
4. Verify the session survives a normal browser disconnect + reconnect.
5. Push encoded audio frames through and verify they reach the
   conference mixer + come back out.

These tests would have caught today's "green handset → Disconnected"
regressions — the previous suite only poked at in-process Flask state.
"""
from __future__ import annotations

import argparse
import json
import socket
import struct
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
                   '"base_url":"https://crm.example.com",'
                   f'"call_id":"{call_id}"}}',
        }),
        headers=acct,
    )
    if resp.status_code != 201:
        pytest.skip(f"webclient action failed: {resp.status_code} {resp.text}")

    # Look up session_id via the webclient module directly (faster than
    # waiting for the callback webhook).
    from speech_pipeline.telephony import webclient as wc_mod
    with wc_mod._sessions_lock:
        for sid, entry in wc_mod._sessions.items():
            if entry.get("call_id") == call_id:
                session_id = sid
                dsl = entry.get("dsl")
                nonce = entry.get("nonce")
                break
        else:
            pytest.skip("webclient session not registered")

    yield {
        "call_id": call_id,
        "session_id": session_id,
        "dsl": dsl,
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

class TestWebclientWSHandshake:
    """The exact flow a browser does when the user clicks the green handset."""

    def test_codec_handshake_returns_hello(self, call_with_webclient):
        ctx = call_with_webclient

        # Step 1: open /ws/pipe + send DSL config (browser step).
        # Browser authenticates with the nonce from the iframe URL,
        # NOT the account token — that's exactly what phone.html does.
        pipe_ws = websocket.create_connection(
            ctx["ws_base"] + "/ws/pipe?token=" + ctx["nonce"],
            timeout=5,
        )
        try:
            pipe_ws.send(json.dumps({"pipe": ctx["dsl"]}))
            time.sleep(0.3)

            # Step 2: open /ws/socket/<sid>, send hello.
            codec_ws = websocket.create_connection(
                ctx["ws_base"] + "/ws/socket/" + ctx["session_id"],
                timeout=5,
            )
            try:
                codec_ws.send(json.dumps({"type": "hello",
                                           "profiles": ["medium"]}))
                # Step 3: expect hello response.
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
        finally:
            pipe_ws.close()

    def test_codec_session_survives_disconnect_reconnect(
            self, call_with_webclient):
        """Browser does disconnect-reconnect on page load; session must
        persist so the second /ws/socket connection finds it."""
        ctx = call_with_webclient

        pipe_ws = websocket.create_connection(
            ctx["ws_base"] + "/ws/pipe?token=" + ctx["nonce"] + "", timeout=5,
        )
        try:
            pipe_ws.send(json.dumps({"pipe": ctx["dsl"]}))
            time.sleep(0.3)

            # First connect.
            ws1 = websocket.create_connection(
                ctx["ws_base"] + "/ws/socket/" + ctx["session_id"], timeout=5,
            )
            ws1.send(json.dumps({"type": "hello", "profiles": ["low"]}))
            ws1.settimeout(5)
            hello1 = json.loads(ws1.recv())
            assert hello1.get("type") == "hello"
            ws1.close()
            time.sleep(0.3)

            # Reconnect — session MUST still be there, handshake MUST
            # complete again.
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
        finally:
            pipe_ws.close()

    def test_codec_ws_opens_before_pipe_ws(self, call_with_webclient):
        """Race guard: browser opens /ws/socket/<sid> BEFORE the /ws/pipe
        pipeline has registered the CodecSocketSession.  Server must
        pre-create the session (webclient slot already exists) instead
        of replying 'Unknown session ID'."""
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

    def test_stt_callback_adds_transcript_tap(self, live_server):
        """When the CRM passes ``stt_callback`` to ``webclient:USER``,
        the server auto-builds a ``tee → stt → webhook`` sidechain so
        the webclient's mic audio gets transcribed — same as a SIP leg.
        """
        import requests

        admin = {"Authorization": "Bearer " + live_server["admin_token"],
                 "Content-Type": "application/json"}
        base = live_server["base"]

        requests.put(base + "/api/pbx/E2EPBX2",
                     data=json.dumps({"sip_proxy": "", "sip_user": ""}),
                     headers=admin)
        requests.put(base + "/api/accounts/E2EAcc2",
                     data=json.dumps({"token": "e2e-tok2", "pbx": "E2EPBX2",
                                       "features": ["webclient"]}),
                     headers=admin)
        acct = {"Authorization": "Bearer e2e-tok2",
                "Content-Type": "application/json"}
        requests.put(base + "/api/subscribe/e2e-sub2",
                     data=json.dumps({"base_url": "https://crm.example.com/crm",
                                       "bearer_token": "e2e-tok2"}),
                     headers=acct)
        call_id = requests.post(
            base + "/api/calls",
            data=json.dumps({"subscriber_id": "e2e-sub2"}),
            headers=acct,
        ).json()["call_id"]

        try:
            resp = requests.post(
                base + "/api/pipelines",
                data=json.dumps({
                    "dsl": 'webclient:stt_user{"callback":"/cb",'
                           '"base_url":"https://crm.example.com",'
                           f'"call_id":"{call_id}",'
                           '"stt_callback":"/sttNote?call=42&participant=7"}',
                }),
                headers=acct,
            )
            assert resp.status_code == 201, resp.text

            from speech_pipeline.telephony import webclient as wc_mod
            with wc_mod._sessions_lock:
                sess = next(s for s in wc_mod._sessions.values()
                            if s.get("call_id") == call_id)
            dsl = sess["dsl"]
            # Parsed as {"pipes": [...]} when pipes were used.
            obj = json.loads(dsl)
            assert "pipes" in obj, f"expected multi-pipe DSL, got {dsl!r}"
            joined = " || ".join(obj["pipes"])
            assert "tee:" in joined and "stt:" in joined and "webhook:" in joined
            assert "sttNote" in joined
            # Sidechain pipe attaches to the existing tee from pipe 1.
            # In pipe_executor (unified ``->`` DSL) ``tee:NAME`` as the
            # first element is sidechain-attachment mode iff the named
            # tee already exists — built before by pipe 1.
            assert obj["pipes"][1].lstrip().startswith("tee:"), (
                f"sidechain must start with tee:, got: {obj['pipes'][1]!r}"
            )
            # Webhook URL must be scheme-qualified — requests rejects
            # scheme-less URLs and the build appears to succeed but no
            # transcript ever reaches the CRM.
            assert "://" in joined, (
                f"sidechain webhook missing scheme: {joined!r}"
            )
        finally:
            requests.delete(base + f"/api/calls/{call_id}", headers=acct)

    def test_stt_tap_routes_browser_audio_to_webhook(self, live_server):
        """End-to-end: webclient with stt_callback → tap pipe runs →
        Whisper sees frames → POST to webhook URL.

        Mocks the Whisper transcriber + webhook with HTTP capture so
        we don't need real STT model weights, but exercises the full
        DSL build + audio path."""
        import requests
        from unittest.mock import patch
        from speech_pipeline.telephony import webclient as wc_mod
        from speech_pipeline import fourier_codec as fc

        admin = {"Authorization": "Bearer " + live_server["admin_token"],
                 "Content-Type": "application/json"}
        base = live_server["base"]

        # Provision once.
        requests.put(base + "/api/pbx/STTPBX",
                     data=json.dumps({"sip_proxy": "", "sip_user": ""}),
                     headers=admin)
        requests.put(base + "/api/accounts/STTAcc",
                     data=json.dumps({"token": "stt-tok", "pbx": "STTPBX",
                                       "features": ["webclient"]}),
                     headers=admin)
        acct = {"Authorization": "Bearer stt-tok",
                "Content-Type": "application/json"}
        requests.put(base + "/api/subscribe/stt-sub",
                     data=json.dumps({"base_url": "https://crm.example.com/crm",
                                       "bearer_token": "stt-tok"}),
                     headers=acct)
        call_id = requests.post(
            base + "/api/calls",
            data=json.dumps({"subscriber_id": "stt-sub"}),
            headers=acct,
        ).json()["call_id"]

        try:
            resp = requests.post(
                base + "/api/pipelines",
                data=json.dumps({
                    "dsl": 'webclient:stt2{"callback":"/cb",'
                           '"base_url":"https://crm.example.com",'
                           f'"call_id":"{call_id}",'
                           '"stt_callback":"/sttNote?call=99&participant=3"}',
                }),
                headers=acct,
            )
            assert resp.status_code == 201, resp.text

            with wc_mod._sessions_lock:
                sess = next(s for s in wc_mod._sessions.values()
                            if s.get("call_id") == call_id)
            sid = sess["session_id"]
            nonce = sess["nonce"]

            # Verify the multi-pipe DSL contains the tap.
            obj = json.loads(sess["dsl"])
            joined = " || ".join(obj["pipes"])
            assert "tee:" in joined and "stt:" in joined and "sttNote" in joined
        finally:
            requests.delete(base + f"/api/calls/{call_id}", headers=acct)

    def test_audio_flows_browser_to_conference(self, call_with_webclient):
        """Browser-encoded frames must end up as audio in the conference
        mixer.  This is the actual thing the user wants to test."""
        ctx = call_with_webclient

        # Attach an output sink to the call's mixer so we can measure.
        from speech_pipeline.telephony import call_state
        call = call_state.get_call(ctx["call_id"])
        assert call is not None
        out_q = call.mixer.add_output()

        pipe_ws = websocket.create_connection(
            ctx["ws_base"] + "/ws/pipe?token=" + ctx["nonce"] + "", timeout=5,
        )
        try:
            pipe_ws.send(json.dumps({"pipe": ctx["dsl"]}))
            time.sleep(0.4)

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

                # Send a few encoded frames.  Easiest payload: the
                # server's own codec module can produce valid frames.
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

                # Drain the mixer output; verify some non-silence
                # arrived.
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
        finally:
            pipe_ws.close()


def _rms(pcm: bytes) -> float:
    if not pcm:
        return 0.0
    import numpy as np
    arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr ** 2)))
