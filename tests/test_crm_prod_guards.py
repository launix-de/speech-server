"""Prod-relevant blind spots (audit #1–#4).

#1 Subscriber events drift — server must warn when a registered
   subscriber omits an event key the server actually fires.
#2 Webclient → SIP-peer audio path (the original prod complaint:
   SIP phone did not hear the webclient user).
#3 DTMF webhook — registered but never tested end-to-end.
#4 max_concurrent_calls enforcement + decrement after DELETE.
"""
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from conftest import ADMIN_TOKEN, ACCOUNT_TOKEN, ACCOUNT_ID, SUBSCRIBER_ID
from fake_crm import FakeCrm
from speech_pipeline.telephony import (
    auth as auth_mod,
    call_state,
    dispatcher,
    leg as leg_mod,
    subscriber as sub_mod,
    _shared as sh,
)

websocket = pytest.importorskip("websocket",
                                 reason="websocket-client not installed")

QUEUE_MP3 = Path(__file__).parent.parent / "examples" / "queue.mp3"


# ---------------------------------------------------------------------------
# #1. Subscriber events drift
# ---------------------------------------------------------------------------

class TestSubscriberEventsDrift:
    """The FakeCrm bug that slipped past CI (``ended`` vs ``call_ended``)
    must be detectable: when a subscriber registers with a missing
    event key, the server must at least warn.  Contract: WARN logged
    on PUT /api/subscribe whenever a known-fired key is absent."""

    def test_subscriber_missing_call_ended_logs_warning(
            self, client, admin, caplog):
        acct = {"Authorization": f"Bearer {ACCOUNT_TOKEN}",
                "Content-Type": "application/json"}
        client.put("/api/pbx/DriftPBX",
                   data=json.dumps({"sip_proxy": "", "sip_user": "",
                                     "sip_password": ""}), headers=admin)
        client.put("/api/accounts/DriftAcc",
                   data=json.dumps({"token": ACCOUNT_TOKEN,
                                     "pbx": "DriftPBX"}), headers=admin)
        with caplog.at_level("WARNING", logger="telephony.subscriber"):
            client.put(f"/api/subscribe/{SUBSCRIBER_ID}",
                       data=json.dumps({
                           "base_url": "https://crm.example.com/crm",
                           "bearer_token": "t",
                           "events": {
                               # Missing call_ended, dtmf, device_dial:
                               "incoming": "POST /p?state=incoming",
                           },
                       }), headers=acct)
        warnings = [r.message for r in caplog.records
                    if "call_ended" in r.message or "missing" in r.message.lower()]
        assert warnings, (
            "Server accepted a subscriber without 'call_ended' without "
            "warning — drift will silently break BYE propagation."
        )


# ---------------------------------------------------------------------------
# #2. Webclient → SIP-peer audio (the prod complaint)
# ---------------------------------------------------------------------------

def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _decode_mp3(sample_rate: int, duration_s: float) -> bytes:
    r = subprocess.run(
        ["ffmpeg", "-i", str(QUEUE_MP3), "-f", "s16le", "-ac", "1",
         "-ar", str(sample_rate), "-t", str(duration_s), "-"],
        capture_output=True,
    )
    assert r.returncode == 0 and r.stdout, (
        f"ffmpeg decode failed: {r.stderr.decode()[:200]}"
    )
    return r.stdout


class TestWebclientToSipPeer:
    """The original prod bug: Handy (SIP) heard silence from webclient.
    One conference, one RTP-backed SIP leg, one webclient slot.  Push
    queue.mp3 into the webclient codec socket; the SIP RTP session
    must receive audio that spectrally matches the source."""

    def test_webclient_mic_reaches_sip_peer(self, client, admin):
        from speech_pipeline.rtp_codec import PCMU
        from speech_pipeline.RTPSession import RTPSession, RTPCallSession
        from speech_pipeline import fourier_codec as fc
        import numpy as np
        import queue as _q

        acct = {"Authorization": f"Bearer {ACCOUNT_TOKEN}",
                "Content-Type": "application/json"}
        client.put("/api/pbx/WcSipPBX",
                   data=json.dumps({"sip_proxy": "", "sip_user": "",
                                     "sip_password": ""}), headers=admin)
        client.put("/api/accounts/WcSipAcc",
                   data=json.dumps({"token": ACCOUNT_TOKEN, "pbx": "WcSipPBX",
                                     "features": ["webclient"]}),
                   headers=admin)
        client.put(f"/api/subscribe/{SUBSCRIBER_ID}",
                   data=json.dumps({
                       "base_url": "https://crm.example.com/crm",
                       "bearer_token": "t",
                   }), headers=acct)
        call_sid = client.post("/api/calls",
                                data=json.dumps({"subscriber_id": SUBSCRIBER_ID}),
                                headers=acct).get_json()["call_id"]

        # Build a live-server loop for the WS path.
        import piper_multi_server as pms
        port = _free_port()
        args = argparse.Namespace(
            host="127.0.0.1", port=port, model=None,
            voices_path="voices-piper", scan_dir=None, cuda=False,
            sentence_silence=0.0, soundpath="../voices/%s.wav",
            bearer="", whisper_model="base", admin_token="wc2sip",
            startup_callback="", startup_callback_token="",
            sip_port=0, debug=False,
        )
        app = pms.create_app(args)
        # Register the same account/subscriber/call on this second app
        # instance (separate process state).
        admin2 = {"Authorization": "Bearer wc2sip",
                  "Content-Type": "application/json"}
        with app.test_client() as c2:
            c2.put("/api/pbx/WcSipPBX",
                   data=json.dumps({"sip_proxy": "", "sip_user": "",
                                     "sip_password": ""}), headers=admin2)
            c2.put("/api/accounts/WcSipAcc",
                   data=json.dumps({"token": ACCOUNT_TOKEN, "pbx": "WcSipPBX",
                                     "features": ["webclient"]}),
                   headers=admin2)
            acct2 = {"Authorization": f"Bearer {ACCOUNT_TOKEN}",
                     "Content-Type": "application/json"}
            c2.put(f"/api/subscribe/{SUBSCRIBER_ID}",
                   data=json.dumps({
                       "base_url": "https://crm.example.com/crm",
                       "bearer_token": "t",
                   }), headers=acct2)
            call_sid2 = c2.post("/api/calls",
                                 data=json.dumps({"subscriber_id": SUBSCRIBER_ID}),
                                 headers=acct2).get_json()["call_id"]

            # RTP peer (the "phone") + server-side RTP session wired into
            # a Leg (direct stage construction — no real SIP signalling).
            local_port = _free_port()
            remote_port = _free_port()
            phone = RTPSession(remote_port, "127.0.0.1", local_port,
                                codec=PCMU.new_session_codec())
            phone.start()
            server_rtp = RTPSession(local_port, "127.0.0.1", remote_port,
                                     codec=PCMU.new_session_codec())
            server_rtp.start()
            sip_session = RTPCallSession(server_rtp)
            sip_leg = leg_mod.Leg(
                leg_id=f"leg-rtp-{local_port}", direction="inbound",
                number="+49174", pbx_id="WcSipPBX",
                subscriber_id=SUBSCRIBER_ID,
            )
            sip_leg.voip_call = server_rtp
            sip_leg.sip_session = sip_session
            sip_leg.rtp_session = server_rtp
            leg_mod._legs[sip_leg.leg_id] = sip_leg

            try:
                # Bridge the SIP leg to the conference.
                c2.post("/api/pipelines", data=json.dumps({
                    "dsl": f"sip:{sip_leg.leg_id} -> call:{call_sid2} "
                           f"-> sip:{sip_leg.leg_id}",
                }), headers=acct2)
                time.sleep(0.2)

                # Create the webclient slot on the same call.
                dsl = ('webclient:user_phone' + json.dumps({
                    "callback": f"/cb?call=1&participant=2",
                    "base_url": f"http://127.0.0.1:{port}",
                    "call_id": call_sid2,
                }))
                resp = c2.post("/api/pipelines",
                               data=json.dumps({"dsl": dsl}),
                               headers=acct2)
                assert resp.status_code == 201, resp.get_data(as_text=True)
                session_id = resp.get_json()["session_id"]

                # Run a real HTTP server for /ws/socket.
                from werkzeug.serving import make_server
                server = make_server("127.0.0.1", port, app, threaded=True)
                t = threading.Thread(target=server.serve_forever,
                                      daemon=True)
                t.start()
                for _ in range(50):
                    try:
                        with socket.create_connection(("127.0.0.1", port),
                                                        timeout=0.2):
                            break
                    except OSError:
                        time.sleep(0.05)

                ws = websocket.create_connection(
                    f"ws://127.0.0.1:{port}/ws/socket/{session_id}",
                    timeout=5)
                try:
                    ws.send(json.dumps({"type": "hello",
                                         "profiles": ["low"]}))
                    ws.settimeout(5)
                    hello = json.loads(ws.recv())
                    profile = hello["profile"]

                    pcm = _decode_mp3(fc.SAMPLE_RATE, duration_s=3.0)
                    samples = (np.frombuffer(pcm, dtype=np.int16)
                               .astype(np.float32) / 32768.0)
                    n = len(samples) // fc.FRAME_SAMPLES

                    # Drain anything the phone already got (silence).
                    while not phone.rx_queue.empty():
                        phone.rx_queue.get_nowait()

                    def _push():
                        for i in range(n):
                            frame = samples[i * fc.FRAME_SAMPLES
                                            : (i + 1) * fc.FRAME_SAMPLES]
                            ws.send_binary(fc.encode_frame(frame, profile))
                            time.sleep(0.01)
                    threading.Thread(target=_push, daemon=True).start()

                    collected = b""
                    deadline = time.monotonic() + 5.0
                    while time.monotonic() < deadline and len(collected) < 32000:
                        try:
                            chunk = phone.rx_queue.get(timeout=0.2)
                            if chunk:
                                collected += chunk
                        except _q.Empty:
                            continue

                    arr = np.frombuffer(collected,
                                          dtype=np.int16).astype(np.float64)
                    rms = float(np.sqrt(np.mean(arr ** 2))) if arr.size else 0
                    assert rms > 100, (
                        f"SIP peer heard silence from webclient "
                        f"(RMS={rms:.0f}).  The original prod bug: "
                        f"mic audio never crosses into the SIP leg."
                    )
                    ref = _decode_mp3(PCMU.sample_rate, 2.0)
                    a = np.frombuffer(ref, dtype=np.int16).astype(np.float64)
                    m = min(len(a), len(arr))
                    spec_a = np.abs(np.fft.rfft(a[:m]))
                    spec_b = np.abs(np.fft.rfft(arr[:m]))
                    ea = np.array([np.sum(x ** 2)
                                    for x in np.array_split(spec_a, 10)])
                    eb = np.array([np.sum(x ** 2)
                                    for x in np.array_split(spec_b, 10)])
                    sim = float(np.dot(ea, eb) /
                                 (np.linalg.norm(ea) * np.linalg.norm(eb)
                                  + 1e-9))
                    assert sim > 0.5, (
                        f"SIP peer received something but it doesn't "
                        f"match queue.mp3 spectrum (sim={sim:.3f}) — "
                        f"mic audio is corrupted on the way."
                    )
                finally:
                    ws.close()
                    server.shutdown()
            finally:
                phone.stop()
                server_rtp.stop()
                leg_mod._legs.pop(sip_leg.leg_id, None)
                c2.delete(f"/api/calls/{call_sid2}", headers=acct2)
        client.delete(f"/api/calls/{call_sid}", headers=acct)


# ---------------------------------------------------------------------------
# #3. DTMF webhook
# ---------------------------------------------------------------------------

class TestDtmfCallback:
    """CRM DSL can opt into DTMF via ``sip:LEG{"dtmf":"/cb"}``; when the
    server's monitor sees a digit it fires ``fire_callback(leg, 'dtmf',
    digit=...)``, which becomes a POST to the subscriber."""

    def test_dtmf_fires_webhook_with_digit_payload(
            self, client, account, admin, monkeypatch):
        # Provision subscriber so the leg has a place to send.
        client.put(f"/api/subscribe/{SUBSCRIBER_ID}",
                   data=json.dumps({
                       "base_url": "https://crm.example.com/crm",
                       "bearer_token": "t",
                   }), headers=account)

        captured: list[dict] = []

        def _capture(url, payload, bearer_token, **kw):
            captured.append({"url": url, "payload": payload})
        monkeypatch.setattr(sh, "post_webhook", _capture)

        leg = leg_mod.Leg(
            leg_id="leg-dtmf", direction="outbound", number="+49",
            pbx_id="TestPBX", subscriber_id=SUBSCRIBER_ID,
        )
        leg.callbacks = {
            "dtmf": "/Telephone/SpeechServer/public?state=dtmf&call=1",
        }
        leg_mod._legs[leg.leg_id] = leg
        try:
            leg_mod.fire_callback(leg, "dtmf", digit="5",
                                    call_id="call-x")
            assert captured, (
                "DTMF callback never fired — CRMs registered dtmf in "
                "heartbeat.fop for nothing."
            )
            assert captured[0]["payload"].get("digit") == "5"
            assert "state=dtmf" in captured[0]["url"]
        finally:
            leg_mod._legs.pop(leg.leg_id, None)


# ---------------------------------------------------------------------------
# #4. max_concurrent_calls enforcement + decrement
# ---------------------------------------------------------------------------

class TestMaxConcurrentCalls:

    def test_limit_rejects_second_call_with_429(self, client, admin):
        acct = {"Authorization": "Bearer cap-tok",
                "Content-Type": "application/json"}
        client.put("/api/pbx/CapPBX",
                   data=json.dumps({"sip_proxy": "", "sip_user": "",
                                     "sip_password": ""}), headers=admin)
        client.put("/api/accounts/CapAcc",
                   data=json.dumps({"token": "cap-tok", "pbx": "CapPBX",
                                     "max_concurrent_calls": 1}),
                   headers=admin)
        client.put("/api/subscribe/cap-sub",
                   data=json.dumps({"base_url": "https://crm.example.com",
                                     "bearer_token": "t"}), headers=acct)

        r1 = client.post("/api/calls",
                          data=json.dumps({"subscriber_id": "cap-sub"}),
                          headers=acct)
        assert r1.status_code == 201, r1.get_data(as_text=True)
        call_sid = r1.get_json()["call_id"]

        r2 = client.post("/api/calls",
                          data=json.dumps({"subscriber_id": "cap-sub"}),
                          headers=acct)
        assert r2.status_code == 429, (
            f"Second call above max_concurrent_calls=1 was not "
            f"rejected: {r2.status_code}"
        )

        # DELETE the first call; counter must decrement so a new call
        # is accepted.
        client.delete(f"/api/calls/{call_sid}", headers=acct)
        r3 = client.post("/api/calls",
                          data=json.dumps({"subscriber_id": "cap-sub"}),
                          headers=acct)
        assert r3.status_code == 201, (
            f"max_concurrent counter did not decrement after DELETE: "
            f"{r3.status_code} {r3.get_data(as_text=True)}"
        )
        client.delete(f"/api/calls/{r3.get_json()['call_id']}",
                        headers=acct)
