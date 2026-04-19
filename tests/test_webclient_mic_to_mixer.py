"""Regression: webclient browser mic audio must reach the other participant.

Pure black-box test — only HTTP API + WebSocket protocol, no in-process
Python hooks.  This matches the real browser/CRM flow exactly.

1. Provision account + webclient feature via ``/api/accounts``.
2. Create a call via ``POST /api/calls``.
3. Create TWO webclient slots on the same call via the
   ``webclient:USER{...}`` DSL action on ``/api/pipelines`` —
   one acts as A (speaker, sends mic), one as B (listener).
4. Each side opens ``/ws/socket/<sid>``, handshakes.
5. A pushes encoded ``examples/queue.mp3`` frames.
6. B must receive non-silence on its WebSocket.

If B hears silence, the production bug is reproduced: the webclient
audio never propagates to other participants in the conference.
"""
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

websocket = pytest.importorskip("websocket", reason="websocket-client not installed")

QUEUE_MP3 = Path(__file__).parent.parent / "examples" / "queue.mp3"


def _decode_mp3_to_s16le(sample_rate: int, duration_s: float = 3.0) -> bytes:
    result = subprocess.run(
        ["ffmpeg", "-i", str(QUEUE_MP3), "-f", "s16le", "-ac", "1",
         "-ar", str(sample_rate), "-t", str(duration_s), "-"],
        capture_output=True,
    )
    assert result.returncode == 0 and result.stdout, (
        f"ffmpeg decode failed: {result.stderr.decode()[:200]}"
    )
    return result.stdout


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _rms_f32(samples) -> float:
    import numpy as np
    if len(samples) == 0:
        return 0.0
    return float(np.sqrt(float((samples.astype("float64") ** 2).mean())))


@pytest.fixture(scope="module")
def live_server():
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    import piper_multi_server as pms

    port = _free_port()
    args = argparse.Namespace(
        host="127.0.0.1", port=port, model=None, voices_path="voices-piper",
        scan_dir=None, cuda=False, sentence_silence=0.0,
        soundpath="../voices/%s.wav", bearer="", whisper_model="base",
        admin_token="mic-admin", startup_callback="",
        startup_callback_token="", sip_port=0, debug=False,
    )
    app = pms.create_app(args)

    from werkzeug.serving import make_server
    server = make_server("127.0.0.1", port, app, threaded=True)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    for _ in range(50):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                break
        except OSError:
            time.sleep(0.05)

    yield {
        "base": f"http://127.0.0.1:{port}",
        "ws_base": f"ws://127.0.0.1:{port}",
        "admin_token": "mic-admin",
    }
    server.shutdown()


def _provision(base, admin, token, pbx, account, subscriber):
    import requests
    requests.put(f"{base}/api/pbx/{pbx}",
                 data=json.dumps({"sip_proxy": "", "sip_user": ""}),
                 headers=admin)
    requests.put(f"{base}/api/accounts/{account}",
                 data=json.dumps({"token": token, "pbx": pbx,
                                   "features": ["webclient"]}),
                 headers=admin)
    acct = {"Authorization": f"Bearer {token}",
            "Content-Type": "application/json"}
    requests.put(f"{base}/api/subscribe/{subscriber}",
                 data=json.dumps({"base_url": "https://crm.example.com/crm",
                                   "bearer_token": token}),
                 headers=acct)
    return acct


def _create_webclient_slot(base, acct, call_id, user):
    """POST the webclient: DSL action exactly like joinWebclientAction.
    Returns session_id by correlating the call + user on a follow-up
    GET of the pipelines endpoint."""
    import requests
    params = {
        "callback": "/cb",
        "base_url": "https://speech.example.com/tts",
        "call_id": call_id,
    }
    payload = {"dsl": f'webclient:{user}{json.dumps(params)}'}
    resp = requests.post(f"{base}/api/pipelines",
                         data=json.dumps(payload), headers=acct)
    assert resp.status_code == 201, resp.text

    # Poll /api/pipelines?dsl=call:<id> to find participants of type
    # webclient associated with this user.
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        info = requests.get(
            f"{base}/api/pipelines",
            params={"dsl": f"call:{call_id}"}, headers=acct,
        ).json()
        for p in info.get("participants", []):
            if p.get("type") == "webclient" and p.get("user") == user:
                return p["id"]
        time.sleep(0.05)
    raise AssertionError("webclient participant never appeared in call")


def _attach_webclient(base, acct, call_id, session_id):
    import requests
    dsl = f"codec:{session_id} -> call:{call_id} -> codec:{session_id}"
    resp = requests.post(f"{base}/api/pipelines",
                         data=json.dumps({"dsl": dsl}), headers=acct)
    assert resp.status_code == 201, resp.text


class TestBrowserMicReachesOtherParticipant:

    def test_queue_mp3_from_A_reaches_B(self, live_server):
        import requests

        admin = {"Authorization": "Bearer " + live_server["admin_token"],
                 "Content-Type": "application/json"}
        base = live_server["base"]
        ws_base = live_server["ws_base"]

        acct = _provision(base, admin, "mic-tok", "MicPBX",
                          "MicAcc", "mic-sub")

        call_id = requests.post(
            f"{base}/api/calls",
            data=json.dumps({"subscriber_id": "mic-sub"}),
            headers=acct,
        ).json()["call_id"]

        try:
            sid_a = _create_webclient_slot(base, acct, call_id, "userA")
            sid_b = _create_webclient_slot(base, acct, call_id, "userB")
            _attach_webclient(base, acct, call_id, sid_a)
            _attach_webclient(base, acct, call_id, sid_b)

            ws_a = websocket.create_connection(
                f"{ws_base}/ws/socket/{sid_a}", timeout=5)
            ws_b = websocket.create_connection(
                f"{ws_base}/ws/socket/{sid_b}", timeout=5)
            if True:
                for ws in (ws_a, ws_b):
                    ws.send(json.dumps({"type": "hello",
                                         "profiles": ["low"]}))
                ws_a.settimeout(5)
                ws_b.settimeout(5)
                hello_a = json.loads(ws_a.recv())
                hello_b = json.loads(ws_b.recv())
                assert hello_a["type"] == "hello" and hello_b["type"] == "hello"
                profile_a = hello_a["profile"]

                # Decode queue.mp3 at codec rate; encode and push from A.
                from speech_pipeline import fourier_codec as fc
                import numpy as np

                pcm = _decode_mp3_to_s16le(fc.SAMPLE_RATE, duration_s=3.0)
                samples = (np.frombuffer(pcm, dtype=np.int16)
                           .astype(np.float32) / 32768.0)
                n_frames = len(samples) // fc.FRAME_SAMPLES

                def _push():
                    for i in range(n_frames):
                        frame = samples[i * fc.FRAME_SAMPLES
                                        : (i + 1) * fc.FRAME_SAMPLES]
                        ws_a.send_binary(fc.encode_frame(frame, profile_a))
                        time.sleep(0.01)
                sender = threading.Thread(target=_push, daemon=True)
                sender.start()

                # B receives decoded PCM frames on its socket.
                received = np.zeros(0, dtype=np.float32)
                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline and len(received) < fc.SAMPLE_RATE * 2:
                    try:
                        ws_b.settimeout(1.0)
                        msg = ws_b.recv()
                    except Exception:
                        break
                    if isinstance(msg, (bytes, bytearray)):
                        decoded, _profile = fc.decode_frame(bytes(msg))
                        received = np.concatenate([received, decoded])

                sender.join(timeout=1.0)

                rms = _rms_f32(received)
                assert rms > 0.01, (
                    f"B received silence (RMS={rms:.4f}, "
                    f"samples={len(received)}) — webclient A's mic audio "
                    f"did not propagate through the conference to B."
                )
        finally:
            try: ws_a.close()
            except Exception: pass
            try: ws_b.close()
            except Exception: pass
            requests.delete(f"{base}/api/calls/{call_id}", headers=acct)
