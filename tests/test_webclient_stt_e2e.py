"""End-to-end: browser mic → tee → STT → webhook POST to CRM.

Pure black-box flow:
1. CRM creates account + subscriber.
2. CRM POSTs ``webclient:USER{stt_callback=...}`` → receives session_id.
3. "Browser" opens ``/ws/socket/<sid>`` and pushes queue.mp3 frames.
4. Server's STT sidechain transcribes → WebhookSink POSTs to the
   stt_callback URL.
5. Test captures the HTTP POST (real Whisper is swapped for a fake
   transcriber so the test doesn't load a 100MB model) and asserts:
   - The webhook lands at the stt_callback URL.
   - The query carries the correct ``call`` and ``participant`` ids.
   - The body contains transcript JSON.

Rationale: individual ingredients (async build, mic→mixer, url_safety
bypass, DSL shape) already have green tests — but nobody proves the
full STT → webhook round trip.  This is the test that would have
failed earlier and forced us to find the url_safety regression.
"""
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import threading
import time
from pathlib import Path

import pytest

websocket = pytest.importorskip("websocket", reason="websocket-client not installed")

QUEUE_MP3 = Path(__file__).parent.parent / "examples" / "queue.mp3"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _decode_mp3_to_s16le(sample_rate: int, duration_s: float = 4.0) -> bytes:
    result = subprocess.run(
        ["ffmpeg", "-i", str(QUEUE_MP3), "-f", "s16le", "-ac", "1",
         "-ar", str(sample_rate), "-t", str(duration_s), "-"],
        capture_output=True,
    )
    assert result.returncode == 0 and result.stdout, (
        f"ffmpeg decode failed: {result.stderr.decode()[:200]}"
    )
    return result.stdout


def _make_fake_transcriber():
    """Build a Stage subclass (so pipe/set_upstream work) that consumes
    PCM and emits one NDJSON transcript line per accumulated second."""
    from speech_pipeline.base import AudioFormat, Stage

    class _FakeTranscriber(Stage):
        def __init__(self, *a, **kw):
            super().__init__()
            self.input_format = AudioFormat(16000, "s16le")
            self.output_format = AudioFormat(0, "ndjson")

        def ensure_model_loaded(self):
            pass

        def stream_pcm24k(self):
            return _transcribe_stream(self)

    return _FakeTranscriber()


def _transcribe_stream(self):
    if not self.upstream:
        return
    buffered = 0
    counter = 0
    for chunk in self.upstream.stream_pcm24k():
        if self.cancelled:
            break
        if not chunk:
            continue
        buffered += len(chunk)
        # 16kHz s16le → 32000 bytes per second
        while buffered >= 32000:
            buffered -= 32000
            counter += 1
            yield (json.dumps({
                "text": f"segment-{counter}",
                "start": counter - 1,
                "end": counter,
            }) + "\n").encode()


@pytest.fixture(scope="module")
def live_server():
    import piper_multi_server as pms

    port = _free_port()
    args = argparse.Namespace(
        host="127.0.0.1", port=port, model=None, voices_path="voices-piper",
        scan_dir=None, cuda=False, sentence_silence=0.0,
        soundpath="../voices/%s.wav", bearer="", whisper_model="base",
        admin_token="stt-admin", startup_callback="",
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
        "admin_token": "stt-admin",
        "port": port,
    }
    server.shutdown()


@pytest.fixture
def patched_stt(monkeypatch):
    """Swap Whisper for a deterministic fake transcriber so the test
    doesn't need to load ~100 MB of model weights."""
    from speech_pipeline.telephony import pipe_executor as pe
    original = pe.CallPipeExecutor._create_stage

    def _wrapped(self, typ, elem_id, params):
        if typ == "stt":
            return _make_fake_transcriber()
        return original(self, typ, elem_id, params)
    monkeypatch.setattr(pe.CallPipeExecutor, "_create_stage", _wrapped)


class TestWebclientSttEndToEnd:

    def test_mic_audio_produces_stt_webhook_to_crm(
            self, live_server, patched_stt, monkeypatch):
        """Push real speech into the webclient codec socket and assert
        a sttNote POST with the right call+participant ids arrives."""
        import requests as real_requests

        # -- fake CRM HTTP capture: monkeypatch requests.post in
        #    WebhookSink's module (WebhookSink uses its own
        #    `http_requests` alias).
        captured: list[dict] = []
        capture_lock = threading.Lock()

        from speech_pipeline import WebhookSink as ws_mod
        orig_post = ws_mod.http_requests.post

        def _capture_post(url, data=None, json=None, headers=None, **kw):
            # Only capture sttNote POSTs — let real api calls through.
            if "/sttNote" in (url or ""):
                with capture_lock:
                    captured.append({
                        "url": url,
                        "data": data,
                        "json": json,
                        "headers": headers or {},
                    })

                class _R:
                    status_code = 200
                    text = ""
                    content = b""
                    def json(self_inner): return {"ok": True}
                return _R()
            return orig_post(url, data=data, json=json, headers=headers, **kw)
        monkeypatch.setattr(ws_mod.http_requests, "post", _capture_post)

        admin = {"Authorization": "Bearer " + live_server["admin_token"],
                 "Content-Type": "application/json"}
        base = live_server["base"]
        real_requests.put(base + "/api/pbx/SttPBX",
                          data=json.dumps({"sip_proxy": "", "sip_user": ""}),
                          headers=admin)
        real_requests.put(base + "/api/accounts/SttAcc",
                          data=json.dumps({"token": "stt-tok", "pbx": "SttPBX",
                                             "features": ["webclient"]}),
                          headers=admin)
        acct = {"Authorization": "Bearer stt-tok",
                "Content-Type": "application/json"}

        # Subscriber base_url = public-looking host so url_safety
        # doesn't reject in its vanilla form; the subscriber-origin
        # trust path is covered by a separate test.
        real_requests.put(base + "/api/subscribe/stt-sub",
                          data=json.dumps({
                              "base_url": "https://crm.example.com/crm",
                              "bearer_token": "stt-tok",
                          }), headers=acct)

        call_id = real_requests.post(
            base + "/api/calls",
            data=json.dumps({"subscriber_id": "stt-sub"}),
            headers=acct,
        ).json()["call_id"]

        try:
            # CRM-side Participant already exists with deterministic id.
            participant_id = 4711

            payload = {
                "callback":
                    f"/Telephone/SpeechServer/public?call=77"
                    f"&participant={participant_id}&state=webclient",
                "base_url": "https://crm.example.com",
                "call_id": call_id,
                "stt_callback":
                    f"/Telephone/SpeechServer/sttNote?call=77"
                    f"&participant={participant_id}",
            }
            resp = real_requests.post(
                base + "/api/pipelines",
                data=json.dumps({
                    "dsl": 'webclient:stt_user' + json.dumps(payload),
                }),
                headers=acct,
            )
            assert resp.status_code == 201, resp.text
            body = resp.json()
            session_id = body["session_id"]

            # Connect the "browser" and push real speech.
            ws = websocket.create_connection(
                live_server["ws_base"] + "/ws/socket/" + session_id,
                timeout=5,
            )
            try:
                ws.send(json.dumps({"type": "hello", "profiles": ["low"]}))
                ws.settimeout(5)
                hello = json.loads(ws.recv())
                assert hello["type"] == "hello", hello
                profile = hello["profile"]

                from speech_pipeline import fourier_codec as fc
                import numpy as np

                pcm = _decode_mp3_to_s16le(fc.SAMPLE_RATE, duration_s=3.0)
                samples = (np.frombuffer(pcm, dtype=np.int16)
                           .astype(np.float32) / 32768.0)
                n_frames = len(samples) // fc.FRAME_SAMPLES
                for i in range(n_frames):
                    frame = samples[i * fc.FRAME_SAMPLES
                                    : (i + 1) * fc.FRAME_SAMPLES]
                    ws.send_binary(fc.encode_frame(frame, profile))
                    time.sleep(0.005)

                # Wait up to 5s for the sttNote POST to land.
                deadline = time.monotonic() + 5.0
                hits = []
                while time.monotonic() < deadline:
                    with capture_lock:
                        hits = [c for c in captured
                                if "/sttNote" in (c["url"] or "")]
                    if hits:
                        break
                    time.sleep(0.1)

                assert hits, (
                    "STT webhook never reached the CRM.  Captured POSTs: "
                    f"{[c['url'] for c in captured]}"
                )
                hit = hits[0]
                assert f"call=77" in hit["url"], (
                    f"sttNote URL missing call id: {hit['url']}"
                )
                assert f"participant={participant_id}" in hit["url"], (
                    f"sttNote URL missing participant id: {hit['url']}"
                )
                body_text = hit["data"] or ""
                if isinstance(body_text, bytes):
                    body_text = body_text.decode("utf-8", errors="replace")
                try:
                    obj = json.loads(body_text)
                except Exception:
                    raise AssertionError(
                        f"sttNote body is not JSON: {body_text[:120]!r}")
                assert "text" in obj, (
                    f"sttNote body has no 'text' field: {obj}"
                )
                assert obj["text"].startswith("segment-"), (
                    f"transcript content unexpected: {obj}"
                )
                # Auth header carries the subscriber's bearer token.
                auth = hit["headers"].get("Authorization", "")
                assert auth.startswith("Bearer "), (
                    f"sttNote POST missing Bearer auth: {auth!r}"
                )
            finally:
                ws.close()
        finally:
            real_requests.delete(base + f"/api/calls/{call_id}", headers=acct)
