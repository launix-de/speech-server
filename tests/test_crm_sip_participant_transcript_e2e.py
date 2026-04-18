"""End-to-end: SIP participant audio reaches CRM transcript webhook."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import pytest

from conftest import ACCOUNT_ID, ACCOUNT_TOKEN, SUBSCRIBER_ID
from fake_crm import FakeCrm
from speech_pipeline.base import AudioFormat, Stage
from test_crm_e2e import _cleanup_leg, _make_rtp_leg, _send_audio

STT_TEST_MP3 = Path(__file__).parent / "fixtures" / "stt-test.mp3"


def _make_fake_transcriber():
    class _FakeTranscriber(Stage):
        def __init__(self):
            super().__init__()
            self.input_format = AudioFormat(16000, "s16le")
            self.output_format = AudioFormat(0, "ndjson")

        def stream_pcm24k(self):
            if not self.upstream:
                return
            buffered = 0
            emitted = 0
            for chunk in self.upstream.stream_pcm24k():
                if self.cancelled:
                    break
                if not chunk:
                    continue
                buffered += len(chunk)
                while buffered >= 32000:
                    buffered -= 32000
                    emitted += 1
                    yield (json.dumps({
                        "text": f"segment-{emitted}",
                        "start": emitted - 1,
                        "end": emitted,
                    }) + "\n").encode()

    return _FakeTranscriber()


def _decode_mp3_to_s16le(sample_rate: int, duration_s: float = 2.0) -> bytes:
    result = subprocess.run(
        ["ffmpeg", "-i", str(STT_TEST_MP3), "-f", "s16le", "-ac", "1",
         "-ar", str(sample_rate), "-t", str(duration_s), "-"],
        capture_output=True,
    )
    assert result.returncode == 0 and result.stdout, (
        f"ffmpeg decode failed: {result.stderr.decode()[:200]}"
    )
    return result.stdout


@pytest.fixture
def crm(client, admin):
    acct = {
        "Authorization": f"Bearer {ACCOUNT_TOKEN}",
        "Content-Type": "application/json",
    }
    client.put(
        "/api/pbx/TestPBX",
        data=json.dumps({"sip_proxy": "", "sip_user": "", "sip_password": ""}),
        headers=admin,
    )
    client.put(
        f"/api/accounts/{ACCOUNT_ID}",
        data=json.dumps({"token": ACCOUNT_TOKEN, "pbx": "TestPBX"}),
        headers=admin,
    )
    c = FakeCrm(client, admin_headers=admin, account_token=ACCOUNT_TOKEN)
    c.register_as_subscriber(SUBSCRIBER_ID, "TestPBX")
    yield c


class TestSipParticipantTranscriptE2E:
    def test_sip_participant_audio_reaches_crm_transcript(
            self, client, account, crm, monkeypatch):
        from speech_pipeline.telephony import call_state
        from speech_pipeline.telephony import pipe_executor as pe

        original_create_stage = pe.CallPipeExecutor._create_stage

        def _wrapped(self, typ, elem_id, params):
            if typ == "stt":
                return _make_fake_transcriber()
            return original_create_stage(self, typ, elem_id, params)

        monkeypatch.setattr(pe.CallPipeExecutor, "_create_stage", _wrapped)

        leg, phone_rtp, _session = _make_rtp_leg(number="sip:carli@launix.de/crm")
        call_sid = ""
        call_db_id = 3301
        participant_id = 9901

        try:
            with crm.active(monkeypatch):
                resp = client.post(
                    "/api/calls",
                    data=json.dumps({"subscriber_id": SUBSCRIBER_ID}),
                    headers=account,
                )
                assert resp.status_code == 201, resp.data
                call_sid = resp.get_json()["call_id"]

                crm.calls[call_db_id] = {
                    "caller": "+491700001",
                    "direction": "inbound",
                    "status": "answered",
                    "sid": call_sid,
                }
                crm.participants[participant_id] = {
                    "call_db_id": call_db_id,
                    "sid": leg.leg_id,
                    "status": "answered",
                    "number": "sip:carli@launix.de/crm",
                }

                tap_id = f"{leg.leg_id}_tap"
                bridge = (
                    f"sip:{leg.leg_id} -> tee:{tap_id} "
                    f"-> call:{call_sid} -> sip:{leg.leg_id}"
                )
                stt = (
                    "tee:" + tap_id
                    + " -> stt:de -> webhook:"
                    + crm.BASE_URL
                    + f"/Telephone/SpeechServer/sttNote?call={call_db_id}"
                    + f"&participant={participant_id}"
                )
                for dsl in (bridge, stt):
                    resp = client.post(
                        "/api/pipelines",
                        data=json.dumps({"dsl": dsl}),
                        headers=account,
                    )
                    assert resp.status_code == 201, resp.data

                ref_pcm = _decode_mp3_to_s16le(phone_rtp.codec.sample_rate, 2.0)
                _send_audio(phone_rtp, ref_pcm)

                deadline = time.monotonic() + 5.0
                transcript = ""
                while time.monotonic() < deadline:
                    transcript = crm.calls[call_db_id].get("transcript", "")
                    if "segment-1" in transcript:
                        break
                    time.sleep(0.1)

                assert "segment-1" in transcript, (
                    f"SIP participant STT never reached CRM transcript: {transcript!r}"
                )
                assert "carli@launix.de:</strong>" in transcript, (
                    f"participant speaker label missing or malformed: {transcript!r}"
                )
        finally:
            _cleanup_leg(leg, phone_rtp)
            if call_sid and call_state.get_call(call_sid) is not None:
                client.delete(f"/api/pipelines?dsl=call:{call_sid}",
                              headers=account)
