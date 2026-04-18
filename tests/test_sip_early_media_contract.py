"""Early-media contract tests for CRM-controlled inbound SIP flows.

The intended contract is:

1. An inbound trunk INVITE gets ``183 Session Progress`` immediately.
2. The CRM may already wire the inbound leg into a conference and play
   wait music during early media.
3. ``200 OK`` must not be sent until the CRM explicitly answers the
   inbound leg, typically only after a second participant joined.

These tests lock that behavior in before changing runtime code.
"""
from __future__ import annotations

import json
import time
from typing import Callable

import pytest

from conftest import SUBSCRIBER_ID
from fake_crm import FakeCrm
from speech_pipeline.RTPSession import RTPSession
from speech_pipeline.rtp_codec import PCMU
from speech_pipeline.telephony import call_state, leg as leg_mod, sip_stack
from test_crm_e2e import _find_free_port, _receive_audio

PBX_ID = "TestPBX"


def _trunk_invite_message(
    *,
    caller: str = "+49174000",
    callee: str = "+493586",
    call_id: str = "early-media-test",
    remote_host: str = "127.0.0.1",
    remote_port: int,
) -> dict:
    body = "\r\n".join([
        "v=0",
        "o=trunk 1 1 IN IP4 127.0.0.1",
        "s=-",
        f"c=IN IP4 {remote_host}",
        "t=0 0",
        f"m=audio {remote_port} RTP/AVP 0 101",
        "a=rtpmap:0 PCMU/8000",
        "a=rtpmap:101 telephone-event/8000",
        "a=fmtp:101 0-15",
        "",
    ])
    lines = [
        f"INVITE sip:{callee}@srv.launix.de SIP/2.0",
        "Via: SIP/2.0/UDP 127.0.0.1:5060;branch=z9hG4bK-early",
        "Max-Forwards: 70",
        f"From: <sip:{caller}@trunk.example>;tag=from-1",
        f"To: <sip:{callee}@srv.launix.de>",
        f"Call-ID: {call_id}",
        "CSeq: 1 INVITE",
        f"Contact: <sip:{caller}@127.0.0.1:5060>",
        "Content-Type: application/sdp",
        f"Content-Length: {len(body.encode('utf-8'))}",
        "",
        body,
    ]
    return sip_stack._parse_sip("\r\n".join(lines).encode("utf-8"))


def _wait_until(
    predicate: Callable[[], bool],
    *,
    timeout: float = 3.0,
    interval: float = 0.02,
    failure: str,
) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise AssertionError(failure)


@pytest.fixture(autouse=True)
def _clear_sip_state():
    sip_stack._trunk_dialogs.clear()
    yield
    sip_stack._trunk_dialogs.clear()


@pytest.fixture
def crm(client, admin, account):
    token = account["Authorization"].split(None, 1)[1]
    crm = FakeCrm(client, admin_headers=admin, account_token=token)
    crm.wait_jingle = "examples/queue.mp3"
    crm.register_as_subscriber(SUBSCRIBER_ID, PBX_ID)
    return crm


def _call_sid(crm: FakeCrm) -> str:
    assert crm.calls, "CRM did not create a call row"
    return next(iter(crm.calls.values()))["sid"]


def _inbound_leg_id_for_call(call_sid: str) -> str:
    call = call_state.get_call(call_sid)
    assert call is not None
    inbound = [
        p for p in call.list_participants()
        if p.get("type") == "sip" and p.get("direction") == "inbound"
    ]
    assert len(inbound) == 1
    return inbound[0]["id"]


class TestCrmControlledEarlyMedia:
    def test_wait_music_flows_during_early_media_before_answer(
        self, client, account, crm, monkeypatch
    ):
        remote_port = _find_free_port()
        phone_rtp = RTPSession(
            remote_port,
            "127.0.0.1",
            9,
            codec=PCMU.new_session_codec(),
        )
        phone_rtp.start()
        sent_messages: list[str] = []
        call_id = f"early-media-{remote_port}"
        msg = _trunk_invite_message(call_id=call_id, remote_port=remote_port)

        monkeypatch.setattr(
            sip_stack,
            "_send",
            lambda data, addr: sent_messages.append(data),
        )

        try:
            with crm.active(monkeypatch):
                sip_stack._handle_trunk_invite(msg, ("127.0.0.1", 5060), PBX_ID)

                _wait_until(
                    lambda: bool(crm.calls) and bool(_call_sid(crm)),
                    failure="CRM did not bootstrap the inbound call",
                )

                call_sid = _call_sid(crm)
                wait_stage = f"play:{call_sid}_wait"
                _wait_until(
                    lambda: client.get(
                        f"/api/pipelines?dsl={wait_stage}",
                        headers=account,
                    ).status_code == 200,
                    failure="CRM did not start wait music during early media",
                )

                audio = _receive_audio(phone_rtp, duration_s=1.0)

            assert any("SIP/2.0 183 Session Progress" in msg for msg in sent_messages)
            assert not any("SIP/2.0 200 OK" in msg for msg in sent_messages), (
                "inbound call was answered before the CRM admitted a second participant"
            )
            assert call_id in sip_stack._trunk_dialogs
            assert sip_stack._trunk_dialogs[call_id]["answered"] is False
            assert len(audio) > 0, "caller received no early-media audio"

            samples = memoryview(audio).cast("h")
            rms = (sum(int(s) * int(s) for s in samples) / max(len(samples), 1)) ** 0.5
            assert rms > 150, f"wait music too quiet during early media (RMS={rms:.0f})"
        finally:
            if crm.calls:
                client.delete(f"/api/pipelines?dsl=call:{_call_sid(crm)}", headers=account)
            phone_rtp.stop()
            sip_stack._trunk_dialogs.pop(call_id, None)

    def test_second_participant_answer_releases_caller_with_200_ok(
        self, client, account, crm, monkeypatch
    ):
        crm.internal_phones = [{"number": "sip:admin@crm.example", "answerer_id": 1}]
        remote_port = _find_free_port()
        phone_rtp = RTPSession(
            remote_port,
            "127.0.0.1",
            9,
            codec=PCMU.new_session_codec(),
        )
        phone_rtp.start()
        sent_messages: list[str] = []
        call_id = f"answer-gate-{remote_port}"
        msg = _trunk_invite_message(call_id=call_id, remote_port=remote_port)

        monkeypatch.setattr(
            sip_stack,
            "_send",
            lambda data, addr: sent_messages.append(data),
        )
        monkeypatch.setattr(leg_mod, "originate_only", lambda leg, pbx: None)

        try:
            with crm.active(monkeypatch):
                sip_stack._handle_trunk_invite(msg, ("127.0.0.1", 5060), PBX_ID)

                _wait_until(
                    lambda: bool(crm.calls),
                    failure="CRM did not create the inbound call",
                )
                call_db_id = next(iter(crm.calls.keys()))
                call_sid = _call_sid(crm)
                inbound_leg_id = _inbound_leg_id_for_call(call_sid)
                wait_stage = f"play:{call_sid}_wait"

                _wait_until(
                    lambda: client.get(
                        f"/api/pipelines?dsl={wait_stage}",
                        headers=account,
                    ).status_code == 200,
                    failure="wait music stage was not present before the second participant answered",
                )

                internal = [
                    (pid, p) for pid, p in crm.participants.items()
                    if p.get("call_db_id") == call_db_id
                    and p.get("number") == "sip:admin@crm.example"
                    and p.get("sid")
                ]
                assert len(internal) == 1, (
                    "CRM did not create exactly one internal participant leg"
                )
                pid, participant = internal[0]
                assert not any("SIP/2.0 200 OK" in msg for msg in sent_messages), (
                    "caller was answered before the internal participant joined"
                )

                crm._route(
                    crm.BASE_URL + "/Telephone/SpeechServer/public"
                    f"?state=leg&event=answered&call={call_db_id}&participant={pid}",
                    {"leg_id": participant["sid"]},
                    crm.account_token,
                    method="POST",
                )

                _wait_until(
                    lambda: any("SIP/2.0 200 OK" in msg for msg in sent_messages),
                    failure="answering the second participant did not release the caller with 200 OK",
                )
                _wait_until(
                    lambda: client.get(
                        f"/api/pipelines?dsl={wait_stage}",
                        headers=account,
                    ).status_code == 404,
                    failure="wait music was not removed after the second participant answered",
                )

            assert sip_stack._trunk_dialogs[call_id]["answered"] is True
            call = call_state.get_call(call_sid)
            assert call is not None
            inbound = next(
                p for p in call.list_participants()
                if p["id"] == inbound_leg_id
            )
            assert inbound["direction"] == "inbound"
        finally:
            if crm.calls:
                client.delete(f"/api/pipelines?dsl=call:{_call_sid(crm)}", headers=account)
            phone_rtp.stop()
            sip_stack._trunk_dialogs.pop(call_id, None)

    def test_retransmitted_invite_stays_183_until_crm_answers(
        self, client, account, crm, monkeypatch
    ):
        crm.internal_phones = [{"number": "sip:admin@crm.example", "answerer_id": 1}]
        remote_port = _find_free_port()
        phone_rtp = RTPSession(
            remote_port,
            "127.0.0.1",
            9,
            codec=PCMU.new_session_codec(),
        )
        phone_rtp.start()
        sent_messages: list[str] = []
        call_id = f"reinvite-{remote_port}"
        msg = _trunk_invite_message(call_id=call_id, remote_port=remote_port)

        monkeypatch.setattr(
            sip_stack,
            "_send",
            lambda data, addr: sent_messages.append(data),
        )
        monkeypatch.setattr(leg_mod, "originate_only", lambda leg, pbx: None)

        try:
            with crm.active(monkeypatch):
                sip_stack._handle_trunk_invite(msg, ("127.0.0.1", 5060), PBX_ID)
                _wait_until(lambda: bool(crm.calls), failure="CRM did not create the call")

                sip_stack._handle_trunk_invite(msg, ("127.0.0.1", 5060), PBX_ID)
                assert "SIP/2.0 183 Session Progress" in sent_messages[-1], (
                    "retransmitted INVITE before answer must get 183, not a final answer"
                )

                call_db_id = next(iter(crm.calls.keys()))
                internal = [
                    (pid, p) for pid, p in crm.participants.items()
                    if p.get("call_db_id") == call_db_id
                    and p.get("number") == "sip:admin@crm.example"
                    and p.get("sid")
                ]
                assert len(internal) == 1
                pid, participant = internal[0]
                crm._route(
                    crm.BASE_URL + "/Telephone/SpeechServer/public"
                    f"?state=leg&event=answered&call={call_db_id}&participant={pid}",
                    {"leg_id": participant["sid"]},
                    crm.account_token,
                    method="POST",
                )
                _wait_until(
                    lambda: sip_stack._trunk_dialogs[call_id]["answered"] is True,
                    failure="second participant answer did not mark the trunk dialog answered",
                )

                sip_stack._handle_trunk_invite(msg, ("127.0.0.1", 5060), PBX_ID)
                assert "SIP/2.0 200 OK" in sent_messages[-1], (
                    "retransmitted INVITE after CRM answer must get 200 OK"
                )
        finally:
            if crm.calls:
                client.delete(f"/api/pipelines?dsl=call:{_call_sid(crm)}", headers=account)
            phone_rtp.stop()
            sip_stack._trunk_dialogs.pop(call_id, None)
