"""CRM contract + feature coverage (audit gaps #11-15).

#11 heartbeat subscriber registration contract.
#12 state=tts-done ends the call after announcement.
#13 sttNote transcript append per speaker.
#14 hold without jingle → silent hold, no play pipe posted.
#15 Outbound + webclient combo via client:USER in findParticipant.
"""
from __future__ import annotations

import json
import time
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
)


@pytest.fixture
def crm(client, admin):
    acct = {"Authorization": f"Bearer {ACCOUNT_TOKEN}",
            "Content-Type": "application/json"}
    client.put("/api/pbx/TestPBX",
               data=json.dumps({"sip_proxy": "", "sip_user": "",
                                "sip_password": ""}), headers=admin)
    client.put(f"/api/accounts/{ACCOUNT_ID}",
               data=json.dumps({"token": ACCOUNT_TOKEN, "pbx": "TestPBX",
                                 "features": ["webclient"]}),
               headers=admin)
    c = FakeCrm(client, admin_headers=admin, account_token=ACCOUNT_TOKEN)
    c.register_as_subscriber(SUBSCRIBER_ID, "TestPBX")
    yield c


# ---------------------------------------------------------------------------
# #11. heartbeat contract
# ---------------------------------------------------------------------------

class TestHeartbeatContract:
    """Mirror what heartbeat.fop registers: subscriber gets base_url,
    bearer, and all four event keys the server fires."""

    def test_required_event_keys_are_accepted(self, client, admin):
        acct = {"Authorization": f"Bearer {ACCOUNT_TOKEN}",
                "Content-Type": "application/json"}
        client.put("/api/pbx/HbPBX",
                   data=json.dumps({"sip_proxy": "", "sip_user": "",
                                     "sip_password": ""}), headers=admin)
        client.put(f"/api/accounts/{ACCOUNT_ID}",
                   data=json.dumps({"token": ACCOUNT_TOKEN, "pbx": "HbPBX"}),
                   headers=admin)
        resp = client.put(f"/api/subscribe/{SUBSCRIBER_ID}",
                          data=json.dumps({
                              "base_url": "https://crm.example.com/crm",
                              "bearer_token": "t",
                              "events": {
                                  "incoming": "POST /p?state=incoming",
                                  "call_ended": "POST /p?state=ended",
                                  "dtmf": "POST /p?state=dtmf",
                                  "device_dial": "POST /p?state=device-dial",
                              },
                          }), headers=acct)
        assert resp.status_code in (200, 201), resp.get_data(as_text=True)
        sub = sub_mod.get(SUBSCRIBER_ID)
        # All four event keys survive the round-trip.
        for key in ("incoming", "call_ended", "dtmf", "device_dial"):
            assert key in sub["events"], (
                f"event key {key!r} missing after subscribe: "
                f"{list(sub['events'].keys())}"
            )


# ---------------------------------------------------------------------------
# #12. tts-done ends the call
# ---------------------------------------------------------------------------

def _on_tts_done(self, qs, body):
    call_db_id = int(qs.get("call", [0])[0])
    if call_db_id in self.calls:
        self.calls[call_db_id]["status"] = "denied"
        call_sid = self.calls[call_db_id].get("sid")
        if call_sid:
            self.client.delete(f"/api/pipelines?dsl=call:{call_sid}",
                               headers=self.account_headers)


# Attach these extras to FakeCrm at test time (keep core FakeCrm minimal).
def _extend_fake_crm(crm):
    import types
    crm._on_tts_done = types.MethodType(_on_tts_done, crm)


class TestTtsDoneEndsCall:
    """Closed-hours flow: inbound call → TTS announcement → tts-done →
    CRM ends the call with status=denied."""

    def test_tts_done_webhook_triggers_endCall(
            self, client, account, crm, monkeypatch):
        _extend_fake_crm(crm)
        with crm.active(monkeypatch):
            resp = client.post("/api/calls",
                               data=json.dumps({"subscriber_id": SUBSCRIBER_ID}),
                               headers=account)
            call_sid = resp.get_json()["call_id"]
            call_db_id = 12
            crm.calls[call_db_id] = {
                "caller": "+49170", "direction": "inbound",
                "status": "ringing", "sid": call_sid,
            }
            # Route a tts-done webhook exactly as the TTS pipe's
            # completed callback would fire.
            crm._route(
                crm.BASE_URL + "/Telephone/SpeechServer/public"
                f"?state=tts-done&call={call_db_id}",
                {},
                crm.account_token, method="POST",
            )
        assert crm.calls[call_db_id]["status"] == "denied", (
            f"tts-done did not flip status: {crm.calls[call_db_id]}"
        )
        assert call_state.get_call(f"{ACCOUNT_ID}:{call_sid}") is None, (
            "server-side call was not torn down after CRM saw tts-done"
        )


# ---------------------------------------------------------------------------
# #13. sttNote appends to transcript
# ---------------------------------------------------------------------------

class TestSttNoteTranscriptAppend:

    def test_two_segments_accumulate_under_speaker_names(
            self, client, account, crm, monkeypatch):
        _extend_fake_crm(crm)
        with crm.active(monkeypatch):
            call_db_id = 13
            crm.calls[call_db_id] = {
                "caller": "+49170", "direction": "inbound",
                "status": "answered", "sid": "",
            }
            crm.participants[42] = {
                "call_db_id": call_db_id,
                "number": "+49170", "answerer_name": "Alice",
                "status": "answered", "sid": "leg-a",
            }
            # Two transcript segments from the same speaker.
            for segment in ("Hallo Welt.", "Wie geht es?"):
                crm._route(
                    crm.BASE_URL + "/Telephone/SpeechServer/sttNote"
                    f"?call={call_db_id}&participant=42",
                    {"text": segment}, crm.account_token, method="POST",
                )
        transcript = crm.calls[call_db_id].get("transcript", "")
        assert "Alice:</strong> <span>Hallo Welt." in transcript, (
            f"first segment missing: {transcript!r}"
        )
        assert "Alice:</strong> <span>Wie geht es?" in transcript, (
            f"second segment missing: {transcript!r}"
        )
        assert transcript.count("Alice:</strong>") == 2

    def test_participant_zero_uses_caller_fallback(
            self, client, account, crm, monkeypatch):
        _extend_fake_crm(crm)
        with crm.active(monkeypatch):
            call_db_id = 14
            crm.calls[call_db_id] = {
                "caller": "+49171", "direction": "inbound",
                "status": "answered", "sid": "",
            }
            crm._route(
                crm.BASE_URL + "/Telephone/SpeechServer/sttNote"
                f"?call={call_db_id}&participant=0",
                {"text": "Fremder Text."},
                crm.account_token, method="POST",
            )
        assert "+49171:</strong>" in crm.calls[call_db_id]["transcript"], (
            f"caller fallback missing when participant=0: "
            f"{crm.calls[call_db_id]['transcript']!r}"
        )

    def test_participant_sip_uri_is_normalized_like_crm(
            self, client, account, crm, monkeypatch):
        _extend_fake_crm(crm)
        with crm.active(monkeypatch):
            call_db_id = 15
            crm.calls[call_db_id] = {
                "caller": "+49172", "direction": "inbound",
                "status": "answered", "sid": "",
            }
            crm.participants[43] = {
                "call_db_id": call_db_id,
                "number": "sip:carli@launix.de/crm",
                "status": "answered", "sid": "leg-b",
            }
            crm._route(
                crm.BASE_URL + "/Telephone/SpeechServer/sttNote"
                f"?call={call_db_id}&participant=43",
                {"text": "Guten Tag."},
                crm.account_token, method="POST",
            )
        transcript = crm.calls[call_db_id]["transcript"]
        assert "carli@launix.de:</strong>" in transcript, (
            f"SIP participant speaker normalization diverged from transcript.fop: "
            f"{transcript!r}"
        )


# ---------------------------------------------------------------------------
# #14. Hold without jingle
# ---------------------------------------------------------------------------

class TestHoldWithoutJingle:

    def test_silent_hold_emits_bridge_delete_but_no_play_post(
            self, client, account, crm, monkeypatch):
        with crm.active(monkeypatch):
            resp = client.post("/api/calls",
                               data=json.dumps({"subscriber_id": SUBSCRIBER_ID}),
                               headers=account)
            call_sid = resp.get_json()["call_id"]
            call_db_id = 99
            crm.calls[call_db_id] = {
                "caller": "", "direction": "outbound",
                "status": "answered", "sid": call_sid,
            }
            crm.participants[42] = {
                "call_db_id": call_db_id, "status": "answered",
                "number": "+49", "sid": "leg-42",
            }
            crm.hold_jingle = ""  # No jingle configured

            posted_dsl: list[str] = []
            orig_post = client.post

            def _spy_post(path, *args, **kwargs):
                if path == "/api/pipelines":
                    try:
                        posted_dsl.append(
                            json.loads(kwargs.get("data") or "{}").get("dsl", "")
                        )
                    except Exception:
                        pass
                return orig_post(path, *args, **kwargs)
            monkeypatch.setattr(client, "post", _spy_post)

            crm.hold_external_legs(call_db_id, 42, "leg-42")

        # Bridge was dropped, but no play: pipe was posted.
        play_posts = [d for d in posted_dsl if d.startswith("play:")]
        assert not play_posts, (
            f"silent hold still posted hold-music pipe: {play_posts}"
        )


# ---------------------------------------------------------------------------
# #15. Outbound + webclient combo (client:USER via findParticipant)
# ---------------------------------------------------------------------------

class TestOutboundWebclientCombo:
    """CRM dials ``client:userA`` via findParticipant: the webclient:
    DSL action must fire, not originate."""

    def test_client_prefix_triggers_webclient_slot(
            self, client, account, crm, monkeypatch):
        with crm.active(monkeypatch):
            resp = client.post("/api/calls",
                               data=json.dumps({"subscriber_id": SUBSCRIBER_ID}),
                               headers=account)
            call_sid = resp.get_json()["call_id"]
            payload = {
                "callback": f"/cb?call=1&participant=77&state=webclient",
                "base_url": "https://crm.example.com",
                "call_id": call_sid,
            }
            dsl = 'webclient:userA' + json.dumps(payload)
            resp = client.post("/api/pipelines",
                               data=json.dumps({"dsl": dsl}),
                               headers=account)
            assert resp.status_code == 201, resp.get_data(as_text=True)
            body = resp.get_json()
            assert body.get("session_id", "").startswith("wc-"), (
                f"webclient slot not created on outbound call: {body}"
            )
            assert body.get("iframe_url"), (
                f"no iframe_url in response: {body}"
            )
            client.delete(f"/api/calls/{call_sid}", headers=account)
