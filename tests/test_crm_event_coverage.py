"""Coverage for CRM webhook event branches the existing suite misses.

Gap audit findings:
- Leg events ``failed`` / ``busy`` / ``no-answer`` / ``canceled`` — CRM
  handler calls ``endCall()`` + ``checkLiveliness()`` but no test fires
  them and verifies the state transition.
- Server → CRM reverse webhook ``state=ended`` (fired by sip_listener
  when an inbound call ends) — entirely untested.
- ``checkLiveliness`` auto-unhold (last non-hold participant leaves while
  another is on hold → hold must lift) — untested race path.
"""
from __future__ import annotations

import json
import time
from unittest.mock import MagicMock

import pytest

from conftest import ADMIN_TOKEN, ACCOUNT_TOKEN, ACCOUNT_ID, SUBSCRIBER_ID
from fake_crm import FakeCrm
from speech_pipeline.telephony import (
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
               data=json.dumps({"token": ACCOUNT_TOKEN, "pbx": "TestPBX"}),
               headers=admin)
    c = FakeCrm(client, admin_headers=admin, account_token=ACCOUNT_TOKEN)
    c.register_as_subscriber(SUBSCRIBER_ID, "TestPBX")
    # Register an internal phone so findParticipant has someone to dial.
    c.internal_phones = [{"number": "+49301", "answerer_id": 1}]
    yield c


def _fake_leg(call_sid: str, number: str, cb_base: str) -> leg_mod.Leg:
    """Build a Leg in ``originate`` state with standard callbacks."""
    fake_voip = MagicMock()
    fake_voip.RTPClients = []
    leg = leg_mod.create_leg(
        direction="outbound", number=number, pbx_id="TestPBX",
        subscriber_id=SUBSCRIBER_ID, voip_call=fake_voip,
    )
    leg.call_id = call_sid
    leg.callbacks = {
        "ringing":   cb_base + "&state=leg&event=ringing",
        "answered":  cb_base + "&state=leg&event=answered",
        "completed": cb_base + "&state=leg&event=completed",
        "failed":    cb_base + "&state=leg&event=failed",
        "no-answer": cb_base + "&state=leg&event=no-answer",
        "busy":      cb_base + "&state=leg&event=busy",
        "canceled":  cb_base + "&state=leg&event=canceled",
    }
    return leg


class TestLegFailureEvents:
    """Originate fails / busy / no-answer / canceled paths all share the
    ``_on_leg_ended`` CRM handler — verify each transitions the row
    correctly and clears the call when no one is left."""

    @pytest.mark.parametrize("event", ["failed", "busy",
                                        "no-answer", "canceled"])
    def test_originate_event_marks_participant_and_ends_call(
            self, client, account, crm, monkeypatch, event):
        with crm.active(monkeypatch):
            # Build a server-side call + CRM participant mirroring
            # what findParticipant would produce.
            resp = client.post("/api/calls",
                               data=json.dumps({"subscriber_id": SUBSCRIBER_ID}),
                               headers=account)
            call_sid = resp.get_json()["call_id"]
            call_db_id = 55
            pid = 777
            crm.calls[call_db_id] = {
                "caller": "", "direction": "outbound", "status": "ringing",
                "sid": call_sid,
            }
            crm.participants[pid] = {
                "call_db_id": call_db_id, "status": "adding",
                "number": "+49301", "sid": "",
            }
            cb_base = (f"/Telephone/SpeechServer/public?call={call_db_id}"
                        f"&participant={pid}")
            leg = _fake_leg(call_sid, "+49301", cb_base)

            # Server fires the lifecycle callback (as originate would).
            leg_mod.fire_callback(leg, event, error="simulated")
            time.sleep(0.3)

            # (a) CRM flipped the participant to the right reason.
            assert crm.participants[pid]["status"] == event, (
                f"CRM did not apply event={event!r}: "
                f"{crm.participants[pid]}"
            )
            assert crm.participants[pid].get("end_reason") == event

            # (b) No active participant remains → call is completed.
            assert crm.calls[call_db_id]["status"] == "completed", (
                f"CRM did not end the call after sole participant "
                f"{event!r}: {crm.calls[call_db_id]}"
            )
            # (c) Server-side call was torn down.
            assert call_state.get_call(call_sid) is None

            # (d) The webhook audit trail shows exactly the event we fired.
            events_seen = [w["query"].get("event")
                           for w in crm.webhooks
                           if w["state"] == "leg"]
            assert event in events_seen, (
                f"expected {event!r} in webhook trail, got {events_seen}"
            )

            leg_mod._legs.pop(leg.leg_id, None)


class TestReverseEndedWebhook:
    """When an inbound SIP call ends via BYE, ``sip_listener`` fires
    ``dispatcher.fire_event(call, 'call_ended', ...)`` which reaches the
    CRM as ``state=ended``.  The handler must mark the call completed."""

    def test_server_initiated_ended_marks_call_completed(
            self, client, account, crm, monkeypatch):
        with crm.active(monkeypatch):
            # Bootstrap an inbound call via the normal flow so the CRM
            # has a row that matches ``callId``.
            leg = leg_mod.create_leg(
                direction="inbound", number="+49170111", pbx_id="TestPBX",
                subscriber_id=SUBSCRIBER_ID, voip_call=MagicMock(RTPClients=[]),
            )
            sub = sub_mod.get(SUBSCRIBER_ID)
            dispatcher.fire_subscriber_event(sub, "incoming", {
                "caller": "+49170111", "callee": "+4935000",
                "leg_id": leg.leg_id,
            })
            time.sleep(0.2)
            call_db_id = next(iter(crm.calls.keys()))
            call_sid = crm.calls[call_db_id]["sid"]

            # Now the server emits call_ended for that call.
            call = call_state.get_call(call_sid)
            dispatcher.fire_event(call, "call_ended",
                                  {"callId": call.call_id})
            time.sleep(0.2)

            # CRM marks call completed purely from the reverse webhook.
            assert crm.calls[call_db_id]["status"] == "completed", (
                f"CRM did not process state=ended webhook: "
                f"{crm.calls[call_db_id]}"
            )
            assert any(w["state"] == "ended" for w in crm.webhooks), (
                f"state=ended never landed in CRM: "
                f"{[w['state'] for w in crm.webhooks]}"
            )

            leg_mod._legs.pop(leg.leg_id, None)


class TestCheckLivelinessAutoUnhold:
    """Scenario: A holds B (parking); then A hangs up.  Only B remains,
    still on hold → CRM must unhold B so audio resumes.

    Mirrors ``calls.fop::checkLiveliness`` ``speech-server-hold`` hook
    (triggered from endCall → checkLiveliness → if leftover == 1 and
    call.status in {'hold','hold-adding'} then status=answered +
    unholdExternalLegs)."""

    def test_last_nonhold_hangup_triggers_unhold(
            self, client, account, crm, monkeypatch):
        """If A hangs up while B is on hold, FakeCrm should revert call
        status and POST the unhold-bridge DSL for B."""
        with crm.active(monkeypatch):
            resp = client.post("/api/calls",
                               data=json.dumps({"subscriber_id": SUBSCRIBER_ID}),
                               headers=account)
            call_sid = resp.get_json()["call_id"]
            call_db_id = 66
            pid_a, pid_b = 901, 902
            crm.calls[call_db_id] = {
                "caller": "", "direction": "outbound",
                "status": "hold", "sid": call_sid,
            }
            crm.participants[pid_a] = {
                "call_db_id": call_db_id, "status": "answered",
                "number": "+491", "sid": "leg-a-sip",
            }
            crm.participants[pid_b] = {
                "call_db_id": call_db_id, "status": "hold",
                "number": "+492", "sid": "leg-b-sip",
            }

            unhold_calls: list[tuple] = []
            orig_unhold = crm.unhold_external_legs

            def _spy_unhold(cdb, pid, lid):
                unhold_calls.append((cdb, pid, lid))
                return orig_unhold(cdb, pid, lid)
            monkeypatch.setattr(crm, "unhold_external_legs", _spy_unhold)

            cb_base = (f"/Telephone/SpeechServer/public?call={call_db_id}"
                        f"&participant={pid_a}")
            leg_a = _fake_leg(call_sid, "+491", cb_base)
            leg_mod.fire_callback(leg_a, "completed", duration=5.0)
            time.sleep(0.3)

            # checkLiveliness in CRM: since the call is on hold and only
            # 1 participant (B) is left, it flips status and unholds B.
            assert crm.calls[call_db_id]["status"] == "answered", (
                f"call still in 'hold' state after last non-hold "
                f"participant left: {crm.calls[call_db_id]}"
            )
            assert unhold_calls, (
                "CRM did not call unholdExternalLegs for the remaining "
                "participant on hold — B stays muted forever"
            )
            assert unhold_calls[0] == (call_db_id, pid_b, "leg-b-sip")

            leg_mod._legs.pop(leg_a.leg_id, None)
            client.delete(f"/api/calls/{call_sid}", headers=account)
