"""End-to-end CRM flow tests using the ``FakeCrm`` webhook receiver.

Reproduces the exact HTTP+webhook sequence from
``backends/businesslogic/telefonanlage/speech-server/*.fop`` against
the live server, so every integration point is exercised — not just
individual endpoints in isolation.

Each test:
1. Registers a ``FakeCrm`` as the subscriber's webhook target.
2. Triggers a real server-side event (inbound INVITE, outbound device-dial,
   hold/unhold action).
3. Lets the server fire webhooks → FakeCrm reacts synchronously, issuing
   the same HTTP calls the real CRM would.
4. Asserts end state: calls/participants/bridges/audio.
"""
from __future__ import annotations

import json
import threading
import time
from unittest.mock import MagicMock

import pytest

from conftest import SUBSCRIBER_ID

PBX_ID = "TestPBX"
from speech_pipeline.telephony import call_state, leg as leg_mod
from fake_crm import FakeCrm


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def crm(client, admin, account):
    token = account["Authorization"].split(None, 1)[1]
    crm = FakeCrm(client, admin_headers=admin, account_token=token)
    crm.register_as_subscriber(SUBSCRIBER_ID, PBX_ID)
    yield crm


# ---------------------------------------------------------------------------
# Inbound — full webhook flow
# ---------------------------------------------------------------------------

class TestInboundFullFlow:
    """Simulates an inbound SIP INVITE → CRM receives `incoming` webhook →
    creates call, bridges leg, dials internal phones → on answer bridges
    them."""

    def test_inbound_incoming_creates_call_and_bridges(self, client, account, crm, monkeypatch):
        """Fire the incoming webhook + verify CRM-side reaction."""
        from speech_pipeline.telephony import dispatcher
        from speech_pipeline.telephony import subscriber as sub_mod

        # Create a fake leg as sip_listener would.
        fake_voip = MagicMock()
        fake_voip.RTPClients = []
        leg = leg_mod.create_leg(
            direction="inbound",
            number="+49174000",
            pbx_id=PBX_ID,
            subscriber_id=SUBSCRIBER_ID,
            voip_call=fake_voip,
        )

        with crm.active(monkeypatch):
            sub = sub_mod.get(SUBSCRIBER_ID)
            dispatcher.fire_subscriber_event(sub, "incoming", {
                "caller": "+49174000",
                "callee": "+493586",
                "leg_id": leg.leg_id,
            })
            time.sleep(0.1)

        # FakeCrm received the webhook …
        assert any(w["state"] == "incoming" for w in crm.webhooks)
        # … created a call row …
        assert len(crm.calls) == 1
        call_db = next(iter(crm.calls.values()))
        assert call_db["sid"].startswith("call-")
        assert call_db["direction"] == "inbound"
        # … and the server bridged the leg into that call.
        call = call_state.get_call(call_db["sid"])
        assert call is not None
        sip_parts = [p for p in call.list_participants()
                     if p.get("type") == "sip"]
        assert len(sip_parts) == 1
        assert sip_parts[0]["id"] == leg.leg_id

        # Cleanup
        client.delete(f"/api/calls/{call_db['sid']}", headers=account)
        leg_mod._legs.pop(leg.leg_id, None)

    def test_inbound_with_internal_phone_dialed(
            self, client, account, crm, monkeypatch):
        """CRM configured with an internal phone → findParticipant fires,
        originate:NUMBER DSL returns a leg_id."""
        crm.internal_phones = [{"number": "sip:admin@crm.example", "answerer_id": 1}]

        from speech_pipeline.telephony import dispatcher, subscriber as sub_mod

        # Stop originate_only from dialing for real.
        monkeypatch.setattr(leg_mod, "originate_only", lambda leg, pbx: None)

        fake_voip = MagicMock()
        fake_voip.RTPClients = []
        leg = leg_mod.create_leg(
            direction="inbound", number="+49174000", pbx_id=PBX_ID,
            subscriber_id=SUBSCRIBER_ID, voip_call=fake_voip,
        )

        with crm.active(monkeypatch):
            sub = sub_mod.get(SUBSCRIBER_ID)
            dispatcher.fire_subscriber_event(sub, "incoming", {
                "caller": "+49174000",
                "callee": "+493586",
                "leg_id": leg.leg_id,
            })
            time.sleep(0.1)

        # Participant for the internal phone was created + originate
        # DSL ran (the leg_id would be non-empty).
        internal_pid = None
        for pid, p in crm.participants.items():
            if p.get("number") == "sip:admin@crm.example":
                internal_pid = pid
                break
        assert internal_pid is not None
        assert crm.participants[internal_pid]["sid"].startswith("leg-")

        # Cleanup
        call_sid = next(iter(crm.calls.values()))["sid"]
        client.delete(f"/api/calls/{call_sid}", headers=account)
        leg_mod._legs.pop(leg.leg_id, None)

    def test_inbound_leg_answered_stops_wait_music(
            self, client, account, crm, monkeypatch):
        """Answered webhook → CRM bridges the answered leg and deletes
        the wait-music stage."""
        crm.wait_jingle = "examples/queue.mp3"

        from speech_pipeline.telephony import dispatcher, subscriber as sub_mod

        fake_voip = MagicMock()
        fake_voip.RTPClients = []
        leg = leg_mod.create_leg(
            direction="inbound", number="+49174000", pbx_id=PBX_ID,
            subscriber_id=SUBSCRIBER_ID, voip_call=fake_voip,
        )

        with crm.active(monkeypatch):
            sub = sub_mod.get(SUBSCRIBER_ID)
            dispatcher.fire_subscriber_event(sub, "incoming", {
                "caller": "+49174000",
                "callee": "+493586",
                "leg_id": leg.leg_id,
            })
            time.sleep(0.1)

            call_sid = next(iter(crm.calls.values()))["sid"]
            call = call_state.get_call(call_sid)
            assert call is not None
            # Verify wait music was started.
            wait_stage = f"play:{call_sid}_wait"
            resp = client.get(f"/api/pipelines?dsl={wait_stage}",
                              headers=account)
            assert resp.status_code == 200

            # Simulate leg answered webhook (CRM's answered handler).
            crm._route(
                crm.BASE_URL + "/Telephone/SpeechServer/public"
                f"?state=leg&event=answered&call="
                f"{next(iter(crm.calls.keys()))}&participant=0",
                {"leg_id": leg.leg_id},
                crm.account_token,
                method="POST",
            )
            time.sleep(0.1)

            # Wait-music stage should be gone.
            resp = client.get(f"/api/pipelines?dsl={wait_stage}",
                              headers=account)
            assert resp.status_code == 404

        # Cleanup
        client.delete(f"/api/calls/{call_sid}", headers=account)
        leg_mod._legs.pop(leg.leg_id, None)


# ---------------------------------------------------------------------------
# Outbound — device-dial webhook flow
# ---------------------------------------------------------------------------

class TestOutboundDeviceDial:
    """Internal SIP phone dials an external number → server fires
    ``device_dial`` webhook → CRM creates outbound call + attaches leg
    + originate external participant."""

    def test_device_dial_creates_outbound_call(
            self, client, account, crm, monkeypatch):
        from speech_pipeline.telephony import dispatcher, subscriber as sub_mod

        monkeypatch.setattr(leg_mod, "originate_only", lambda leg, pbx: None)

        # Simulate the device's existing leg.
        fake_voip = MagicMock()
        fake_voip.RTPClients = []
        internal_leg = leg_mod.create_leg(
            direction="outbound", number="sip:admin@crm.example",
            pbx_id=PBX_ID, subscriber_id=SUBSCRIBER_ID,
            voip_call=fake_voip,
        )

        with crm.active(monkeypatch):
            sub = sub_mod.get(SUBSCRIBER_ID)
            dispatcher.fire_subscriber_event(sub, "device_dial", {
                "number": "+4935863",
                "leg_id": internal_leg.leg_id,
                "sip_user": "admin",
                "user_id": 42,
            })
            time.sleep(0.1)

        # CRM created an outbound call row.
        assert len(crm.calls) == 1
        call_db = next(iter(crm.calls.values()))
        assert call_db["direction"] == "outbound"
        assert call_db["callee"] == "+4935863"
        assert call_db["sid"].startswith("call-")

        # CRM issued the originate for the external participant.
        external = [p for p in crm.participants.values()
                    if p.get("number") == "+4935863"]
        assert len(external) == 1
        assert external[0]["sid"].startswith("leg-")

        # Cleanup
        client.delete(f"/api/calls/{call_db['sid']}", headers=account)
        leg_mod._legs.pop(internal_leg.leg_id, None)
        for p in external:
            leg_mod._legs.pop(p["sid"], None)


# ---------------------------------------------------------------------------
# Completion → cleanup
# ---------------------------------------------------------------------------

class TestLegCompletedCleansUp:
    """``leg&event=completed`` webhook → CRM updates participant + closes
    the call if no one is left."""

    def test_single_participant_completed_ends_call(
            self, client, account, crm, monkeypatch):
        from speech_pipeline.telephony import dispatcher, subscriber as sub_mod

        fake_voip = MagicMock()
        fake_voip.RTPClients = []
        leg = leg_mod.create_leg(
            direction="inbound", number="+49174000", pbx_id=PBX_ID,
            subscriber_id=SUBSCRIBER_ID, voip_call=fake_voip,
        )

        with crm.active(monkeypatch):
            sub = sub_mod.get(SUBSCRIBER_ID)
            dispatcher.fire_subscriber_event(sub, "incoming", {
                "caller": "+49174000",
                "callee": "+493586",
                "leg_id": leg.leg_id,
            })
            time.sleep(0.1)

            call_db_id = next(iter(crm.calls.keys()))
            call_sid = crm.calls[call_db_id]["sid"]

            # Fire completed webhook for the caller leg.
            crm._route(
                crm.BASE_URL + "/Telephone/SpeechServer/public"
                f"?state=leg&event=completed&call={call_db_id}",
                {"leg_id": leg.leg_id},
                crm.account_token,
                method="POST",
            )
            time.sleep(0.2)

        assert crm.calls[call_db_id]["status"] == "completed"
        assert call_state.get_call(call_sid) is None

        leg_mod._legs.pop(leg.leg_id, None)
