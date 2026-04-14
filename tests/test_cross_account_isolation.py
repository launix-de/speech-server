"""Cross-account isolation contract.

Account-scoped operations must NEVER touch another account's resources.
The admin token is allowed through (ops / monitoring / emergency reset).

The specific customer concern: CRM A must not be able to cancel CRM B's
ongoing calls, either by accident or maliciously.  Admin still can.
"""
from __future__ import annotations

import json

import pytest

from conftest import (
    ADMIN_TOKEN, ACCOUNT_TOKEN, ACCOUNT_ID, SUBSCRIBER_ID,
    ACCOUNT2_TOKEN, ACCOUNT2_ID, SUBSCRIBER2_ID,
)


class TestCrossAccountCallDelete:

    def test_account_cannot_delete_other_accounts_call(
            self, client, account, account2):
        """Account A creates a call; account B's token must get 403 on
        DELETE and the call must still be alive afterwards."""
        call_sid = client.post("/api/calls",
                                data=json.dumps({"subscriber_id": SUBSCRIBER_ID}),
                                headers=account).get_json()["call_id"]
        try:
            resp = client.delete(f"/api/calls/{call_sid}",
                                  headers=account2)
            assert resp.status_code == 403, (
                f"Cross-account DELETE was NOT rejected: "
                f"{resp.status_code} {resp.get_data(as_text=True)}"
            )
            # Still alive.
            resp2 = client.get(f"/api/pipelines?dsl=call:{call_sid}",
                                headers=account)
            assert resp2.status_code == 200, (
                f"Call was torn down despite the 403: "
                f"{resp2.status_code} {resp2.get_data(as_text=True)}"
            )
        finally:
            client.delete(f"/api/calls/{call_sid}", headers=account)

    def test_admin_can_delete_any_call(self, client, admin, account):
        """Admin token bypasses the per-account ownership check."""
        call_sid = client.post("/api/calls",
                                data=json.dumps({"subscriber_id": SUBSCRIBER_ID}),
                                headers=account).get_json()["call_id"]
        resp = client.delete(f"/api/calls/{call_sid}", headers=admin)
        assert resp.status_code == 204, (
            f"Admin DELETE was blocked: {resp.status_code} "
            f"{resp.get_data(as_text=True)}"
        )
        # And truly gone.
        resp2 = client.get(f"/api/pipelines?dsl=call:{call_sid}",
                            headers=account)
        assert resp2.status_code == 404

    def test_unknown_call_still_404_not_403(self, client, account):
        """Sanity: asking to delete something that never existed
        returns 404 (not 403) so callers can distinguish 'you don't
        own it' from 'it isn't here'."""
        resp = client.delete("/api/calls/call-does-not-exist",
                              headers=account)
        assert resp.status_code == 404, resp.get_data(as_text=True)


class TestCrossAccountPipelineMutation:
    """Extra coverage: POST /api/pipelines referencing another
    account's call must also be rejected."""

    def test_post_pipeline_on_other_accounts_call_rejected(
            self, client, account, account2):
        call_sid = client.post("/api/calls",
                                data=json.dumps({"subscriber_id": SUBSCRIBER_ID}),
                                headers=account).get_json()["call_id"]
        try:
            resp = client.post("/api/pipelines",
                                data=json.dumps({
                                    "dsl": f"tts:de_DE-thorsten-medium"
                                           f'{{"text":"hi"}} -> call:{call_sid}',
                                }),
                                headers=account2)
            assert resp.status_code in (403, 404), (
                f"Account 2 was able to POST a pipeline into account "
                f"1's call: {resp.status_code} "
                f"{resp.get_data(as_text=True)}"
            )
        finally:
            client.delete(f"/api/calls/{call_sid}", headers=account)

    def test_delete_pipeline_cannot_kill_other_accounts_stage(
            self, client, account, account2):
        """DELETE /api/pipelines {dsl: play:X} must be scoped to the
        caller's account — account B can't reach into A's stages."""
        call_sid = client.post("/api/calls",
                                data=json.dumps({"subscriber_id": SUBSCRIBER_ID}),
                                headers=account).get_json()["call_id"]
        try:
            # A POSTs a play stage.
            client.post("/api/pipelines", data=json.dumps({
                "dsl": f'play:xa_stage{{"url":"examples/queue.mp3",'
                       f'"loop":true}} -> call:{call_sid}',
            }), headers=account)

            # B tries to DELETE it.
            resp = client.delete("/api/pipelines", data=json.dumps({
                "dsl": "play:xa_stage",
            }), headers=account2)
            assert resp.status_code in (403, 404), (
                f"Account 2 was able to delete account 1's play "
                f"stage: {resp.status_code}"
            )
        finally:
            client.delete(f"/api/calls/{call_sid}", headers=account)
