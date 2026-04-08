"""Commands endpoint API tests.

Covers: validation, nonexistent call, empty list, async acceptance.
"""
import json

from conftest import create_call


class TestCommands:
    def test_commands_missing_list(self, client, account):
        call_id = create_call(client, account)
        resp = client.post(f"/api/calls/{call_id}/commands",
                           data=json.dumps({"commands": "not-a-list"}),
                           headers=account)
        assert resp.status_code == 400

    def test_commands_empty_list_accepted(self, client, account):
        call_id = create_call(client, account)
        resp = client.post(f"/api/calls/{call_id}/commands",
                           data=json.dumps({"commands": []}),
                           headers=account)
        assert resp.status_code == 202
        data = resp.get_json()
        assert data["queued"] == 0
        assert data["call_id"] == call_id

    def test_commands_nonexistent_call(self, client, account):
        resp = client.post("/api/calls/no-such/commands",
                           data=json.dumps({"commands": []}),
                           headers=account)
        assert resp.status_code == 404

    def test_commands_returns_queued_count(self, client, account):
        call_id = create_call(client, account)
        resp = client.post(f"/api/calls/{call_id}/commands",
                           data=json.dumps({"commands": [
                               {"action": "play", "url": "x"},
                               {"action": "hangup"},
                           ]}),
                           headers=account)
        assert resp.status_code == 202
        assert resp.get_json()["queued"] == 2
