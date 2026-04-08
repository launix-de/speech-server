"""Stage management API tests.

Covers: delete stage, nonexistent call/stage, no executor.
"""
import json

from conftest import create_call


class TestStages:
    def test_delete_stage_nonexistent_call(self, client, account):
        resp = client.delete("/api/calls/no-such/stages/play:x", headers=account)
        assert resp.status_code == 404

    def test_delete_stage_no_executor(self, client, account):
        call_id = create_call(client, account)
        resp = client.delete(f"/api/calls/{call_id}/stages/play:x", headers=account)
        assert resp.status_code == 400

    def test_delete_stage_not_found(self, client, account):
        call_id = create_call(client, account)
        # Create pipe executor by sending empty pipes
        client.post(f"/api/calls/{call_id}/pipes",
                    data=json.dumps({"pipes": []}),
                    headers=account)
        resp = client.delete(f"/api/calls/{call_id}/stages/play:nonexistent",
                             headers=account)
        assert resp.status_code == 404
