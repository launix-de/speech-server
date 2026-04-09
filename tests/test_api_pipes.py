"""Pipeline creation API tests (POST /api/pipelines with DSL).

Tests DSL validation, error reporting, and call ownership via the
unified /api/pipelines endpoint.
"""
import json

from conftest import create_call, SUBSCRIBER2_ID


class TestPipelineCreation:
    def test_missing_dsl(self, client, account):
        resp = client.post("/api/pipelines",
                           data=json.dumps({}),
                           headers=account)
        assert resp.status_code == 400

    def test_empty_dsl(self, client, account):
        resp = client.post("/api/pipelines",
                           data=json.dumps({"dsl": ""}),
                           headers=account)
        assert resp.status_code == 400


class TestPipelineDSLErrors:
    def test_invalid_syntax(self, client, account):
        resp = client.post("/api/pipelines",
                           data=json.dumps({"dsl": "123invalid"}),
                           headers=account)
        assert resp.status_code == 400

    def test_unknown_element_type(self, client, account):
        call_id = create_call(client, account)
        resp = client.post("/api/pipelines",
                           data=json.dumps({"dsl": f"bogus:x -> call:{call_id}"}),
                           headers=account)
        assert resp.status_code == 400

    def test_webhook_not_terminal(self, client, account):
        call_id = create_call(client, account)
        resp = client.post("/api/pipelines",
                           data=json.dumps({
                               "dsl": f"webhook:https://example.com -> call:{call_id}"
                           }),
                           headers=account)
        assert resp.status_code == 400

    def test_call_as_first_element(self, client, account):
        call_id = create_call(client, account)
        resp = client.post("/api/pipelines",
                           data=json.dumps({
                               "dsl": f"call:{call_id} -> sip:l1"
                           }),
                           headers=account)
        assert resp.status_code == 400

    def test_nonexistent_call_rejected(self, client, account):
        resp = client.post("/api/pipelines",
                           data=json.dumps({
                               "dsl": "play:x -> call:call-doesnotexist"
                           }),
                           headers=account)
        assert resp.status_code in (400, 403)


class TestPipelineCallOwnership:
    def test_account_cannot_pipe_into_other_call(self, client, account, account2):
        """Account A cannot create pipeline referencing account B's call."""
        call_id = create_call(client, account2, SUBSCRIBER2_ID)
        resp = client.post("/api/pipelines",
                           data=json.dumps({
                               "dsl": f"play:x -> call:{call_id}"
                           }),
                           headers=account)
        assert resp.status_code == 403

    def test_admin_can_pipe_into_any_call(self, client, admin, account):
        call_id = create_call(client, account)
        resp = client.post("/api/pipelines",
                           data=json.dumps({
                               "dsl": f"play:x -> call:{call_id}"
                           }),
                           headers=admin)
        # May fail on play URL resolution, but auth must pass (not 403)
        assert resp.status_code != 403
