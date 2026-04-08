"""Pipes endpoint API tests.

Covers: input validation, nonexistent call, empty pipes, DSL error reporting.
"""
import json

from conftest import create_call


class TestPipesEndpoint:
    def test_pipes_missing_body_treated_as_empty(self, client, account):
        call_id = create_call(client, account)
        resp = client.post(f"/api/calls/{call_id}/pipes",
                           data=json.dumps({}),
                           headers=account)
        # No "pipes" key → defaults to [] → valid empty list
        assert resp.status_code == 202

    def test_pipes_invalid_type(self, client, account):
        call_id = create_call(client, account)
        resp = client.post(f"/api/calls/{call_id}/pipes",
                           data=json.dumps({"pipes": "not-a-list"}),
                           headers=account)
        assert resp.status_code == 400

    def test_pipes_nonexistent_call(self, client, account):
        resp = client.post("/api/calls/no-such/pipes",
                           data=json.dumps({"pipes": ["play:x"]}),
                           headers=account)
        assert resp.status_code == 404

    def test_pipes_empty_list_ok(self, client, account):
        call_id = create_call(client, account)
        resp = client.post(f"/api/calls/{call_id}/pipes",
                           data=json.dumps({"pipes": []}),
                           headers=account)
        assert resp.status_code == 202
        assert resp.get_json()["results"] == []

    def test_pipes_returns_call_id(self, client, account):
        call_id = create_call(client, account)
        resp = client.post(f"/api/calls/{call_id}/pipes",
                           data=json.dumps({"pipes": []}),
                           headers=account)
        assert resp.get_json()["call_id"] == call_id


class TestPipesDSLErrors:
    """DSL errors are reported per-pipe in results, not as HTTP errors."""

    def test_invalid_dsl_syntax_reported(self, client, account):
        call_id = create_call(client, account)
        resp = client.post(f"/api/calls/{call_id}/pipes",
                           data=json.dumps({"pipes": ["123invalid"]}),
                           headers=account)
        assert resp.status_code == 202
        results = resp.get_json()["results"]
        assert len(results) == 1
        assert results[0]["ok"] is False
        assert "error" in results[0]

    def test_unknown_element_type_reported(self, client, account):
        call_id = create_call(client, account)
        resp = client.post(f"/api/calls/{call_id}/pipes",
                           data=json.dumps({"pipes": [
                               f"bogus:x -> call:{call_id}"
                           ]}),
                           headers=account)
        assert resp.status_code == 202
        results = resp.get_json()["results"]
        assert results[0]["ok"] is False

    def test_call_id_mismatch_reported(self, client, account):
        call_id = create_call(client, account)
        resp = client.post(f"/api/calls/{call_id}/pipes",
                           data=json.dumps({"pipes": [
                               "play:x -> call:call-wrong"
                           ]}),
                           headers=account)
        results = resp.get_json()["results"]
        assert results[0]["ok"] is False
        assert "bound to" in results[0]["error"]

    def test_multiple_pipes_mixed_results(self, client, account):
        """One valid, one invalid pipe → results reflect each."""
        call_id = create_call(client, account)
        resp = client.post(f"/api/calls/{call_id}/pipes",
                           data=json.dumps({"pipes": [
                               "123invalid",
                               "play:x -> call:call-wrong",
                           ]}),
                           headers=account)
        results = resp.get_json()["results"]
        assert len(results) == 2
        assert all(r["ok"] is False for r in results)

    def test_webhook_not_terminal_reported(self, client, account):
        call_id = create_call(client, account)
        resp = client.post(f"/api/calls/{call_id}/pipes",
                           data=json.dumps({"pipes": [
                               f"webhook:https://example.com -> call:{call_id}"
                           ]}),
                           headers=account)
        results = resp.get_json()["results"]
        assert results[0]["ok"] is False
        assert "final" in results[0]["error"].lower()

    def test_call_as_first_element_reported(self, client, account):
        call_id = create_call(client, account)
        resp = client.post(f"/api/calls/{call_id}/pipes",
                           data=json.dumps({"pipes": [
                               f"call:{call_id} -> sip:l1"
                           ]}),
                           headers=account)
        results = resp.get_json()["results"]
        assert results[0]["ok"] is False
        assert "first" in results[0]["error"].lower()
