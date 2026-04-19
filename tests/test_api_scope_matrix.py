"""HTTP-level scope and namespace contract tests."""
from __future__ import annotations

from conftest import ACCOUNT2_ID, ACCOUNT_ID, SUBSCRIBER2_ID, create_call


class TestApiScopeMatrix:
    def test_account_reads_local_call_id_but_admin_reads_global_call_id(
        self, client, account, admin
    ):
        local_call_id = create_call(client, account)
        global_call_id = f"{ACCOUNT_ID}:{local_call_id}"

        resp = client.get(f"/api/pipelines?dsl=call:{local_call_id}", headers=account)
        assert resp.status_code == 200
        assert resp.get_json()["call_id"] == local_call_id

        resp = client.get(f"/api/pipelines?dsl=call:{global_call_id}", headers=admin)
        assert resp.status_code == 200
        assert resp.get_json()["call_id"] == global_call_id

    def test_account_call_list_stays_local_while_admin_list_is_global(
        self, client, account, account2, admin
    ):
        call1 = create_call(client, account)
        call2 = create_call(client, account2, SUBSCRIBER2_ID)

        resp = client.get("/api/calls", headers=account)
        assert resp.status_code == 200
        listed = [c["call_id"] for c in resp.get_json()]
        assert call1 in listed
        assert call2 not in listed
        assert all(":" not in cid for cid in listed)

        resp = client.get("/api/calls", headers=admin)
        assert resp.status_code == 200
        listed = [c["call_id"] for c in resp.get_json()]
        assert f"{ACCOUNT_ID}:{call1}" in listed
        assert f"{ACCOUNT2_ID}:{call2}" in listed

    def test_account_webclient_slot_response_stays_local(self, client, account):
        import json

        call_id = create_call(client, account)
        dsl = (
            'webclient:user_42{"base_url":"https://speech.example.com/tts",'
            f'"call_id":"{call_id}"'
            "}"
        )
        resp = client.post(
            "/api/pipelines",
            data=json.dumps({"dsl": dsl}),
            headers=account,
        )
        assert resp.status_code == 201, resp.get_data(as_text=True)
        data = resp.get_json()
        assert ":" not in data["session_id"]
        assert ":" not in data["nonce"]
        assert data["iframe_url"].startswith("https://speech.example.com/tts/phone/")
