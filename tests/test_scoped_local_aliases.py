from __future__ import annotations

import json

from conftest import ACCOUNT_ID, ACCOUNT2_ID
from speech_pipeline.telephony import auth as auth_mod
from speech_pipeline.telephony import call_state


def _rekey_call(call, new_id: str) -> None:
    old = call.call_id
    call.call_id = new_id
    call_state._calls[new_id] = call_state._calls.pop(old)


class TestScopedLocalAliases:
    def test_account_lookup_prefixes_local_call_name_but_admin_can_use_global_name(
            self, client, account, account2, admin):
        c1 = call_state.create_call("test-subscriber", ACCOUNT_ID, "TestPBX")
        c2 = call_state.create_call("test-subscriber2", ACCOUNT2_ID, "TestPBX2")
        try:
            _rekey_call(c1, f"{ACCOUNT_ID}:call-same")
            _rekey_call(c2, f"{ACCOUNT2_ID}:call-same")

            resp1 = client.get("/api/pipelines?dsl=call:call-same", headers=account)
            assert resp1.status_code == 200
            assert resp1.get_json()["call_id"] == "call-same"

            resp2 = client.get("/api/pipelines?dsl=call:call-same", headers=account2)
            assert resp2.status_code == 200
            assert resp2.get_json()["call_id"] == "call-same"

            adm1 = client.get(
                f"/api/pipelines?dsl=call:{ACCOUNT_ID}:call-same",
                headers=admin,
            )
            adm2 = client.get(
                f"/api/pipelines?dsl=call:{ACCOUNT2_ID}:call-same",
                headers=admin,
            )
            assert adm1.status_code == 200
            assert adm2.status_code == 200
            assert adm1.get_json()["call_id"] == f"{ACCOUNT_ID}:call-same"
            assert adm2.get_json()["call_id"] == f"{ACCOUNT2_ID}:call-same"
        finally:
            call_state.delete_call(c1.call_id)
            call_state.delete_call(c2.call_id)

    def test_account_lookup_prefixes_local_nonce_name(self, client, account, account2, admin):
        n1 = auth_mod.create_nonce(ACCOUNT_ID, "test-subscriber", "u1", ttl=60)
        n2 = auth_mod.create_nonce(ACCOUNT2_ID, "test-subscriber2", "u2", ttl=60)
        local = "n-same"
        auth_mod._nonces[f"{ACCOUNT_ID}:{local}"] = {**n1, "nonce": f"{ACCOUNT_ID}:{local}"}
        auth_mod._nonces[f"{ACCOUNT2_ID}:{local}"] = {**n2, "nonce": f"{ACCOUNT2_ID}:{local}"}
        auth_mod._nonces.pop(n1["nonce"], None)
        auth_mod._nonces.pop(n2["nonce"], None)

        resp1 = client.delete(f"/api/nonce/{local}", headers=account)
        assert resp1.status_code == 204
        assert f"{ACCOUNT_ID}:{local}" not in auth_mod._nonces
        assert f"{ACCOUNT2_ID}:{local}" in auth_mod._nonces

        resp2 = client.delete(f"/api/nonce/{local}", headers=account2)
        assert resp2.status_code == 204
        assert f"{ACCOUNT2_ID}:{local}" not in auth_mod._nonces

        n3 = auth_mod.create_nonce(ACCOUNT_ID, "test-subscriber", "u1", ttl=60)
        auth_mod._nonces[f"{ACCOUNT_ID}:{local}"] = {**n3, "nonce": f"{ACCOUNT_ID}:{local}"}
        auth_mod._nonces.pop(n3["nonce"], None)
        listed = client.get("/api/nonces", headers=account)
        assert listed.status_code == 200
        assert any(n["nonce"] == local for n in listed.get_json())
        admin_listed = client.get("/api/nonces", headers=admin)
        assert any(n["nonce"] == f"{ACCOUNT_ID}:{local}" for n in admin_listed.get_json())

    def test_non_admin_foreign_global_name_gets_prefixed_again(self, client, account):
        other = f"{ACCOUNT2_ID}:call-not-yours"
        resp = client.delete(f"/api/calls/{other}", headers=account)
        assert resp.status_code == 404
