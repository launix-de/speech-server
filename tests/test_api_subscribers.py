"""Subscriber management API tests.

Covers: CRUD, DID conflicts, SIP domain mapping, ownership, idempotent
re-registration, inbound routing lookups.
"""
import json

from conftest import SUBSCRIBER_ID


class TestSubscriberCRUD:
    def test_create_subscriber(self, client, account):
        resp = client.get(f"/api/subscribers/{SUBSCRIBER_ID}", headers=account)
        assert resp.status_code == 200
        assert resp.get_json()["base_url"] == "https://example.com/crm"

    def test_list_subscribers(self, client, account):
        resp = client.get("/api/subscribers", headers=account)
        assert resp.status_code == 200
        assert len(resp.get_json()) >= 1

    def test_delete_subscriber(self, client, account):
        resp = client.delete(f"/api/subscribers/{SUBSCRIBER_ID}", headers=account)
        assert resp.status_code in (200, 204)

    def test_get_nonexistent_subscriber(self, client, account):
        resp = client.get("/api/subscribers/no-such", headers=account)
        assert resp.status_code == 404

    def test_delete_nonexistent_subscriber(self, client, account):
        resp = client.delete("/api/subscribers/no-such", headers=account)
        assert resp.status_code == 404


class TestSubscriberSIPDomain:
    def test_sip_domain_registered(self, client, account):
        from speech_pipeline.telephony.subscriber import find_by_sip_domain
        sub = find_by_sip_domain("crm.example.com")
        assert sub is not None
        assert sub["id"] == SUBSCRIBER_ID

    def test_sip_domain_updated_on_reregister(self, client, account):
        from speech_pipeline.telephony.subscriber import find_by_sip_domain
        client.put("/api/subscribe/sub-sip",
                   data=json.dumps({
                       "base_url": "https://old.example.com/crm",
                       "bearer_token": "t",
                   }),
                   headers=account)
        assert find_by_sip_domain("crm.old.example.com") is not None
        client.put("/api/subscribe/sub-sip",
                   data=json.dumps({
                       "base_url": "https://new.example.com/crm",
                       "bearer_token": "t",
                   }),
                   headers=account)
        assert find_by_sip_domain("crm.old.example.com") is None
        assert find_by_sip_domain("crm.new.example.com") is not None

    def test_sip_domain_deep_path(self, client, account):
        from speech_pipeline.telephony.subscriber import find_by_sip_domain
        client.put("/api/subscribe/sub-deep",
                   data=json.dumps({
                       "base_url": "https://launix.de/fop/crm-neu",
                       "bearer_token": "t",
                   }),
                   headers=account)
        assert find_by_sip_domain("crm-neu.fop.launix.de") is not None

    def test_sip_domain_no_path(self, client, account):
        from speech_pipeline.telephony.subscriber import find_by_sip_domain
        client.put("/api/subscribe/sub-nopath",
                   data=json.dumps({
                       "base_url": "https://crm.launix.de",
                       "bearer_token": "t",
                   }),
                   headers=account)
        assert find_by_sip_domain("crm.launix.de") is not None


class TestSubscriberDIDs:
    def test_did_conflict_rejected(self, client, account):
        """Two subscribers cannot claim the same DID."""
        client.put("/api/subscribe/sub-did-a",
                   data=json.dumps({
                       "base_url": "https://a.example.com",
                       "bearer_token": "t",
                       "inbound_dids": ["+4912345"],
                   }),
                   headers=account)
        resp = client.put("/api/subscribe/sub-did-b",
                          data=json.dumps({
                              "base_url": "https://b.example.com",
                              "bearer_token": "t",
                              "inbound_dids": ["+4912345"],
                          }),
                          headers=account)
        assert resp.status_code == 400

    def test_did_freed_after_delete(self, client, account):
        """Deleting subscriber frees its DIDs for reuse."""
        client.put("/api/subscribe/sub-did-free",
                   data=json.dumps({
                       "base_url": "https://free.example.com",
                       "bearer_token": "t",
                       "inbound_dids": ["+49999"],
                   }),
                   headers=account)
        client.delete("/api/subscribers/sub-did-free", headers=account)
        resp = client.put("/api/subscribe/sub-did-reuse",
                          data=json.dumps({
                              "base_url": "https://reuse.example.com",
                              "bearer_token": "t",
                              "inbound_dids": ["+49999"],
                          }),
                          headers=account)
        assert resp.status_code == 200

    def test_did_lookup(self, client, account):
        from speech_pipeline.telephony.subscriber import find_by_did
        client.put("/api/subscribe/sub-did-lookup",
                   data=json.dumps({
                       "base_url": "https://lookup.example.com",
                       "bearer_token": "t",
                       "inbound_dids": ["+49111"],
                   }),
                   headers=account)
        sub = find_by_did("+49111")
        assert sub is not None
        assert sub["id"] == "sub-did-lookup"

    def test_did_lookup_miss(self, client, account):
        from speech_pipeline.telephony.subscriber import find_by_did
        assert find_by_did("+49000000") is None

    def test_multiple_dids(self, client, account):
        from speech_pipeline.telephony.subscriber import find_by_did
        client.put("/api/subscribe/sub-multi-did",
                   data=json.dumps({
                       "base_url": "https://multi.example.com",
                       "bearer_token": "t",
                       "inbound_dids": ["+49222", "+49333"],
                   }),
                   headers=account)
        assert find_by_did("+49222")["id"] == "sub-multi-did"
        assert find_by_did("+49333")["id"] == "sub-multi-did"

    def test_did_update_on_reregister(self, client, account):
        """Reregistering subscriber updates DID map."""
        from speech_pipeline.telephony.subscriber import find_by_did
        client.put("/api/subscribe/sub-did-upd",
                   data=json.dumps({
                       "base_url": "https://upd.example.com",
                       "bearer_token": "t",
                       "inbound_dids": ["+49444"],
                   }),
                   headers=account)
        client.put("/api/subscribe/sub-did-upd",
                   data=json.dumps({
                       "base_url": "https://upd.example.com",
                       "bearer_token": "t",
                       "inbound_dids": ["+49555"],
                   }),
                   headers=account)
        assert find_by_did("+49444") is None
        assert find_by_did("+49555")["id"] == "sub-did-upd"


class TestSubscriberOwnership:
    def test_idempotent_reregister(self, client, account):
        resp = client.put(f"/api/subscribe/{SUBSCRIBER_ID}",
                          data=json.dumps({
                              "base_url": "https://updated.example.com/crm",
                              "bearer_token": "new-token",
                          }),
                          headers=account)
        assert resp.status_code == 200
        assert resp.get_json()["base_url"] == "https://updated.example.com/crm"

    def test_subscriber_stores_events(self, client, account):
        client.put("/api/subscribe/sub-events",
                   data=json.dumps({
                       "base_url": "https://events.example.com",
                       "bearer_token": "t",
                       "events": {"inbound": "/Telephone/Inbound"},
                   }),
                   headers=account)
        resp = client.get("/api/subscribers/sub-events", headers=account)
        assert resp.get_json()["events"]["inbound"] == "/Telephone/Inbound"
