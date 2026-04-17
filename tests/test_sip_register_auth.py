"""SIP REGISTER digest path: HA1 cache + 3 s timeout.

The original 408 Request Timeout incident: ``_crm_login`` blocked
for up to 10 s on a slow CRM, longer than the SIP device's REGISTER
timeout.  Tests now enforce both the cache (5 min) and the per-call
3 s upper bound.
"""
from __future__ import annotations

import time

import pytest

from speech_pipeline.telephony import sip_stack


@pytest.fixture(autouse=True)
def _clear_ha1_cache():
    sip_stack._HA1_CACHE.clear()
    yield
    sip_stack._HA1_CACHE.clear()


class TestHA1Cache:

    def test_second_lookup_hits_cache(self, monkeypatch):
        """Second ``_crm_login`` for same (subscriber, user, realm,
        sip_user) does not hit the CRM again within the TTL."""
        calls = []

        class _FakeResp:
            status_code = 200
            def json(self):
                return {"ha1": "abcdef0123456789", "user_id": 42}

        def fake_get(url, params=None, headers=None, timeout=None):
            calls.append({"url": url, "params": params,
                          "timeout": timeout})
            return _FakeResp()

        monkeypatch.setattr(sip_stack.requests, "get", fake_get)
        sub = {"id": "sub1", "base_url": "https://crm.example",
               "bearer_token": "tok"}

        first = sip_stack._crm_login(sub, "carli", "realm", "carli@x")
        second = sip_stack._crm_login(sub, "carli", "realm", "carli@x")
        assert first == second
        assert len(calls) == 1, (
            f"_crm_login hit the CRM {len(calls)} times; cache broken"
        )

    def test_lookup_uses_short_timeout(self, monkeypatch):
        """Per-call timeout MUST stay below the SIP device's REGISTER
        wait (~5 s) — otherwise the device times out before us."""
        captured = {}

        class _FakeResp:
            status_code = 200
            def json(self):
                return {"ha1": "x"}

        def fake_get(url, params=None, headers=None, timeout=None):
            captured["timeout"] = timeout
            return _FakeResp()

        monkeypatch.setattr(sip_stack.requests, "get", fake_get)
        sip_stack._crm_login(
            {"id": "s2", "base_url": "https://crm", "bearer_token": "t"},
            "u", "r", "u@x",
        )
        assert captured["timeout"] is not None
        assert captured["timeout"] < 5.0, (
            f"_crm_login timeout = {captured['timeout']}s — must be "
            f"< 5s so SIP REGISTER doesn't 408 first"
        )

    def test_cache_expires_after_ttl(self, monkeypatch):
        sip_stack._HA1_CACHE.clear()
        # Force a tiny TTL.
        monkeypatch.setattr(sip_stack, "_HA1_TTL", 0.1)
        calls = []

        class _R:
            status_code = 200
            def json(self):
                return {"ha1": "x"}

        def fake_get(*a, **kw):
            calls.append(1)
            return _R()

        monkeypatch.setattr(sip_stack.requests, "get", fake_get)
        sub = {"id": "s3", "base_url": "https://crm", "bearer_token": "t"}
        sip_stack._crm_login(sub, "u", "r", "u@x")
        time.sleep(0.15)
        sip_stack._crm_login(sub, "u", "r", "u@x")
        assert len(calls) == 2, "cache did not expire"

    def test_cache_distinguishes_users(self, monkeypatch):
        """Different (username, realm, sip_user) tuples must NOT
        collide in the cache."""
        calls = []

        class _R:
            status_code = 200
            def __init__(self, ha1):
                self._ha1 = ha1
            def json(self):
                return {"ha1": self._ha1}

        def fake_get(url, params=None, **kw):
            ha1 = "h-" + (params or {}).get("username", "?")
            calls.append(ha1)
            return _R(ha1)

        monkeypatch.setattr(sip_stack.requests, "get", fake_get)
        sub = {"id": "s4", "base_url": "https://crm", "bearer_token": "t"}
        a = sip_stack._crm_login(sub, "alice", "r", "alice@x")
        b = sip_stack._crm_login(sub, "bob",   "r", "bob@x")
        assert a["ha1"] != b["ha1"]
        assert len(calls) == 2



class TestRegisterNonceRecovery:

    def test_unknown_nonce_rechallenges_with_fresh_401(self, monkeypatch):
        sip_stack._nonces.clear()
        sent = {}

        def fake_send(data, addr):
            sent["data"] = data
            sent["addr"] = addr

        monkeypatch.setattr(sip_stack, "_send", fake_send)
        monkeypatch.setattr(sip_stack, "_gen_tag", lambda: "tag-1")
        monkeypatch.setattr(sip_stack, "_rand_hex", lambda n: "fresh-nonce")

        msg = sip_stack._parse_sip(
            b"REGISTER sip:srv.launix.de SIP/2.0\r\n"
            b"Via: SIP/2.0/UDP 155.133.219.85:5060;branch=z9hG4bK-1\r\n"
            b"From: <sip:alice@crm.launix.de>;tag=from-1\r\n"
            b"To: <sip:alice@crm.launix.de>\r\n"
            b"Call-ID: cid-1\r\n"
            b"CSeq: 1 REGISTER\r\n"
            b"Contact: <sip:alice@155.133.219.85:5060>\r\n"
            b'Authorization: Digest username="alice@crm.launix.de", realm="crm.launix.de", nonce="stale-nonce", uri="sip:srv.launix.de", response="deadbeef"\r\n'
            b"Content-Length: 0\r\n"
            b"\r\n"
        )

        sip_stack._handle_inbound_register(msg, ("155.133.219.85", 5060))

        assert sent["addr"] == ("155.133.219.85", 5060)
        assert "SIP/2.0 401 Unauthorized" in sent["data"]
        assert 'WWW-Authenticate: Digest realm="crm.launix.de", nonce="fresh-nonce", algorithm=MD5' in sent["data"]
        assert sip_stack._nonces["fresh-nonce"] > 0
