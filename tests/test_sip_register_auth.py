"""SIP REGISTER digest path: HA1 cache + 3 s timeout.

The original 408 Request Timeout incident: ``_crm_login`` blocked
for up to 10 s on a slow CRM, longer than the SIP device's REGISTER
timeout.  Tests now enforce both the cache (5 min) and the per-call
3 s upper bound.
"""
from __future__ import annotations

import hashlib
import re
import time

import pytest

from speech_pipeline.telephony import sip_stack


def _register_msg(
    request_host: str,
    home_domain: str,
    *,
    route_host: str | None = None,
    user_agent: str | None = None,
    auth_header: str | None = None,
    auth_header_name: str = "Authorization",
    call_id: str = "cid",
    cseq: int = 1,
    contact_user: str = "alice",
) -> dict:
    lines = [
        f"REGISTER sip:{request_host} SIP/2.0",
        "Via: SIP/2.0/UDP 203.0.113.85:5060;rport;branch=z9hG4bK-test",
    ]
    if route_host:
        lines.append(f"Route: <sip:{route_host}:5061;lr>")
    lines.extend(
        [
            f"From: <sip:alice@{home_domain}>;tag=from-1",
            f"To: <sip:alice@{home_domain}>",
            f"Call-ID: {call_id}",
            f"CSeq: {cseq} REGISTER",
            f"Contact: <sip:{contact_user}@203.0.113.85:5060>",
        ]
    )
    if user_agent:
        lines.append(f"User-Agent: {user_agent}")
    if auth_header:
        lines.append(f"{auth_header_name}: {auth_header}")
    lines.extend(["Content-Length: 0", "", ""])
    msg = sip_stack._parse_sip("\r\n".join(lines).encode("utf-8"))
    msg["_source_addr"] = ("203.0.113.85", 5060)
    return msg


def _challenge_params(payload: str) -> dict:
    match = re.search(r"^WWW-Authenticate: (Digest .+)$", payload, re.MULTILINE)
    assert match, payload
    return sip_stack._parse_www_authenticate(match.group(1))


@pytest.fixture(autouse=True)
def _clear_ha1_cache():
    sip_stack._HA1_CACHE.clear()
    sip_stack._register_challenges.clear()
    yield
    sip_stack._HA1_CACHE.clear()
    sip_stack._register_challenges.clear()


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

    def test_register_without_auth_uses_registrar_challenge(self, monkeypatch):
        sip_stack._nonces.clear()
        sent = {}

        def fake_send(data, addr):
            sent["data"] = data
            sent["addr"] = addr

        monkeypatch.setattr(sip_stack, "_send", fake_send)
        monkeypatch.setattr(sip_stack, "_gen_tag", lambda: "tag-0")
        monkeypatch.setattr(sip_stack, "_rand_hex", lambda n: "first-nonce")

        msg = _register_msg(
            "tenant1.voice.example.net",
            "tenant1.voice.example.net",
            call_id="cid-0",
        )

        sip_stack._handle_inbound_register(msg, ("203.0.113.85", 5060))

        assert sent["addr"] == ("203.0.113.85", 5060)
        assert "SIP/2.0 401 Unauthorized" in sent["data"]
        assert "Contact: <sip:" in sent["data"]
        assert "Allow:" not in sent["data"]
        assert "User-Agent: tts-piper SIP/1.0" in sent["data"]
        # RFC 3581 §4: rport without a value in the request MUST be echoed
        # with the source port, and received MUST be populated — AVM
        # FRITZ!Box firmwares silently drop 401s that violate this.
        assert "rport=5060" in sent["data"]
        assert ";received=203.0.113.85" in sent["data"]
        assert "branch=z9hG4bK-test" in sent["data"]
        params = _challenge_params(sent["data"])
        assert params["realm"] == "tenant1.voice.example.net"
        assert params["nonce"] == "first-nonce"
        assert params["algorithm"] == "MD5"
        assert "qop" not in params
        assert "opaque" not in params

    @pytest.mark.parametrize(
        ("request_host", "home_domain", "route_host"),
        [
            ("crm01.example.net", "crm01.example.net", "edge01.example.net"),
            ("support.foo.example", "support.foo.example", "sip.foo.example"),
            ("north.division.example.org", "north.division.example.org", "pbx.example.org"),
        ],
    )
    def test_route_register_without_auth_uses_route_realm(
            self, monkeypatch, request_host, home_domain, route_host):
        sip_stack._nonces.clear()
        sent = {}

        def fake_send(data, addr):
            sent["data"] = data
            sent["addr"] = addr

        monkeypatch.setattr(sip_stack, "_send", fake_send)
        monkeypatch.setattr(sip_stack, "_gen_tag", lambda: "tag-p")
        monkeypatch.setattr(sip_stack, "_rand_hex", lambda n: "proxy-nonce")

        msg = _register_msg(
            request_host,
            home_domain,
            route_host=route_host,
            call_id="cid-p",
        )

        sip_stack._handle_inbound_register(msg, ("203.0.113.85", 5060))

        assert sent["addr"] == ("203.0.113.85", 5060)
        assert "SIP/2.0 401 Unauthorized" in sent["data"]
        assert "Contact: <sip:" in sent["data"]
        assert "User-Agent: tts-piper SIP/1.0" in sent["data"]
        assert "rport=5060" in sent["data"]
        assert ";received=203.0.113.85" in sent["data"]
        assert "branch=z9hG4bK-test" in sent["data"]
        params = _challenge_params(sent["data"])
        assert params["realm"] == home_domain
        assert params["nonce"] == "proxy-nonce"
        assert params["algorithm"] == "MD5"
        assert "qop" not in params
        assert "opaque" not in params
        assert f'realm="{route_host}"' not in sent["data"]
        assert "Allow:" not in sent["data"]

    def test_avm_route_register_without_auth_uses_request_realm(
            self, monkeypatch):
        sent = {}

        def fake_send(data, addr):
            sent["data"] = data
            sent["addr"] = addr

        monkeypatch.setattr(sip_stack, "_send", fake_send)
        monkeypatch.setattr(sip_stack, "_gen_tag", lambda: "tag-avm")
        monkeypatch.setattr(sip_stack, "_rand_hex", lambda n: "avm-nonce")

        msg = _register_msg(
            "crm.example.test",
            "crm.example.test",
            route_host="edge.example.test",
            user_agent="AVM FRITZ!Box 7430 test",
            call_id="cid-avm",
        )

        sip_stack._handle_inbound_register(msg, ("203.0.113.85", 5060))

        assert sent["addr"] == ("203.0.113.85", 5060)
        params = _challenge_params(sent["data"])
        assert params["realm"] == "crm.example.test"
        assert params["nonce"] == "avm-nonce"
        assert params["algorithm"] == "MD5"
        assert "qop" not in params
        assert "opaque" not in params
        assert 'realm="edge.example.test"' not in sent["data"]

    def test_register_challenge_adds_received_for_domain_via_without_rport(
            self, monkeypatch):
        sent = {}

        def fake_send(data, addr):
            sent["data"] = data
            sent["addr"] = addr

        monkeypatch.setattr(sip_stack, "_send", fake_send)
        monkeypatch.setattr(sip_stack, "_gen_tag", lambda: "tag-d")
        monkeypatch.setattr(sip_stack, "_rand_hex", lambda n: "domain-nonce")

        msg = sip_stack._parse_sip(
            "\r\n".join(
                [
                    "REGISTER sip:tenant1.voice.example.net SIP/2.0",
                    "Via: SIP/2.0/UDP sip-edge.example.net;branch=z9hG4bK-dom",
                    "From: <sip:alice@tenant1.voice.example.net>;tag=from-1",
                    "To: <sip:alice@tenant1.voice.example.net>",
                    "Call-ID: cid-dom",
                    "CSeq: 1 REGISTER",
                    "Content-Length: 0",
                    "",
                    "",
                ]
            ).encode("utf-8")
        )
        msg["_source_addr"] = ("203.0.113.85", 5060)

        sip_stack._handle_inbound_register(msg, ("203.0.113.85", 5060))

        assert sent["addr"] == ("203.0.113.85", 5060)
        assert "Via: SIP/2.0/UDP sip-edge.example.net;branch=z9hG4bK-dom" in sent["data"]
        assert "rport=" not in sent["data"]

    def test_unknown_nonce_rechallenges_with_fresh_401(self, monkeypatch):
        sip_stack._nonces.clear()
        sent = {}

        def fake_send(data, addr):
            sent["data"] = data
            sent["addr"] = addr

        monkeypatch.setattr(sip_stack, "_send", fake_send)
        monkeypatch.setattr(sip_stack, "_gen_tag", lambda: "tag-1")
        monkeypatch.setattr(sip_stack, "_rand_hex", lambda n: "fresh-nonce")

        msg = _register_msg(
            "tenant2.crm.example.net",
            "tenant2.crm.example.net",
            route_host="edge2.example.net",
            call_id="cid-1",
            auth_header=(
                'Digest username="alice@tenant2.crm.example.net", '
                'realm="tenant2.crm.example.net", nonce="stale-nonce", '
                'uri="sip:tenant2.crm.example.net", response="deadbeef"'
            ),
            auth_header_name="Proxy-Authorization",
        )

        sip_stack._handle_inbound_register(msg, ("203.0.113.85", 5060))

        assert sent["addr"] == ("203.0.113.85", 5060)
        assert "SIP/2.0 401 Unauthorized" in sent["data"]
        assert "Contact: <sip:" in sent["data"]
        assert "User-Agent: tts-piper SIP/1.0" in sent["data"]
        params = _challenge_params(sent["data"])
        assert params["realm"] == "tenant2.crm.example.net"
        assert params["nonce"] == "fresh-nonce"
        assert params["algorithm"] == "MD5"
        assert "qop" not in params
        assert "opaque" not in params
        assert sip_stack._nonces["fresh-nonce"] > 0

    def test_valid_auth_with_uncached_nonce_is_accepted(self, monkeypatch):
        sip_stack._nonces.clear()
        sip_stack._registrations.clear()
        sent = {}

        def fake_send(data, addr):
            sent["data"] = data
            sent["addr"] = addr

        monkeypatch.setattr(sip_stack, "_send", fake_send)
        monkeypatch.setattr(sip_stack, "_gen_tag", lambda: "tag-ok")
        monkeypatch.setattr(
            sip_stack,
            "_resolve_sip_identity",
            lambda uri: (
                "alice",
                "tenant3.crm.example.net",
                {"id": "sub-1", "base_url": "https://crm.example"},
            ),
        )
        ha1 = hashlib.md5(
            b"alice@tenant3.crm.example.net:tenant3.crm.example.net:secret"
        ).hexdigest()
        monkeypatch.setattr(
            sip_stack,
            "_crm_login",
            lambda subscriber, username, realm, raw_sip_user: {
                "ha1": ha1,
                "user_id": 42,
            },
        )

        nonce = "reused-nonce"
        uri = "sip:tenant3.crm.example.net"
        response = sip_stack._compute_digest_response(
            "alice@tenant3.crm.example.net",
            "secret",
            "tenant3.crm.example.net",
            nonce,
            "REGISTER",
            uri,
        )
        msg = _register_msg(
            "tenant3.crm.example.net",
            "tenant3.crm.example.net",
            call_id="cid-ok",
            auth_header=(
                'Digest username="alice@tenant3.crm.example.net", '
                'realm="tenant3.crm.example.net", '
                f'nonce="{nonce}", '
                f'uri="{uri}", '
                f'response="{response}", '
                "algorithm=MD5"
            ),
        )

        sip_stack._handle_inbound_register(msg, ("203.0.113.85", 5060))

        assert sent["addr"] == ("203.0.113.85", 5060)
        assert "SIP/2.0 200 OK" in sent["data"]
        assert sip_stack._nonces[nonce] > 0

    def test_register_challenge_preserves_request_uri_path(self, monkeypatch):
        sip_stack._nonces.clear()
        sent = {}

        monkeypatch.setattr(sip_stack, "_send", lambda data, addr: sent.update(
            data=data, addr=addr
        ))
        monkeypatch.setattr(sip_stack, "_gen_tag", lambda: "tag-path")
        monkeypatch.setattr(sip_stack, "_rand_hex", lambda n: "path-nonce")

        msg = _register_msg(
            "example.test/crm",
            "crm.example.test",
            call_id="cid-path",
        )

        sip_stack._handle_inbound_register(msg, ("203.0.113.85", 5060))

        assert sent["addr"] == ("203.0.113.85", 5060)
        params = _challenge_params(sent["data"])
        assert params["realm"] == "crm.example.test"
        assert params["nonce"] == "path-nonce"
        assert "qop" not in params
        assert "opaque" not in params

    def test_register_retransmit_reuses_same_challenge(self, monkeypatch):
        sip_stack._nonces.clear()
        sip_stack._register_challenges.clear()
        sent = []
        tags = iter(["tag-a", "tag-b"])
        challenge_parts = iter(["nonce-a", "nonce-b"])

        monkeypatch.setattr(sip_stack, "_send", lambda data, addr: sent.append(data))
        monkeypatch.setattr(sip_stack, "_gen_tag", lambda: next(tags))
        monkeypatch.setattr(sip_stack, "_rand_hex", lambda n: next(challenge_parts))

        msg = _register_msg(
            "hq.voice.example.net",
            "hq.voice.example.net",
            route_host="edge.voice.example.net",
            call_id="cid-r",
            cseq=7,
        )

        sip_stack._handle_inbound_register(msg, ("203.0.113.85", 5060))
        sip_stack._handle_inbound_register(msg, ("203.0.113.85", 5060))

        assert len(sent) == 2
        assert sent[0] == sent[1]
        assert 'nonce="nonce-a"' in sent[0]
        assert 'To: <sip:alice@hq.voice.example.net>;tag=tag-a' in sent[0]
