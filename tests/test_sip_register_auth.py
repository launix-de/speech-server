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
    contact_user: str | None = "alice",
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
        ]
    )
    if contact_user is not None:
        lines.append(f"Contact: <sip:{contact_user}@203.0.113.85:5060>")
    if user_agent:
        lines.append(f"User-Agent: {user_agent}")
    if auth_header:
        lines.append(f"{auth_header_name}: {auth_header}")
    lines.extend(["Content-Length: 0", "", ""])
    msg = sip_stack._parse_sip("\r\n".join(lines).encode("utf-8"))
    msg["_source_addr"] = ("203.0.113.85", 5060)
    return msg


def _challenge_params(payload: str, header_name: str = "WWW-Authenticate") -> dict:
    match = re.search(
        rf"^{re.escape(header_name)}: (Digest .+)$", payload, re.MULTILINE
    )
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
               "login_url": "https://crm.example/login",
               "bearer_token": "tok"}

        first = sip_stack._crm_login(sub, "alice", "realm", "alice@x")
        second = sip_stack._crm_login(sub, "alice", "realm", "alice@x")
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
            {"id": "s2", "base_url": "https://crm",
             "login_url": "https://crm/login", "bearer_token": "t"},
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
        sub = {"id": "s3", "base_url": "https://crm",
               "login_url": "https://crm/login", "bearer_token": "t"}
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
        sub = {"id": "s4", "base_url": "https://crm",
               "login_url": "https://crm/login", "bearer_token": "t"}
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
        # AVM FRITZ!Box firmwares need an Allow list and a Date header to
        # accept the 401 challenge; including them is RFC-correct anyway.
        assert "Allow:" in sent["data"]
        assert "Date:" in sent["data"]
        assert "User-Agent: tts-piper SIP/1.0" in sent["data"]
        # RFC 2617 — qop is required for FRITZ!Box auth retries.
        assert 'qop="auth"' in sent["data"]
        # ``stale`` must NOT be present on first challenge — AVM
        # interprets ``stale=false`` as "credentials were bad" and gives up.
        assert "stale=" not in sent["data"]
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
        # qop + opaque are required for AVM FRITZ!Box-compatible challenges.
        assert params["qop"] == "auth"
        assert params["opaque"]

    def test_register_challenge_keeps_legacy_local_contact_and_no_record_route(
            self, monkeypatch):
        """Keep the REGISTER 401 wire image close to the last green trace."""
        sent = {}

        monkeypatch.setattr(sip_stack, "_send", lambda data, addr: sent.update(
            data=data, addr=addr
        ))
        monkeypatch.setattr(sip_stack, "_gen_tag", lambda: "tag-wan")
        monkeypatch.setattr(sip_stack, "_rand_hex", lambda n: "wan-nonce")
        # Force stable addresses for the test assertions.
        monkeypatch.setattr(sip_stack, "_public_ip", "198.51.100.77")
        monkeypatch.setattr(sip_stack, "_local_ip", "10.0.0.42")
        monkeypatch.setattr(sip_stack, "_sip_port", 5061)

        raw = "\r\n".join([
            "REGISTER sip:crm.example.net SIP/2.0",
            "Via: SIP/2.0/UDP 203.0.113.85:5060;rport;branch=z9hG4bK-wan",
            "Route: <sip:sip.edge.example.net:5061;lr>",
            "From: <sip:alice@crm.example.net>;tag=from-wan",
            "To: <sip:alice@crm.example.net>",
            "Call-ID: cid-wan",
            "CSeq: 1 REGISTER",
            "Contact: <sip:alice@203.0.113.85:5060>",
            "User-Agent: AVM FRITZ!Box 7430 test",
            "Content-Length: 0",
            "",
            "",
        ])
        msg = sip_stack._parse_sip(raw.encode("utf-8"))
        msg["_source_addr"] = ("203.0.113.85", 5060)

        sip_stack._handle_inbound_register(msg, ("203.0.113.85", 5060))

        data = sent["data"]
        assert "SIP/2.0 401 Unauthorized" in data
        assert "Contact: <sip:10.0.0.42:5061>" in data
        assert "Record-Route:" not in data

    def test_register_challenge_uses_local_contact_for_fritzbox_compat(self, monkeypatch):
        sent = {}

        monkeypatch.setattr(sip_stack, "_send", lambda data, addr: sent.update(
            data=data, addr=addr
        ))
        monkeypatch.setattr(sip_stack, "_gen_tag", lambda: "tag-public")
        monkeypatch.setattr(sip_stack, "_rand_hex", lambda n: "public-nonce")
        monkeypatch.setattr(sip_stack, "_public_ip", "198.51.100.77")

        msg = _register_msg(
            "tenant1.voice.example.net",
            "tenant1.voice.example.net",
            call_id="cid-public",
        )

        sip_stack._handle_inbound_register(msg, ("203.0.113.85", 5060))

        assert f"Contact: <sip:{sip_stack._local_ip}:" in sent["data"]

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
        assert params.get("qop") == "auth"
        assert params.get("opaque")
        assert f'realm="{route_host}"' not in sent["data"]
        # AVM-compatible 401 includes Allow + Date.
        assert "Allow:" in sent["data"]
        assert "Date:" in sent["data"]

    def test_avm_route_register_without_auth_uses_request_realm(
            self, monkeypatch):
        sent = {}

        def fake_send(data, addr):
            sent["data"] = data
            sent["addr"] = addr

        monkeypatch.setattr(sip_stack, "_send", fake_send)
        monkeypatch.setattr(sip_stack, "_gen_tag", lambda: "tag-avm")
        monkeypatch.setattr(sip_stack, "_rand_hex", lambda n: "avm-nonce")

        raw = "\r\n".join(
            [
                "REGISTER sip:crm.example.test SIP/2.0",
                "Via: SIP/2.0/UDP 203.0.113.85:5060;rport;branch=z9hG4bK-test",
                "Route: <sip:edge.example.test:5061;lr>",
                "From: <sip:1@crm.example.test>;tag=from-1",
                "To: <sip:1@crm.example.test>",
                "Call-ID: cid-avm",
                "CSeq: 1 REGISTER",
                "Contact: <sip:1@203.0.113.85;uniq=DEVICE1>",
                "User-Agent: AVM FRITZ!Box 7430 test",
                "Content-Length: 0",
                "",
                "",
            ]
        )
        msg = sip_stack._parse_sip(raw.encode("utf-8"))
        msg["_source_addr"] = ("203.0.113.85", 5060)

        sip_stack._handle_inbound_register(msg, ("203.0.113.85", 5060))

        assert sent["addr"] == ("203.0.113.85", 5060)
        params = _challenge_params(sent["data"])
        assert params["realm"] == "crm.example.test"
        assert params["nonce"] == "avm-nonce"
        assert params["algorithm"] == "MD5"
        assert params.get("qop") == "auth"
        assert params.get("opaque")
        assert 'realm="edge.example.test"' not in sent["data"]

    def test_avm_named_identity_without_auth_uses_aor_domain_realm(
            self, monkeypatch):
        """AVM REGISTERs with a named (non-numeric) AOR user are
        challenged with the AOR domain as realm — NOT the local-part.

        An older branch returned the local-part (``realm="alice"``) as a
        workaround for a pre-01.04 bug that polluted the FritzBox
        credential cache. Freshly-configured AVM firmwares reject that
        nonsense realm and drop the 401 silently.
        """
        sent = {}

        def fake_send(data, addr):
            sent["data"] = data
            sent["addr"] = addr

        monkeypatch.setattr(sip_stack, "_send", fake_send)
        monkeypatch.setattr(sip_stack, "_gen_tag", lambda: "tag-avm-user")
        monkeypatch.setattr(sip_stack, "_rand_hex", lambda n: "avm-user-nonce")

        raw = "\r\n".join(
            [
                "REGISTER sip:crm.example.test SIP/2.0",
                "Via: SIP/2.0/UDP 203.0.113.85:5060;rport;branch=z9hG4bK-test",
                "Route: <sip:edge.example.test:5061;lr>",
                "From: <sip:alice@crm.example.test>;tag=from-1",
                "To: <sip:alice@crm.example.test>",
                "Call-ID: cid-avm-user",
                "CSeq: 1 REGISTER",
                "Contact: <sip:alice@203.0.113.85:5060>",
                "User-Agent: AVM FRITZ!Box 7430 test",
                "Content-Length: 0",
                "",
                "",
            ]
        )
        msg = sip_stack._parse_sip(raw.encode("utf-8"))
        msg["_source_addr"] = ("203.0.113.85", 5060)

        sip_stack._handle_inbound_register(msg, ("203.0.113.85", 5060))

        assert sent["addr"] == ("203.0.113.85", 5060)
        params = _challenge_params(sent["data"])
        assert params["realm"] == "crm.example.test"
        assert params["nonce"] == "avm-user-nonce"
        assert params["algorithm"] == "MD5"
        assert 'realm="alice"' not in sent["data"]
        assert 'realm="edge.example.test"' not in sent["data"]
        # RFC 3581 §4: the topmost Via MUST carry rport=<source-port> and
        # received=<source-ip> in the response when the request used rport
        # without a value — AVM FRITZ!Box silently drops 401s that violate
        # this, so register challenges never preserve the Via verbatim.
        assert "branch=z9hG4bK-test" in sent["data"]
        assert "rport=5060" in sent["data"]
        assert ";received=203.0.113.85" in sent["data"]

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
        assert params.get("qop") == "auth"
        assert params.get("opaque")
        assert sip_stack._nonces["fresh-nonce"] > 0

    def test_valid_auth_with_recent_challenge_nonce_is_accepted_when_nonce_cache_is_lost(
            self, monkeypatch):
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

        monkeypatch.setattr(sip_stack, "_rand_hex", lambda n: "recent-nonce")

        first = _register_msg(
            "tenant3.crm.example.net",
            "tenant3.crm.example.net",
            call_id="cid-ok",
        )
        sip_stack._handle_inbound_register(first, ("203.0.113.85", 5060))
        assert 'nonce="recent-nonce"' in sent["data"]

        sip_stack._nonces.clear()
        uri = "sip:tenant3.crm.example.net"
        response = sip_stack._compute_digest_response(
            "alice@tenant3.crm.example.net",
            "secret",
            "tenant3.crm.example.net",
            "recent-nonce",
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
                'nonce="recent-nonce", '
                f'uri="{uri}", '
                f'response="{response}", '
                "algorithm=MD5"
            ),
        )

        sip_stack._handle_inbound_register(msg, ("203.0.113.85", 5060))

        assert sent["addr"] == ("203.0.113.85", 5060)
        assert "SIP/2.0 200 OK" in sent["data"]
        assert sip_stack._nonces["recent-nonce"] > 0

    def test_valid_auth_with_unknown_uncached_nonce_is_accepted_in_interop_mode(
            self, monkeypatch):
        sip_stack._nonces.clear()
        sip_stack._registrations.clear()
        sent = {}

        monkeypatch.setattr(sip_stack, "_send", lambda data, addr: sent.update(
            data=data, addr=addr
        ))
        monkeypatch.setattr(sip_stack, "_gen_tag", lambda: "tag-unknown")
        monkeypatch.setattr(sip_stack, "_rand_hex", lambda n: "fresh-nonce")
        monkeypatch.setattr(
            sip_stack,
            "_resolve_sip_identity",
            lambda uri: (
                "alice",
                "tenant4.crm.example.net",
                {"id": "sub-2", "base_url": "https://crm.example"},
            ),
        )
        ha1 = hashlib.md5(
            b"alice@tenant4.crm.example.net:tenant4.crm.example.net:secret"
        ).hexdigest()
        monkeypatch.setattr(
            sip_stack,
            "_crm_login",
            lambda subscriber, username, realm, raw_sip_user: {
                "ha1": ha1,
                "user_id": 42,
            },
        )

        uri = "sip:tenant4.crm.example.net"
        response = sip_stack._compute_digest_response(
            "alice@tenant4.crm.example.net",
            "secret",
            "tenant4.crm.example.net",
            "forged-nonce",
            "REGISTER",
            uri,
        )
        msg = _register_msg(
            "tenant4.crm.example.net",
            "tenant4.crm.example.net",
            call_id="cid-unknown",
            auth_header=(
                'Digest username="alice@tenant4.crm.example.net", '
                'realm="tenant4.crm.example.net", '
                'nonce="forged-nonce", '
                f'uri="{uri}", '
                f'response="{response}", '
                "algorithm=MD5"
            ),
        )

        sip_stack._handle_inbound_register(msg, ("203.0.113.85", 5060))

        assert sent["addr"] == ("203.0.113.85", 5060)
        assert "SIP/2.0 200 OK" in sent["data"]
        assert sip_stack._nonces["forged-nonce"] > 0

    def test_invalid_auth_with_unknown_uncached_nonce_is_rechallenged(
            self, monkeypatch):
        sip_stack._nonces.clear()
        sip_stack._registrations.clear()
        sent = {}

        monkeypatch.setattr(sip_stack, "_send", lambda data, addr: sent.update(
            data=data, addr=addr
        ))
        monkeypatch.setattr(sip_stack, "_gen_tag", lambda: "tag-bad-unknown")
        monkeypatch.setattr(sip_stack, "_rand_hex", lambda n: "fresh-nonce")
        monkeypatch.setattr(
            sip_stack,
            "_resolve_sip_identity",
            lambda uri: (
                "alice",
                "tenant4.crm.example.net",
                {"id": "sub-2", "base_url": "https://crm.example"},
            ),
        )
        ha1 = hashlib.md5(
            b"alice@tenant4.crm.example.net:tenant4.crm.example.net:secret"
        ).hexdigest()
        monkeypatch.setattr(
            sip_stack,
            "_crm_login",
            lambda subscriber, username, realm, raw_sip_user: {
                "ha1": ha1,
                "user_id": 42,
            },
        )

        uri = "sip:tenant4.crm.example.net"
        msg = _register_msg(
            "tenant4.crm.example.net",
            "tenant4.crm.example.net",
            call_id="cid-bad-unknown",
            auth_header=(
                'Digest username="alice@tenant4.crm.example.net", '
                'realm="tenant4.crm.example.net", '
                'nonce="forged-nonce", '
                f'uri="{uri}", '
                'response="deadbeefdeadbeefdeadbeefdeadbeef", '
                "algorithm=MD5"
            ),
        )

        sip_stack._handle_inbound_register(msg, ("203.0.113.85", 5060))

        assert sent["addr"] == ("203.0.113.85", 5060)
        assert "SIP/2.0 401 Unauthorized" in sent["data"]
        assert 'nonce="fresh-nonce"' in sent["data"]
        assert "forged-nonce" not in sip_stack._nonces

    def test_valid_auth_with_reused_nonce_inside_timeout_is_accepted(self, monkeypatch):
        sip_stack._nonces.clear()
        sip_stack._registrations.clear()
        sent = {}

        monkeypatch.setattr(sip_stack, "_send", lambda data, addr: sent.update(
            data=data, addr=addr
        ))
        monkeypatch.setattr(sip_stack, "_gen_tag", lambda: "tag-reuse")
        monkeypatch.setattr(
            sip_stack,
            "_resolve_sip_identity",
            lambda uri: (
                "alice",
                "tenant7.crm.example.net",
                {"id": "sub-7", "base_url": "https://crm.example"},
            ),
        )
        ha1 = hashlib.md5(
            b"alice@tenant7.crm.example.net:tenant7.crm.example.net:secret"
        ).hexdigest()
        monkeypatch.setattr(
            sip_stack,
            "_crm_login",
            lambda subscriber, username, realm, raw_sip_user: {
                "ha1": ha1,
                "user_id": 42,
            },
        )
        monkeypatch.setattr(
            sip_stack.time,
            "time",
            lambda: 5000.0,
        )

        sip_stack._nonces["reused-nonce"] = 5000.0 - 1800.0
        uri = "sip:tenant7.crm.example.net"
        response = sip_stack._compute_digest_response(
            "alice@tenant7.crm.example.net",
            "secret",
            "tenant7.crm.example.net",
            "reused-nonce",
            "REGISTER",
            uri,
        )
        msg = _register_msg(
            "tenant7.crm.example.net",
            "tenant7.crm.example.net",
            call_id="cid-reuse",
            auth_header=(
                'Digest username="alice@tenant7.crm.example.net", '
                'realm="tenant7.crm.example.net", '
                'nonce="reused-nonce", '
                f'uri="{uri}", '
                f'response="{response}", '
                "algorithm=MD5"
            ),
        )

        sip_stack._handle_inbound_register(msg, ("203.0.113.85", 5060))

        assert "SIP/2.0 200 OK" in sent["data"]

    def test_cleanup_expires_register_nonce_after_reuse_timeout(self, monkeypatch):
        sip_stack._nonces.clear()
        sip_stack._register_challenges.clear()
        sip_stack._running = True
        sip_stack._nonces["expired-register-nonce"] = 5000.0 - (
            sip_stack._REGISTER_NONCE_REUSE_TTL + 1
        )

        steps = iter([None, None])

        def fake_sleep(_seconds):
            sip_stack._running = False
            next(steps, None)

        monkeypatch.setattr(sip_stack.time, "time", lambda: 5000.0)
        monkeypatch.setattr(sip_stack.time, "sleep", fake_sleep)

        sip_stack._cleanup_loop()

        assert "expired-register-nonce" not in sip_stack._nonces

    def test_resolve_sip_identity_accepts_plus_mangled_username(self, monkeypatch):
        subscriber = {"id": "sub-plus", "base_url": "https://example.net/app/crm-next"}
        monkeypatch.setattr(
            sip_stack.sub_mod,
            "find_by_base_url",
            lambda base_url: subscriber if base_url == "https://example.net/app/crm-next" else None,
        )

        username, realm, found = sip_stack._resolve_sip_identity(
            "sip:alice+example.net~app~crm-next@sip.edge.example.net"
        )

        assert username == "alice"
        assert realm == "example.net/app/crm-next"
        assert found == subscriber

    def test_resolve_sip_identity_falls_back_from_transport_host_to_unique_subscriber(
            self, monkeypatch):
        subscriber = {"id": "sub-solo", "base_url": "https://example.net/crm"}
        monkeypatch.setattr(
            sip_stack.sub_mod,
            "find_by_registration_target",
            lambda target: None,
        )
        monkeypatch.setattr(
            sip_stack.sub_mod,
            "find_unique_subscriber",
            lambda: subscriber,
        )

        username, realm, found = sip_stack._resolve_sip_identity(
            "sip:alice@sip.edge.example.net"
        )

        assert username == "alice"
        assert realm == "crm.example.net"
        assert found == subscriber

    def test_extract_host_handles_ipv6_uris(self):
        assert sip_stack._extract_host("sip:alice@[2001:db8::1]") == "2001:db8::1"
        assert sip_stack._extract_host("sip:[2001:db8::2]") == "2001:db8::2"
        assert sip_stack._extract_host("sip:alice@[2001:db8::3]:5061") == "2001:db8::3"

    def test_register_accepts_proxy_authorization_header(self, monkeypatch):
        sip_stack._nonces.clear()
        sent = []

        monkeypatch.setattr(sip_stack, "_send", lambda data, addr: sent.append((data, addr)))
        monkeypatch.setattr(sip_stack, "_gen_tag", lambda: "tag-proxy")
        monkeypatch.setattr(sip_stack, "_rand_hex", lambda n: "proxy-auth-nonce")

        msg = _register_msg(
            "tenant5.crm.example.net",
            "tenant5.crm.example.net",
            call_id="cid-proxy",
        )
        sip_stack._handle_inbound_register(msg, ("203.0.113.85", 5060))

        challenge = sent[-1][0]
        params = _challenge_params(challenge)
        auth = sip_stack._build_authorization(
            "Proxy-Authorization",
            "alice@tenant5.crm.example.net",
            "secret",
            params,
            "REGISTER",
            "sip:tenant5.crm.example.net",
        )
        monkeypatch.setattr(
            sip_stack,
            "_resolve_sip_identity",
            lambda uri: (
                "alice",
                "tenant5.crm.example.net",
                {"id": "sub-5", "base_url": "https://crm.example"},
            ),
        )
        ha1 = hashlib.md5(
            b"alice@tenant5.crm.example.net:tenant5.crm.example.net:secret"
        ).hexdigest()
        monkeypatch.setattr(
            sip_stack,
            "_crm_login",
            lambda subscriber, username, realm, raw_sip_user: {
                "ha1": ha1,
                "user_id": 42,
            },
        )
        retry = _register_msg(
            "tenant5.crm.example.net",
            "tenant5.crm.example.net",
            call_id="cid-proxy",
            auth_header=auth,
            auth_header_name="Proxy-Authorization",
        )

        sip_stack._handle_inbound_register(retry, ("203.0.113.85", 5060))

        assert "SIP/2.0 200 OK" in sent[-1][0]

    def test_plain_auth_username_does_not_override_canonical_aor(
            self, monkeypatch):
        sip_stack._nonces.clear()
        sip_stack._registrations.clear()
        sent = []

        monkeypatch.setattr(sip_stack, "_send", lambda data, addr: sent.append((data, addr)))
        monkeypatch.setattr(sip_stack, "_gen_tag", lambda: "tag-plain-user")
        monkeypatch.setattr(
            sip_stack,
            "_resolve_sip_identity",
            lambda uri: (
                "alice",
                "tenant8.crm.example.net",
                {"id": "sub-8", "base_url": "https://crm.example"},
            ),
        )
        ha1 = hashlib.md5(
            b"alice:tenant8.crm.example.net:secret"
        ).hexdigest()
        monkeypatch.setattr(
            sip_stack,
            "_crm_login",
            lambda subscriber, username, realm, raw_sip_user: {
                "ha1": ha1,
                "user_id": 42,
            },
        )

        sip_stack._nonces["plain-user-nonce"] = time.time()
        msg = _register_msg(
            "tenant8.crm.example.net",
            "tenant8.crm.example.net",
            call_id="cid-plain-user",
            auth_header=(
                'Digest username="alice", '
                'realm="tenant8.crm.example.net", '
                'nonce="plain-user-nonce", '
                'uri="sip:tenant8.crm.example.net", '
                f'response="{sip_stack._compute_digest_response("alice", "secret", "tenant8.crm.example.net", "plain-user-nonce", "REGISTER", "sip:tenant8.crm.example.net")}", '
                "algorithm=MD5"
            ),
        )

        sip_stack._handle_inbound_register(msg, ("203.0.113.85", 5060))

        assert "SIP/2.0 200 OK" in sent[-1][0]
        regs = sip_stack.get_registrations("alice@tenant8.crm.example.net")
        assert len(regs) == 1

    def test_register_without_contact_and_auth_is_challenged(
            self, monkeypatch):
        """A contact-less REGISTER without credentials is challenged with
        401 like any other REGISTER.
        """
        sip_stack._nonces.clear()
        sip_stack._registrations.clear()
        sent = []

        monkeypatch.setattr(sip_stack, "_send", lambda data, addr: sent.append((data, addr)))
        monkeypatch.setattr(sip_stack, "_gen_tag", lambda: "tag-contact")
        monkeypatch.setattr(sip_stack, "_rand_hex", lambda n: "probe-nonce")
        monkeypatch.setattr(
            sip_stack,
            "_resolve_sip_identity",
            lambda uri: (
                "alice",
                "tenant6.crm.example.net",
                {"id": "sub-6", "base_url": "https://crm.example"},
            ),
        )

        probe = _register_msg(
            "tenant6.crm.example.net",
            "tenant6.crm.example.net",
            call_id="cid-contact",
            contact_user=None,
        )
        sip_stack._handle_inbound_register(probe, ("203.0.113.85", 5060))

        reply = sent[-1][0]
        assert "SIP/2.0 401 Unauthorized" in reply
        params = _challenge_params(reply)
        assert params["realm"] == "tenant6.crm.example.net"
        assert params["nonce"] == "probe-nonce"
        assert sip_stack.get_registrations("alice@tenant6.crm.example.net") == []

    def test_register_without_contact_with_auth_falls_back_to_addr_contact(
            self, monkeypatch):
        """Contact-less REGISTER that carries credentials is treated as a
        real registration; the server falls back to an addr-derived
        contact URI (client intends to register but omitted Contact)."""
        sip_stack._nonces.clear()
        sip_stack._registrations.clear()
        sent = []

        monkeypatch.setattr(sip_stack, "_send", lambda data, addr: sent.append((data, addr)))
        monkeypatch.setattr(sip_stack, "_gen_tag", lambda: "tag-contact")
        monkeypatch.setattr(sip_stack, "_rand_hex", lambda n: "contact-nonce")
        monkeypatch.setattr(
            sip_stack,
            "_resolve_sip_identity",
            lambda uri: (
                "alice",
                "tenant6.crm.example.net",
                {"id": "sub-6", "base_url": "https://crm.example"},
            ),
        )
        ha1 = hashlib.md5(
            b"alice@tenant6.crm.example.net:tenant6.crm.example.net:secret"
        ).hexdigest()
        monkeypatch.setattr(
            sip_stack,
            "_crm_login",
            lambda subscriber, username, realm, raw_sip_user: {
                "ha1": ha1,
                "user_id": 42,
            },
        )

        # Seed a nonce so the real register leg can verify against it
        # without going through the query path first.
        nonce = "contact-nonce"
        sip_stack._nonces[nonce] = time.time()
        challenge_params = {
            "realm": "tenant6.crm.example.net",
            "nonce": nonce,
            "algorithm": "MD5",
        }

        auth = sip_stack._build_authorization(
            "Authorization",
            "alice@tenant6.crm.example.net",
            "secret",
            challenge_params,
            "REGISTER",
            "sip:tenant6.crm.example.net",
        )
        retry = _register_msg(
            "tenant6.crm.example.net",
            "tenant6.crm.example.net",
            call_id="cid-contact",
            contact_user=None,
            auth_header=auth,
        )
        sip_stack._handle_inbound_register(retry, ("203.0.113.85", 5060))

        regs = sip_stack.get_registrations("alice@tenant6.crm.example.net")
        assert len(regs) == 1
        assert regs[0]["contact_uri"] == "sip:alice@203.0.113.85:5060"

    def test_register_to_transport_host_is_challenged_with_unique_subscriber_realm(
            self, monkeypatch):
        sent = []
        monkeypatch.setattr(sip_stack, "_send", lambda data, addr: sent.append((data, addr)))
        monkeypatch.setattr(sip_stack, "_gen_tag", lambda: "tag-transport")
        monkeypatch.setattr(sip_stack, "_rand_hex", lambda n: "transport-nonce")
        monkeypatch.setattr(
            sip_stack.sub_mod,
            "find_by_registration_target",
            lambda target: None,
        )
        monkeypatch.setattr(
            sip_stack.sub_mod,
            "find_unique_subscriber",
            lambda: {"id": "sub-solo", "base_url": "https://example.net/crm"},
        )

        msg = _register_msg(
            "sip.edge.example.net",
            "sip.edge.example.net",
            route_host="sip.edge.example.net",
            user_agent="AVM FRITZ!Box 7430 test",
            call_id="cid-transport",
            contact_user=None,
        )

        sip_stack._handle_inbound_register(msg, ("203.0.113.85", 5060))

        reply = sent[-1][0]
        assert "SIP/2.0 401 Unauthorized" in reply
        params = _challenge_params(reply)
        assert params["realm"] == "crm.example.net"

    def test_transport_host_with_plus_encoded_crm_uses_transport_realm(
            self, monkeypatch):
        sent = []
        monkeypatch.setattr(sip_stack, "_send", lambda data, addr: sent.append((data, addr)))
        monkeypatch.setattr(sip_stack, "_gen_tag", lambda: "tag-transport-plus")
        monkeypatch.setattr(sip_stack, "_rand_hex", lambda n: "transport-plus-nonce")

        raw = "\r\n".join(
            [
                "REGISTER sip:sip.edge.example.net SIP/2.0",
                "Via: SIP/2.0/UDP 203.0.113.85:5060;rport;branch=z9hG4bK-test",
                "Route: <sip:sip.edge.example.net:5061;lr>",
                "From: <sip:alice+crm.example.net@sip.edge.example.net>;tag=from-plus",
                "To: <sip:alice+crm.example.net@sip.edge.example.net>",
                "Call-ID: cid-transport-plus",
                "CSeq: 1 REGISTER",
                "User-Agent: AVM FRITZ!Box 7430 test",
                "Content-Length: 0",
                "",
                "",
            ]
        )
        msg = sip_stack._parse_sip(raw.encode("utf-8"))
        msg["_source_addr"] = ("203.0.113.85", 5060)

        sip_stack._handle_inbound_register(msg, ("203.0.113.85", 5060))

        reply = sent[-1][0]
        assert "SIP/2.0 401 Unauthorized" in reply
        params = _challenge_params(reply)
        assert params["realm"] == "sip.edge.example.net"

    def test_transport_host_contact_register_uses_registrar_challenge(
            self, monkeypatch):
        sent = []
        monkeypatch.setattr(sip_stack, "_send", lambda data, addr: sent.append((data, addr)))
        monkeypatch.setattr(sip_stack, "_gen_tag", lambda: "tag-transport-register")
        monkeypatch.setattr(sip_stack, "_rand_hex", lambda n: "transport-register-nonce")

        raw = "\r\n".join(
            [
                "REGISTER sip:sip.edge.example.net SIP/2.0",
                "Via: SIP/2.0/UDP 203.0.113.85:5060;rport;branch=z9hG4bK-test",
                "Route: <sip:sip.edge.example.net:5061;lr>",
                "From: <sip:alice+crm.example.net@sip.edge.example.net>;tag=from-plus",
                "To: <sip:alice+crm.example.net@sip.edge.example.net>",
                "Call-ID: cid-transport-register",
                "CSeq: 2 REGISTER",
                "Contact: <sip:alice+crm.example.net@203.0.113.85:5060>",
                "Expires: 1800",
                "User-Agent: AVM FRITZ!Box 7430 test",
                "Content-Length: 0",
                "",
                "",
            ]
        )
        msg = sip_stack._parse_sip(raw.encode("utf-8"))
        msg["_source_addr"] = ("203.0.113.85", 5060)

        sip_stack._handle_inbound_register(msg, ("203.0.113.85", 5060))

        reply = sent[-1][0]
        assert "SIP/2.0 401 Unauthorized" in reply
        params = _challenge_params(reply)
        assert params["realm"] == "sip.edge.example.net"
        assert params.get("qop") == "auth"
        assert params.get("opaque")
        assert "Proxy-Authenticate:" not in reply
        assert "User-Agent: tts-piper SIP/1.0" in reply

    def test_fritzbox_contact_less_register_is_challenged(
        self, monkeypatch
    ):
        """A contact-less REGISTER from a device is challenged with 401."""
        sent = []

        monkeypatch.setattr(sip_stack, "_send", lambda data, addr: sent.append((data, addr)))
        monkeypatch.setattr(sip_stack, "_gen_tag", lambda: "tag-refresh")
        monkeypatch.setattr(sip_stack, "_rand_hex", lambda n: "refresh-nonce")

        raw = "\r\n".join(
            [
                "REGISTER sip:crm.example.net SIP/2.0",
                "Via: SIP/2.0/UDP 203.0.113.85:5060;rport;branch=z9hG4bK-test",
                "From: <sip:1@crm.example.net>;tag=from-1",
                "To: <sip:1@crm.example.net>",
                "Call-ID: cid-refresh",
                "CSeq: 2 REGISTER",
                "User-Agent: AVM FRITZ!Box 7430 test",
                "Content-Length: 0",
                "",
                "",
            ]
        )
        msg = sip_stack._parse_sip(raw.encode("utf-8"))
        msg["_source_addr"] = ("203.0.113.85", 5060)

        sip_stack._handle_inbound_register(msg, ("203.0.113.85", 5060))

        assert "SIP/2.0 401 Unauthorized" in sent[-1][0]

    def test_fritzbox_numeric_identity_without_auth_is_challenged(self, monkeypatch):
        sent = []

        monkeypatch.setattr(sip_stack, "_send", lambda data, addr: sent.append((data, addr)))
        monkeypatch.setattr(sip_stack, "_gen_tag", lambda: "tag-bootstrap")
        monkeypatch.setattr(
            sip_stack.sub_mod,
            "find_by_sip_domain",
            lambda domain: {
                "id": "sub-boot",
                "base_url": "https://example.net/crm",
            } if domain == "crm.example.net" else None,
        )

        raw = "\r\n".join(
            [
                "REGISTER sip:crm.example.net SIP/2.0",
                "Via: SIP/2.0/UDP 203.0.113.85:5060;rport;branch=z9hG4bK-test",
                "Route: <sip:sip.edge.example.net:5061;lr>",
                "From: <sip:1@crm.example.net>;tag=from-1",
                "To: <sip:1@crm.example.net>",
                "Call-ID: cid-bootstrap",
                "CSeq: 1 REGISTER",
                "Contact: <sip:1@203.0.113.85;uniq=DEVICE1>",
                "Expires: 1800",
                "User-Agent: AVM FRITZ!Box 7430 test",
                "Content-Length: 0",
                "",
                "",
            ]
        )
        msg = sip_stack._parse_sip(raw.encode("utf-8"))
        msg["_source_addr"] = ("203.0.113.85", 5060)

        sip_stack._handle_inbound_register(msg, ("203.0.113.85", 5060))

        assert "SIP/2.0 401 Unauthorized" in sent[-1][0]
        params = _challenge_params(sent[-1][0])
        assert params["realm"] == "crm.example.net"
        assert sip_stack.get_registrations("1@crm.example.net") == []

    def test_fritzbox_unauthenticated_retransmit_reuses_same_401(self, monkeypatch):
        sent = []

        monkeypatch.setattr(sip_stack, "_send", lambda data, addr: sent.append((data, addr)))
        monkeypatch.setattr(sip_stack, "_gen_tag", lambda: "tag-once")
        monkeypatch.setattr(sip_stack, "_public_ip", "198.51.100.77")
        monkeypatch.setattr(
            sip_stack.sub_mod,
            "find_by_sip_domain",
            lambda domain: {
                "id": "sub-boot",
                "base_url": "https://example.net/crm",
            } if domain == "crm.example.net" else None,
        )

        raw = "\r\n".join(
            [
                "REGISTER sip:crm.example.net SIP/2.0",
                "Via: SIP/2.0/UDP 203.0.113.85:5060;rport;branch=z9hG4bK-test",
                "Route: <sip:sip.edge.example.net:5061;lr>",
                "From: <sip:1@crm.example.net>;tag=from-1",
                "To: <sip:1@crm.example.net>",
                "Call-ID: cid-bootstrap-repeat",
                "CSeq: 1 REGISTER",
                "Contact: <sip:1@203.0.113.85;uniq=DEVICE1>",
                "Expires: 1800",
                "User-Agent: AVM FRITZ!Box 7430 test",
                "Content-Length: 0",
                "",
                "",
            ]
        )
        msg = sip_stack._parse_sip(raw.encode("utf-8"))
        msg["_source_addr"] = ("203.0.113.85", 5060)

        sip_stack._handle_inbound_register(msg, ("203.0.113.85", 5060))
        sip_stack._handle_inbound_register(msg, ("203.0.113.85", 5060))

        assert "SIP/2.0 401 Unauthorized" in sent[0][0]
        assert sent[0][0] == sent[1][0]
        assert "To: <sip:1@crm.example.net>;tag=tag-once" in sent[0][0]
        assert 'WWW-Authenticate: Digest realm="crm.example.net"' in sent[0][0]

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
        assert params.get("qop") == "auth"
        assert params.get("opaque")

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
