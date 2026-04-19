"""Post-audit hardening: nonce atomicity, session entropy, /metrics."""
from __future__ import annotations

import threading

import pytest

from speech_pipeline.telephony import auth, call_state, leg, webclient


# ---------------------------------------------------------------------------
# Nonce validate is atomic (check-then-set under a lock)
# ---------------------------------------------------------------------------

class TestNonceAtomic:

    def test_concurrent_validate_only_one_wins(self):
        entry = auth.create_nonce("acc", "sub", "u1", ttl=60)
        nonce = entry["nonce"]

        results = []
        barrier = threading.Barrier(32)

        def _worker():
            barrier.wait()
            results.append(auth.validate_nonce(nonce))

        threads = [threading.Thread(target=_worker) for _ in range(32)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        wins = [r for r in results if r is not None]
        assert len(wins) == 1, f"Expected exactly one winner, got {len(wins)}"


# ---------------------------------------------------------------------------
# Session ID has ≥128 bit entropy (URL-safe base64 → len>=22)
# ---------------------------------------------------------------------------

class TestSessionIdEntropy:

    def test_generated_session_id_at_least_128_bit(self):
        call = call_state.create_call("sub", "acc", "pbx")
        try:
            sess = webclient.register_webclient(call, "u1", "n-" + "x" * 24)
            sid = sess["session_id"]
            assert sid.startswith("acc:wc-")
            assert len(sid.split(":", 1)[1]) - 3 >= 22
        finally:
            call_state.delete_call(call.call_id)


class TestScopedIds:

    @staticmethod
    def _scope(token: str) -> str:
        owner, local = token.split(":", 1)
        assert local.startswith(("call-", "leg-", "wc-", "n-"))
        return owner

    def test_call_ids_share_account_scope(self):
        c1 = call_state.create_call("sub-A", "acc-A", "pbx")
        c2 = call_state.create_call("sub-A", "acc-A", "pbx")
        c3 = call_state.create_call("sub-B", "acc-B", "pbx")
        try:
            assert self._scope(c1.call_id) == self._scope(c2.call_id)
            assert self._scope(c1.call_id) != self._scope(c3.call_id)
        finally:
            call_state.delete_call(c1.call_id)
            call_state.delete_call(c2.call_id)
            call_state.delete_call(c3.call_id)

    def test_leg_ids_share_account_scope(self):
        l1 = leg.create_leg("inbound", "+491", "pbx", "sub-A")
        l2 = leg.create_leg("outbound", "+492", "pbx", "sub-A")
        l3 = leg.create_leg("outbound", "+493", "pbx", "sub-B")
        try:
            assert self._scope(l1.leg_id) == self._scope(l2.leg_id)
            assert self._scope(l1.leg_id) != self._scope(l3.leg_id)
        finally:
            leg.delete_leg(l1.leg_id)
            leg.delete_leg(l2.leg_id)
            leg.delete_leg(l3.leg_id)

    def test_webclient_and_nonce_ids_are_account_scoped(self):
        call = call_state.create_call("sub-A", "acc-A", "pbx")
        try:
            n1 = auth.create_nonce("acc-A", "sub-A", "u1")["nonce"]
            n2 = auth.create_nonce("acc-A", "sub-A", "u1")["nonce"]
            n3 = auth.create_nonce("acc-B", "sub-B", "u2")["nonce"]
            s1 = webclient.register_webclient(call, "u1", n1)["session_id"]
            s2 = webclient.register_webclient(call, "u1", n2)["session_id"]
            call_b = call_state.create_call("sub-B", "acc-B", "pbx")
            s3 = webclient.register_webclient(call_b, "u2", n3)["session_id"]

            assert self._scope(n1) == self._scope(n2)
            assert self._scope(n1) != self._scope(n3)
            assert self._scope(s1) == self._scope(s2)
            assert self._scope(s1) != self._scope(s3)
        finally:
            call_state.delete_call(call.call_id)
            try:
                call_state.delete_call(call_b.call_id)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# /metrics endpoint exposes Prometheus-format gauges
# ---------------------------------------------------------------------------

class TestMetrics:

    def test_metrics_exposition_format(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "text/plain" in resp.headers.get("Content-Type", "")
        body = resp.get_data(as_text=True)
        for needle in (
            "speech_calls_total",
            "speech_legs_total",
            "speech_nonces_active",
            "speech_webclient_sessions",
            "# TYPE",
            "# HELP",
        ):
            assert needle in body, f"missing metric/header: {needle}"

    def test_metrics_reflects_active_call(self, client, account):
        from conftest import create_call
        call_id = create_call(client, account)
        try:
            body = client.get("/metrics").get_data(as_text=True)
            # At least one active call — the exact number depends on other
            # tests running, so just assert the count is > 0 for active.
            active_line = [l for l in body.splitlines()
                           if l.startswith('speech_calls_total{status="active"}')][0]
            count = int(active_line.split()[-1])
            assert count >= 1
        finally:
            client.delete(f"/api/calls/{call_id}", headers=account)

    def test_metrics_unauth_accessible(self, client):
        """Metrics must be scrapeable without bearer token
        (Prometheus pulls from localhost)."""
        resp = client.get("/metrics")
        assert resp.status_code == 200
