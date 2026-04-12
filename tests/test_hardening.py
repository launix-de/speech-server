"""Post-audit hardening: nonce atomicity, session entropy, /metrics."""
from __future__ import annotations

import threading

import pytest

from speech_pipeline.telephony import auth, call_state, webclient


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
            # wc- + token_urlsafe(16) → 16 bytes → 22 base64 chars
            assert sid.startswith("wc-")
            assert len(sid) - 3 >= 22
        finally:
            call_state.delete_call(call.call_id)


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
