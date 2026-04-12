"""HTTP response-latency guardrails.

The CRM is PHP and holds a session lock for the duration of every
outbound HTTP call.  If ``POST /api/pipelines`` takes >1 s to return,
follow-up CRM requests stall — the user sees hangs on inbound/outbound
calls.  These tests enforce a hard latency budget on the server.
"""
from __future__ import annotations

import json
import time
from unittest.mock import MagicMock

import pytest

from conftest import SUBSCRIBER_ID, create_call
from speech_pipeline.telephony import leg as leg_mod


LATENCY_BUDGET_MS = 500   # CRM can tolerate at most this per call


def _elapsed_ms(fn):
    t0 = time.monotonic()
    r = fn()
    return r, (time.monotonic() - t0) * 1000


class TestPipelinePostLatency:

    def test_originate_returns_under_budget(self, client, account, monkeypatch):
        """``originate:NUM -> call:C`` must return the leg_id before
        the async build finishes.  PHP blocks on this round-trip."""
        # Stop originate_only from dialing for real.
        monkeypatch.setattr(leg_mod, "originate_only", lambda leg, pbx: None)

        call_id = create_call(client, account)
        try:
            dsl = ('originate:+491234567890{"ringing":"/cb"} '
                   f'-> call:{call_id}')
            (resp, elapsed) = _elapsed_ms(lambda: client.post(
                "/api/pipelines",
                data=json.dumps({"dsl": dsl}),
                headers=account,
            ))
            assert resp.status_code == 201, resp.data
            assert elapsed < LATENCY_BUDGET_MS, (
                f"originate took {elapsed:.0f}ms > {LATENCY_BUDGET_MS}ms budget"
            )
            body = resp.get_json()
            assert "leg_id" in body, f"leg_id missing: {body}"
            assert body["leg_id"].startswith("leg-")
        finally:
            client.delete(f"/api/calls/{call_id}", headers=account)

    def test_bridge_returns_under_budget(self, client, account, monkeypatch):
        """``sip:LEG -> call:C -> sip:LEG`` (the hot path for inbound)."""
        # Register a fake leg the DSL can resolve.
        fake_voip = MagicMock()
        fake_voip.RTPClients = []
        leg = leg_mod.create_leg(
            direction="inbound", number="+49174000",
            pbx_id="TestPBX", subscriber_id=SUBSCRIBER_ID,
            voip_call=fake_voip,
        )
        try:
            call_id = create_call(client, account)
            dsl = f"sip:{leg.leg_id} -> call:{call_id} -> sip:{leg.leg_id}"
            (resp, elapsed) = _elapsed_ms(lambda: client.post(
                "/api/pipelines",
                data=json.dumps({"dsl": dsl}),
                headers=account,
            ))
            assert resp.status_code == 201, resp.data
            assert elapsed < LATENCY_BUDGET_MS, (
                f"bridge took {elapsed:.0f}ms > {LATENCY_BUDGET_MS}ms budget"
            )
            client.delete(f"/api/calls/{call_id}", headers=account)
        finally:
            leg_mod._legs.pop(leg.leg_id, None)

    def test_play_returns_under_budget(self, client, account):
        call_id = create_call(client, account)
        try:
            dsl = f'play:hold{{"url":"examples/queue.mp3","loop":true}} -> call:{call_id}'
            (resp, elapsed) = _elapsed_ms(lambda: client.post(
                "/api/pipelines",
                data=json.dumps({"dsl": dsl}),
                headers=account,
            ))
            assert resp.status_code == 201
            assert elapsed < LATENCY_BUDGET_MS, (
                f"play-pipeline took {elapsed:.0f}ms > "
                f"{LATENCY_BUDGET_MS}ms budget"
            )
        finally:
            client.delete(f"/api/calls/{call_id}", headers=account)

    def test_delete_pipelines_works_with_query_string(
            self, client, account):
        """PHP curl with CURLOPT_CUSTOMREQUEST ``DELETE`` drops the body
        unless CURLOPT_POSTFIELDS is set.  The server MUST accept
        ``?dsl=...`` as a fallback so wait-music stopping doesn't 400."""
        call_id = create_call(client, account)
        try:
            client.post(
                "/api/pipelines",
                data=json.dumps({
                    "dsl": f'play:{call_id}_wait'
                           f'{{"url":"examples/queue.mp3","loop":true}} '
                           f'-> call:{call_id}',
                }),
                headers=account,
            )
            # DELETE with query-string only (no body) — simulates
            # the PHP-curl bug.
            import urllib.parse as _u
            q = _u.quote(f"play:{call_id}_wait")
            resp = client.delete(f"/api/pipelines?dsl={q}", headers=account)
            assert resp.status_code == 204, (
                f"DELETE with query-string fallback failed "
                f"(code {resp.status_code}, body {resp.data!r})"
            )
        finally:
            client.delete(f"/api/calls/{call_id}", headers=account)

    def test_stage_is_killable_immediately_after_create(
            self, client, account):
        """Regression: a quick ``POST play → DELETE play`` sequence
        (the CRM wait-music lifecycle) used to 404 on DELETE because
        the async build thread hadn't registered the stage yet."""
        call_id = create_call(client, account)
        try:
            # POST wait-music pipeline.
            resp = client.post(
                "/api/pipelines",
                data=json.dumps({
                    "dsl": f'play:{call_id}_wait'
                           f'{{"url":"examples/queue.mp3","loop":true}} '
                           f'-> call:{call_id}',
                }),
                headers=account,
            )
            assert resp.status_code == 201

            # Immediately (no sleep) DELETE it — this is the exact
            # sequence the CRM emits when answering an inbound leg.
            resp = client.delete(
                "/api/pipelines",
                data=json.dumps({"dsl": f"play:{call_id}_wait"}),
                headers=account,
            )
            assert resp.status_code == 204, (
                f"Stage unkillable right after creation "
                f"(POST /api/pipelines build is probably still async). "
                f"Response: {resp.data!r}"
            )

            # And a second DELETE should now return 404 (already gone).
            resp = client.delete(
                "/api/pipelines",
                data=json.dumps({"dsl": f"play:{call_id}_wait"}),
                headers=account,
            )
            assert resp.status_code == 404
        finally:
            client.delete(f"/api/calls/{call_id}", headers=account)

    def test_trunk_pbx_does_not_start_pyvoip_listener(
            self, client, admin, monkeypatch):
        """Regression: when sip_stack is running it registers the PBX as
        a trunk.  Starting an additional pyVoIP VoIPPhone for the same
        credentials causes both to claim inbound INVITEs; pyVoIP then
        decodes the trunk's G.722 payload as A-law (its default) →
        inbound audio garbage."""
        import json as _json
        from speech_pipeline.telephony import sip_listener, sip_stack

        # Pretend sip_stack is running (production setup).
        monkeypatch.setattr(sip_stack, "is_running", lambda: True)
        started: list[str] = []
        import pyVoIP.VoIP.VoIP as _pv
        monkeypatch.setattr(_pv, "VoIPPhone",
                            lambda *a, **kw: started.append("should_not") or None)

        resp = client.put(
            "/api/pbx/TrunkPBX",
            data=_json.dumps({
                "sip_proxy": "voip.example.com",
                "sip_user": "trunk_user",
                "sip_password": "secret",
            }),
            headers=admin,
        )
        assert resp.status_code == 200
        assert started == [], (
            "pyVoIP VoIPPhone was started in parallel to sip_stack — "
            "will fight over inbound INVITEs and mis-decode the codec"
        )
        assert "TrunkPBX" not in sip_listener._phones
        client.delete("/api/pbx/TrunkPBX", headers=admin)

    def test_put_pbx_without_sip_creds_returns_fast(self, client, admin):
        """Regression guard: ``PUT /api/pbx/<id>`` without sip_proxy/
        sip_user used to block for ~15s in ``VoIPPhone.start`` → pyVoIP
        REGISTER.  That hung the test fixture AND the production
        startup callback when the CRM provisioned a PBX with empty
        credentials (built-in sip_stack handles trunking)."""
        import json as _json
        (resp, elapsed) = _elapsed_ms(lambda: client.put(
            "/api/pbx/DeadlockTestPBX",
            data=_json.dumps({"sip_proxy": "", "sip_user": ""}),
            headers=admin,
        ))
        assert resp.status_code == 200
        assert elapsed < LATENCY_BUDGET_MS, (
            f"PUT /api/pbx blocked for {elapsed:.0f}ms — "
            f"in-process VoIPPhone.start is running without guard"
        )
        client.delete("/api/pbx/DeadlockTestPBX", headers=admin)

    def test_many_concurrent_crm_calls_are_not_deadlocked(
            self, client, account, monkeypatch):
        """Simulate PHP behaviour: the CRM fires N pipelines in quick
        succession.  If any single build blocks, the whole burst stalls."""
        monkeypatch.setattr(leg_mod, "originate_only", lambda leg, pbx: None)
        call_id = create_call(client, account)
        try:
            t0 = time.monotonic()
            for i in range(6):
                resp = client.post(
                    "/api/pipelines",
                    data=json.dumps({
                        "dsl": f'play:stage{i}{{"url":"examples/queue.mp3"}} '
                               f'-> call:{call_id}',
                    }),
                    headers=account,
                )
                assert resp.status_code == 201
            elapsed_ms = (time.monotonic() - t0) * 1000
            # 6 sequential calls must fit comfortably within a few
            # budgets; this fails loud if we regressed to sync builds.
            assert elapsed_ms < LATENCY_BUDGET_MS * 4, (
                f"6 sequential pipelines took {elapsed_ms:.0f}ms — "
                f"likely a deadlock or sync build"
            )
        finally:
            client.delete(f"/api/calls/{call_id}", headers=account)
