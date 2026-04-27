"""Smoke tests for piper_multi_server.

The regular pytest fixture builds its own Flask app out of blueprints
— it does NOT import ``piper_multi_server``.  That means bootstrap
bugs (IndentationError, UnboundLocalError, missing imports in
``create_app``, wrong route definitions in the main entry point) go
completely unnoticed by the existing suite until the production pm2
process starts crash-looping.

This module closes that gap by importing the module and exercising
the routes registered directly on the main ``app`` (not via blueprint).
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest


def _load_main_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "piper_multi_server.py"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("piper_multi_server", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["piper_multi_server"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def main_app():
    """Build the real ``piper_multi_server`` app with minimal args."""
    pms = _load_main_module()

    args = argparse.Namespace(
        host="127.0.0.1",
        port=0,
        model=None,
        voices_path="voices-piper",
        scan_dir=None,
        cuda=False,
        sentence_silence=0.0,
        soundpath="../voices/%s.wav",
        media_folder=str(Path(__file__).resolve().parents[1]),
        bearer="",
        whisper_model="base",
        admin_token="test-admin-token",
        startup_callback="",
        startup_callback_token="",
        sip_port=0,
        debug=False,
    )
    app = pms.create_app(args)
    app.config["TESTING"] = True
    return app


@pytest.fixture
def main_client(main_app):
    return main_app.test_client()


class TestBootstrapCompiles:
    """Module import + create_app() must succeed — catches
    IndentationError, NameError, UnboundLocalError at the main
    entrypoint."""

    def test_module_imports(self):
        mod = _load_main_module()
        assert mod is not None

    def test_create_app_succeeds(self, main_app):
        assert main_app is not None
        # The app has at least the core routes wired.
        rules = {r.rule for r in main_app.url_map.iter_rules()}
        for expected in ("/healthz", "/metrics", "/voices", "/tts/say", "/"):
            assert expected in rules, f"missing route: {expected}"


class TestCoreRoutes:
    """Hit the routes the blueprint-only test fixture can't see."""

    def test_healthz(self, main_client):
        resp = main_client.get("/healthz")
        assert resp.status_code == 200
        assert resp.data.strip() == b"ok"

    def test_metrics_is_exposition_format(self, main_client):
        resp = main_client.get("/metrics")
        assert resp.status_code == 200
        assert "text/plain" in resp.headers.get("Content-Type", "")
        assert b"speech_calls_total" in resp.data

    def test_voices_returns_json(self, main_client):
        resp = main_client.get("/voices")
        assert resp.status_code == 200
        assert resp.is_json

    def test_unknown_route_404(self, main_client):
        resp = main_client.get("/no-such-route")
        assert resp.status_code == 404

    def test_legacy_inputstream_removed(self, main_client):
        """Removed in an earlier hardening — guard against re-introduction."""
        resp = main_client.post("/inputstream")
        assert resp.status_code == 404

    def test_root_sound_rejects_parent_traversal(self, main_client):
        resp = main_client.post(
            "/",
            json={"sound": "../secret.wav"},
        )
        assert resp.status_code == 400
        assert b"must not contain" in resp.data

    def test_root_sound_accepts_media_folder_file(self, main_client):
        resp = main_client.post(
            "/",
            json={"sound": "examples/queue.mp3"},
        )
        assert resp.status_code == 200
        assert len(resp.data) > 100

    def test_root_vc_rejects_parent_traversal(self, main_client):
        resp = main_client.post(
            "/",
            json={"sound": "examples/queue.mp3", "voice2": "../secret.wav"},
        )
        assert resp.status_code == 400
        assert b"must not contain" in resp.data


class TestStartupCallbackResilience:
    """Regression guard: a transient network error at boot must NOT leave
    the server permanently unprovisioned.

    A real incident: at a pm2 restart the host couldn't reach the
    orchestrator ('Network is unreachable'), the fire-and-forget
    callback logged a warning and gave up, and the server kept running
    for hours with no PBX / no accounts / no subscribers. Every inbound
    SIP REGISTER got 403 Forbidden because the subscriber lookup had
    nothing to match against. The callback must retry with exponential
    backoff so provisioning eventually completes once the network
    recovers.
    """

    def test_connection_error_triggers_retry_until_success(self, monkeypatch):
        import argparse
        import threading
        import time

        pms = _load_main_module()

        calls = {"n": 0}
        done = threading.Event()

        class _FakeResp:
            status_code = 204

        def fake_get(url, headers=None, timeout=None):
            calls["n"] += 1
            if calls["n"] < 3:
                # First two attempts: simulate transient network fault
                from requests import exceptions as rexc
                raise rexc.ConnectionError("Network is unreachable")
            done.set()
            return _FakeResp()

        import requests as _requests
        monkeypatch.setattr(_requests, "get", fake_get)

        args = argparse.Namespace(
            host="127.0.0.1", port=0, model=None, voices_path="voices-piper",
            scan_dir=None, cuda=False, sentence_silence=0.0,
            soundpath="../voices/%s.wav",
            media_folder=str(Path(__file__).resolve().parents[1]),
            bearer="", whisper_model="base",
            admin_token="test-admin-token",
            startup_callback="https://orchestrator.example.test/startup",
            startup_callback_token="cb-token",
            # Fast backoff so the test finishes quickly
            startup_callback_initial_delay=0.05,
            startup_callback_max_delay=0.1,
            startup_callback_max_attempts=10,
            sip_port=0, debug=False,
        )
        pms.create_app(args)

        assert done.wait(timeout=5.0), (
            f"callback never succeeded after {calls['n']} attempts — "
            f"retry loop is broken or gave up too early"
        )
        assert calls["n"] >= 3, (
            f"expected at least 3 attempts (two ConnectionError + one OK), "
            f"got {calls['n']}"
        )

    def test_gives_up_after_max_attempts(self, monkeypatch):
        """Retry must be bounded — an endlessly-unreachable orchestrator
        should not spin forever. After ``max_attempts`` it logs an error
        and stops (the operator now has to either fix the network and
        restart pm2, or configure a higher cap)."""
        import argparse
        import threading

        pms = _load_main_module()

        calls = {"n": 0}
        gave_up = threading.Event()

        def fake_get(url, headers=None, timeout=None):
            calls["n"] += 1
            from requests import exceptions as rexc
            raise rexc.ConnectionError("still unreachable")

        import requests as _requests
        monkeypatch.setattr(_requests, "get", fake_get)

        # Detect the "gave up" log line via a logging handler
        import logging

        class _SignalHandler(logging.Handler):
            def emit(self, record):
                if "gave up" in record.getMessage():
                    gave_up.set()

        handler = _SignalHandler()
        logging.getLogger("piper-multi-server").addHandler(handler)

        try:
            args = argparse.Namespace(
                host="127.0.0.1", port=0, model=None, voices_path="voices-piper",
                scan_dir=None, cuda=False, sentence_silence=0.0,
                soundpath="../voices/%s.wav",
                media_folder=str(Path(__file__).resolve().parents[1]),
                bearer="", whisper_model="base",
                admin_token="test-admin-token",
                startup_callback="https://orchestrator.example.test/startup",
                startup_callback_token="cb-token",
                startup_callback_initial_delay=0.01,
                startup_callback_max_delay=0.02,
                startup_callback_max_attempts=3,
                sip_port=0, debug=False,
            )
            pms.create_app(args)

            assert gave_up.wait(timeout=5.0), (
                f"retry loop did not surrender after 3 attempts ({calls['n']} calls)"
            )
            assert calls["n"] == 3, (
                f"expected exactly 3 attempts, got {calls['n']}"
            )
        finally:
            logging.getLogger("piper-multi-server").removeHandler(handler)
