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
