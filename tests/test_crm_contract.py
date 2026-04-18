"""Contract guardrail: every HTTP endpoint the CRM calls must exist.

Scans the deployed CRM ``.fop`` files for ``SpeechServer::request('VERB',
'/path', ...)`` calls and asserts that each ``path`` is served by the
current Flask app.  If we remove or rename a server endpoint without
updating the CRM, this test fails before production does.

Location of the CRM is configurable via env var ``CRM_FOP_DIR``.
Defaults to ``/home/carli/projekte/fop-dev/backends/businesslogic/
telefonanlage/speech-server`` (developer layout).
"""
from __future__ import annotations

import os
import re
import sys
import urllib.parse
from pathlib import Path

import pytest

_DEFAULT = Path(
    "/home/carli/projekte/fop-dev/backends/businesslogic/"
    "telefonanlage/speech-server"
)
CRM_DIR = Path(os.environ.get("CRM_FOP_DIR", str(_DEFAULT)))

_CALL_RE = re.compile(
    r"""SpeechServer::request\s*\(\s*
        ['"](?P<verb>GET|POST|PUT|DELETE|PATCH)['"]\s*,\s*
        ['"](?P<path>[^'"]+)['"]
    """,
    re.VERBOSE,
)


def _collect_crm_calls() -> list[tuple[str, str, str]]:
    """Return list of (verb, path, origin_file:line) from every .fop file."""
    if not CRM_DIR.exists():
        pytest.skip(f"CRM not available at {CRM_DIR} — set CRM_FOP_DIR")
    out: list[tuple[str, str, str]] = []
    for fop in CRM_DIR.rglob("*.fop"):
        text = fop.read_text(encoding="utf-8", errors="replace")
        for m in _CALL_RE.finditer(text):
            verb, path = m.group("verb"), m.group("path")
            # Cut query string for route matching.
            path_only = urllib.parse.urlsplit(path).path
            line_no = text.count("\n", 0, m.start()) + 1
            out.append((verb, path_only, f"{fop.name}:{line_no}"))
    return out


def _route_exists(app, verb: str, path: str) -> bool:
    """Check if *path* maps to a registered route that accepts *verb*."""
    for rule in app.url_map.iter_rules():
        if verb not in (rule.methods or set()):
            continue
        # Convert Flask variable syntax (<var>) into a regex.  Handle
        # PHP string templating used by CRM (``/api/calls/`` + var);
        # match only up to the first template boundary.
        pattern = re.sub(r"<[^>]+>", r"[^/]+", rule.rule)
        pattern = f"^{pattern}$"
        # CRM paths often end with a ``/`` and interpolated ID —
        # strip the trailing dynamic segment for route checks.
        candidate = path.rstrip("/")
        # Truncate at first variable-like token in the CRM path
        # (e.g. ``/api/calls/`` used as prefix for urlencode()).
        candidate_stripped = re.sub(r"/$", "", candidate)
        if re.match(pattern, candidate_stripped):
            return True
        # Allow the CRM path to be a *prefix* that the code appends
        # an id to at runtime.  Check if pattern's prefix matches.
        rule_prefix = re.sub(r"/<[^>]+>.*", "", rule.rule).rstrip("/")
        if candidate_stripped == rule_prefix:
            return True
    return False


@pytest.fixture(scope="module")
def app():
    """Build the full Flask app (not just blueprints) for route checks."""
    import argparse
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    import piper_multi_server as pms
    args = argparse.Namespace(
        host="127.0.0.1", port=0, model=None, voices_path="voices-piper",
        scan_dir=None, cuda=False, sentence_silence=0.0,
        soundpath="../voices/%s.wav", bearer="", whisper_model="base",
        admin_token="test-admin", startup_callback="",
        startup_callback_token="", sip_port=0, debug=False,
    )
    return pms.create_app(args)


class TestCrmCallsHitRealRoutes:

    def test_every_crm_endpoint_is_served(self, app):
        """Every CRM ``SpeechServer::request`` path must exist on the
        server.  Fails loud with file:line + verb+path on mismatch."""
        missing = []
        for verb, path, origin in _collect_crm_calls():
            if not _route_exists(app, verb, path):
                missing.append(f"{origin}: {verb} {path}")
        assert not missing, (
            "CRM is calling removed/renamed server endpoints:\n  "
            + "\n  ".join(missing)
        )

    def test_no_legs_originate_in_crm(self):
        """Specifically ban the removed ``/api/legs/originate`` path
        (caused a silent production break on 2026-04-12)."""
        calls = _collect_crm_calls()
        bad = [c for c in calls if "/api/legs/originate" in c[1]]
        assert not bad, (
            "CRM still calls removed /api/legs/originate:\n  "
            + "\n  ".join(f"{origin}: {v} {p}" for v, p, origin in bad)
        )

    def test_no_commands_endpoint_in_crm(self):
        """``/api/calls/<id>/commands`` was removed in favour of DSL."""
        calls = _collect_crm_calls()
        bad = [c for c in calls if "/commands" in c[1]]
        assert not bad, (
            "CRM still calls removed /commands endpoint:\n  "
            + "\n  ".join(f"{origin}: {v} {p}" for v, p, origin in bad)
        )

    def test_no_legacy_call_delete_in_crm(self):
        """Full call teardown must use DSL, not ``DELETE /api/calls/<id>``."""
        calls = _collect_crm_calls()
        bad = [c for c in calls if c[0] == "DELETE" and c[1] == "/api/calls/"]
        assert not bad, (
            "CRM still calls legacy DELETE /api/calls/<id> instead of "
            "DELETE /api/pipelines?dsl=call:<id>:\n  "
            + "\n  ".join(f"{origin}: {v} {p}" for v, p, origin in bad)
        )
