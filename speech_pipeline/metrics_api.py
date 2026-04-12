"""Operational endpoints: liveness + Prometheus metrics.

Extracted into a blueprint so tests (and the main server) share one
implementation.  Scrape the text exposition at ``/metrics`` without
any authentication — assume the scraper lives on the private network
(``prometheus.yml`` -> ``static_configs.targets``).
"""
from __future__ import annotations

from flask import Blueprint

api = Blueprint("metrics_api", __name__)


@api.route("/healthz", methods=["GET"])
def healthz():
    return ("ok", 200, {"Content-Type": "text/plain"})


@api.route("/metrics", methods=["GET"])
def metrics():
    from speech_pipeline.telephony import call_state as _cs
    from speech_pipeline.telephony import leg as _leg
    from speech_pipeline.telephony import auth as _auth
    from speech_pipeline.telephony import webclient as _wc

    calls = list(_cs._calls.values())
    legs = list(_leg._legs.values())

    active_calls = sum(1 for c in calls if c.status == "active")
    completed_calls = sum(1 for c in calls if c.status == "completed")
    ringing_legs = sum(1 for lg in legs if lg.status == "ringing")
    in_progress_legs = sum(1 for lg in legs if lg.status == "in-progress")

    with _auth._nonce_lock:
        active_nonces = sum(1 for n in _auth._nonces.values()
                            if not n.get("connected"))
    with _wc._sessions_lock:
        webclient_sessions = len(_wc._sessions)

    lines = [
        "# HELP speech_calls_total Calls by status",
        "# TYPE speech_calls_total gauge",
        f'speech_calls_total{{status="active"}} {active_calls}',
        f'speech_calls_total{{status="completed"}} {completed_calls}',
        "# HELP speech_legs_total SIP legs by status",
        "# TYPE speech_legs_total gauge",
        f'speech_legs_total{{status="ringing"}} {ringing_legs}',
        f'speech_legs_total{{status="in-progress"}} {in_progress_legs}',
        "# HELP speech_nonces_active Unconsumed webclient nonces",
        "# TYPE speech_nonces_active gauge",
        f"speech_nonces_active {active_nonces}",
        "# HELP speech_webclient_sessions Active webclient sessions",
        "# TYPE speech_webclient_sessions gauge",
        f"speech_webclient_sessions {webclient_sessions}",
    ]
    return ("\n".join(lines) + "\n", 200, {"Content-Type": "text/plain"})
