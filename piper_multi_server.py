#!/usr/bin/env python3
"""Authenticated speech-pipeline server.

Clients build audio networks through /api/pipelines or /ws/pipe.
Prepared HTTP/codec endpoints use temporary secrets for downstream clients.
Install with: pip install -e '.[tts,stt,sip,server,vc]'
"""
from __future__ import annotations

import argparse
import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, Optional
import os

from flask import Flask, jsonify, request
from flask_sock import Sock

# Prefer the installed Piper build; fall back to local sources if absent.
import sys as _sys
import importlib.util as _importlib_util
from pathlib import Path as _Path
_candidates = [
    _Path('/home/carli/sources/piper/src'),
    (_Path(__file__).resolve().parents[2] / 'sources' / 'piper' / 'src'),
]
for _p in _candidates if _importlib_util.find_spec('piper') is None else []:
    try:
        if _p.exists():
            _sys.path.insert(0, str(_p))
    except Exception:
        pass

# Optional: FreeVC via Coqui TTS for voice conversion
# FreeVC availability and initialization handled inside VCConverter stage


_LOGGER = logging.getLogger("piper-multi-server")

def create_app(args: argparse.Namespace) -> Flask:
    # Voices live in a single folder (default: ./voices-piper). Allow override via --voices-path.
    # Back-compat: --scan-dir (single) behaves like --voices-path
    voices_arg = getattr(args, 'scan_dir', None) or getattr(args, 'voices_path', 'voices-piper')
    voices_dir = Path(voices_arg).resolve()
    _LOGGER.info("Voices dir: %s", voices_dir)
    # Use TTSRegistry for voice discovery, caching, and loading
    from lib.registry import TTSRegistry, load_voice_info, VoiceInfo  # type: ignore
    registry = TTSRegistry(voices_dir, use_cuda=args.cuda,
                           voice_ttl_seconds=int(getattr(args, 'voice_ttl_seconds', 7200)),
                           voice_cache_max=int(getattr(args, 'voice_cache_max', 64)))
    _LOGGER.info("Discovered %d voices", len(registry.index))
    # VC handled inside VCConverter; no global service here

    def ensure_loaded(model_id: str):
        return registry.ensure_loaded(model_id)

    # Preload default voice if provided
    default_model_id: Optional[str] = None
    if args.model:
        default_model_path = Path(args.model)
        if not default_model_path.exists():
            raise SystemExit(f"Model not found: {default_model_path}")
        default_model_id = default_model_path.name.rstrip(".onnx")
        _ = registry.ensure_loaded(default_model_id)
        registry.index.setdefault(default_model_id, default_model_path)

    # If no explicit default, pick the first discovered voice id (stable order)
    if (default_model_id is None) and registry.index:
        prefer = 'de_DE-thorsten-medium'
        default_model_id = prefer if prefer in registry.index else sorted(registry.index.keys())[0]

    app = Flask(__name__)
    sock = Sock(app)

    # Pipeline control API (requires --admin-token)
    admin_token = getattr(args, 'admin_token', None) or ''
    if admin_token:
        from speech_pipeline.pipeline_api import api as pipeline_api_bp, init as pipeline_api_init
        pipeline_api_init(admin_token)
        app.register_blueprint(pipeline_api_bp)
        _LOGGER.info("Pipeline control API enabled at /api/ (bearer-authenticated)")

        # Telephony API (shares the admin token, adds account-scoped auth)
        from speech_pipeline.telephony.auth import init as tel_auth_init
        from speech_pipeline.telephony.api import api as telephony_api_bp
        tel_auth_init(admin_token)
        accounts_file = getattr(args, "accounts_file", None)
        if accounts_file:
            from speech_pipeline.telephony.auth import put_account
            import json
            with open(accounts_file) as account_config:
                for account_id, account_data in json.load(account_config).items():
                    put_account(account_id, account_data)
        app.register_blueprint(telephony_api_bp)
        _LOGGER.info("Telephony API enabled at /api/ (admin + account auth)")

        # Share TTS registry and app with telephony modules
        import speech_pipeline.telephony._shared as _tel_shared
        _tel_shared.tts_registry = registry
        _tel_shared.flask_app = app
        _tel_shared.media_folder = getattr(args, "media_folder", None)

        # WebClient: phone UI + WebSocket
        from speech_pipeline.telephony.webclient import bp as webclient_bp
        app.register_blueprint(webclient_bp)
        _LOGGER.info("WebClient phone UI enabled")

        # Startup callback: notify the external provisioner that the speech
        # server is ready.  In production this is LDS, but from the server's
        # perspective it is intentionally generic: the callback target is the
        # admin-side orchestrator that will provision PBXs/accounts via the
        # API.  The speech server itself does not know LDS-specific logic.
        #
        # Resilience: a transient network error at boot (DNS not yet up,
        # upstream firewall not yet open, etc.) used to leave the server
        # permanently unprovisioned — no PBX, no accounts, no subscribers —
        # so every SIP REGISTER got 403 Forbidden until someone restarted
        # pm2 at the right time. We now retry with exponential backoff on
        # connection errors until the provisioner responds (any HTTP status
        # counts as "responded" — that's the provisioner's concern).
        startup_cb = getattr(args, 'startup_callback', None) or ''
        if startup_cb:
            startup_cb_token = getattr(args, 'startup_callback_token', None) or admin_token
            startup_max_attempts = getattr(args, 'startup_callback_max_attempts', 0) or 0
            startup_initial_delay = getattr(args, 'startup_callback_initial_delay', 0.0) or 1.0
            startup_max_delay = getattr(args, 'startup_callback_max_delay', 0.0) or 300.0

            def _fire_startup_callback(url: str, token: str) -> None:
                import requests
                import time as _time
                attempt = 0
                delay = startup_initial_delay
                while True:
                    attempt += 1
                    try:
                        _LOGGER.info(
                            "Sending startup callback to %s (attempt %d)",
                            url, attempt,
                        )
                        resp = requests.get(url, headers={
                            "Authorization": f"Bearer {token}",
                        }, timeout=30)
                        _LOGGER.info(
                            "Startup callback response: %s", resp.status_code,
                        )
                        return
                    except requests.exceptions.ConnectionError as e:
                        # Transient network issue: the box is up but the
                        # orchestrator (or DNS/route to it) is not yet
                        # reachable. Keep trying — the registrar is useless
                        # without provisioning.
                        if startup_max_attempts and attempt >= startup_max_attempts:
                            _LOGGER.error(
                                "Startup callback to %s gave up after %d "
                                "attempts: %s", url, attempt, e,
                            )
                            return
                        _LOGGER.warning(
                            "Startup callback to %s failed (attempt %d, "
                            "retrying in %.1fs): %s",
                            url, attempt, delay, e,
                        )
                        _time.sleep(delay)
                        delay = min(delay * 2, startup_max_delay)
                    except Exception as e:
                        # Non-connection errors (timeout, bad response,
                        # etc.) are logged but not retried — they indicate
                        # a server-side issue the orchestrator has to
                        # resolve on its own.
                        _LOGGER.warning(
                            "Startup callback to %s failed: %s", url, e,
                        )
                        return
            # Fire after app is ready (in a thread to not block startup)
            threading.Thread(
                target=_fire_startup_callback,
                args=(startup_cb, startup_cb_token),
                daemon=True,
                name="startup-callback",
            ).start()

    # Ensure our module logger emits at desired level and propagates to root handler
    try:
        _LOGGER.setLevel(logging.DEBUG if args.debug else logging.INFO)
        _LOGGER.propagate = True
    except Exception:
        pass

    # (processing helpers removed; handled by stages/util.py)

    # (ffmpeg/ffprobe helpers removed)

    # Basic CORS for all responses
    @app.after_request
    def add_cors_headers(resp):  # type: ignore
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, OPTIONS"
        # Allow Authorization so server-side bearer fetches are configurable if ever proxied
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Accept, Authorization, X-Sample-Rate"
        # Media streaming hint for proxies
        resp.headers["X-Accel-Buffering"] = "no"
        # Be explicit about referrer policy if the browser surfaces strict-origin-when-cross-origin
        resp.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        # Note: Content headers for audio are applied by ResponseWriter
        return resp

    from speech_pipeline.telephony.auth import require_account

    # ---- Static files from examples/ directory ----
    from flask import send_from_directory as _send_from_directory
    _examples_dir = Path(__file__).resolve().parent / "examples"

    @app.route("/examples/<path:filename>")
    @require_account
    def serve_examples(filename):
        return _send_from_directory(str(_examples_dir), filename)

    @app.route("/phone/<nonce>/codec.js")
    def phone_codec(nonce):
        from speech_pipeline.telephony.auth import check_nonce
        from speech_pipeline.telephony.webclient import _sessions, _sessions_lock
        if not check_nonce(nonce):
            return ("Invalid or expired nonce\n", 403)
        with _sessions_lock:
            known = any(session.get("nonce") == nonce for session in _sessions.values())
        if not known:
            return ("Session not found\n", 404)
        return _send_from_directory(str(_examples_dir), "codec.js")

    # Operational endpoints (/healthz + /metrics) live in a shared
    # blueprint so the test app can exercise them too.
    from speech_pipeline.metrics_api import api as metrics_bp
    app.register_blueprint(metrics_bp)

    @app.route("/voices", methods=["GET"])  # list models
    @require_account
    def voices() -> Any:
        # Ensure index is current
        registry.refresh_index()
        result: Dict[str, Any] = {}
        for mid, path in registry.index.items():
            info = registry.infos.get(mid)
            if not info:
                try:
                    info = load_voice_info(path)
                    registry.infos[mid] = info
                except Exception:
                    info = None
            if info:
                result[mid] = {
                    "path": str(path),
                    "espeak_voice": info.espeak_voice,
                    "sample_rate": info.sample_rate,
                    "num_speakers": info.num_speakers,
                    "speaker_id_map": info.speaker_id_map,
                }
            else:
                result[mid] = {"path": str(path)}
        return jsonify(result)

    def _ws_require_auth(ws) -> dict | None:
        """Check Bearer token from query param or first WS message.
        Returns account dict (or None for admin). Sends error and returns False on failure."""
        token = request.args.get("token", "")
        if not token:
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                token = auth_header[7:]
        if not token:
            return False
        if not admin_token:
            return False
        if token == admin_token:
            return None  # admin
        from speech_pipeline.telephony.auth import _find_account_by_token
        acct = _find_account_by_token(token)
        if acct:
            return acct
        return False

    # ---- Generic pipeline endpoint ----
    @sock.route("/ws/pipe")
    def ws_pipe(ws):
        """Generic pipeline endpoint: client sends JSON config, then data flows.

        Requires authentication: pass token as ?token=... query param
        or Authorization: Bearer header.

        Protocol:
        - Client -> Server: first message is JSON config:
            {"pipe": "ws:pcm | resample:48000:16000 | stt:de | ws:ndjson"}
          or for duplex:
            {"pipes": ["sip:100@pbx | resample:8000:16000 | stt:de | ws:ndjson",
                        "ws:text | tts{\"voice\":\"de_DE-thorsten-medium\"} | resample:24000:8000 | sip:100@pbx"]}
        - Then data flows according to the pipeline definition.
        """
        import json as _json
        auth_result = _ws_require_auth(ws)
        if auth_result is False:
            ws.send(_json.dumps({"error": "Unauthorized — pass ?token=... or Authorization header"}))
            return

        from lib.PipelineBuilder import PipelineBuilder

        try:
            config_msg = ws.receive(timeout=10)
        except Exception:
            ws.send(_json.dumps({"error": "No config message received"}))
            return

        try:
            config = _json.loads(config_msg)
        except Exception:
            ws.send(_json.dumps({"error": "Invalid JSON config"}))
            return

        # Track in LivePipeline registry if admin API is enabled
        from speech_pipeline.live_pipeline import LivePipeline, register as _lp_register, unregister as _lp_unregister
        dsl = config.get("pipe", "") or "; ".join(config.get("pipes", []))
        from flask import g
        from speech_pipeline.pipeline_api import _check_call_ownership
        g.account = auth_result
        for chain in config.get("pipes", [config.get("pipe", "")]):
            error = _check_call_ownership(chain)
            if error:
                ws.send(_json.dumps({"error": error}))
                return
        lp = LivePipeline(dsl=dsl)
        lp.account_id = auth_result["id"] if auth_result is not None else None
        builder = PipelineBuilder(ws, registry, args, live_pipeline=lp)

        # Inject conference mixers referenced in the DSL
        from speech_pipeline.PipelineBuilder import inject_conference_mixers
        inject_conference_mixers(builder, dsl)

        # After build, wire ConferenceLeg callbacks to subscriber events
        def _wire_leg_callbacks(run_obj):
            _LOGGER.info("_wire_leg_callbacks called with %d stages", len(run_obj.stages))
            try:
                from speech_pipeline.ConferenceLeg import ConferenceLeg
                from speech_pipeline.telephony.webclient import get_webclient_session
                from speech_pipeline.telephony import call_state as _cs, subscriber as _sub
                import requests as _http
                import re as _re

                leg_count = sum(1 for s in run_obj.stages if isinstance(s, ConferenceLeg))
                _LOGGER.info("_wire_leg_callbacks: %d legs in %d stages, dsl=%s",
                             leg_count, len(run_obj.stages), dsl[:80])
                for stage in run_obj.stages:
                    if not isinstance(stage, ConferenceLeg):
                        continue
                    # Find the webclient session for this leg
                    for sid_match in _re.findall(r'codec:(wc-[^\s|]+)', dsl):
                        sess = get_webclient_session(sid_match)
                        if not sess:
                            continue
                        call = _cs.get_call(sess["call_id"])
                        if not call:
                            continue
                        sub = _sub.get(call.subscriber_id)
                        if not sub:
                            continue
                        participant = call.get_participant(sess["session_id"])
                        callback_path = participant.get("callback") if participant else None

                        _LOGGER.info("_wire: participant=%s callback=%s", sess["session_id"], callback_path)

                        def _make_on_attached(sub_info, call_obj, cb_path, part_id):
                            def _on_attached(leg):
                                _LOGGER.info("on_attached fired for %s cb=%s", part_id, cb_path)
                                if not cb_path:
                                    return
                                url = sub_info["base_url"].rstrip("/") + "/" + cb_path.lstrip("/")
                                # MUST be async — this fires from the
                                # mixer's yield loop; a blocking POST
                                # deadlocks the whole conference until
                                # the CRM responds.
                                def _send():
                                    try:
                                        _http.post(url, json={
                                            "callId": call_obj.call_id,
                                            "command": "webclient",
                                            "participantId": part_id,
                                            "result": "joined",
                                        }, headers={"Authorization": f"Bearer {sub_info['bearer_token']}"}, timeout=5)
                                    except Exception:
                                        pass
                                threading.Thread(
                                    target=_send, daemon=True,
                                    name=f"wc-attach-{part_id}",
                                ).start()
                            return _on_attached

                        stage.on_attached = _make_on_attached(sub, call, callback_path, sess["session_id"])

                        def _make_on_detached(sub_info, call_obj, cb_path, part_id):
                            def _on_detached(leg):
                                if not cb_path:
                                    return
                                url = sub_info["base_url"].rstrip("/") + "/" + cb_path.lstrip("/")
                                def _send():
                                    try:
                                        _http.post(url, json={
                                            "callId": call_obj.call_id,
                                            "command": "webclient",
                                            "participantId": part_id,
                                            "result": "left",
                                        }, headers={"Authorization": f"Bearer {sub_info['bearer_token']}"}, timeout=5)
                                    except Exception:
                                        pass
                                threading.Thread(
                                    target=_send, daemon=True,
                                    name=f"wc-detach-{part_id}",
                                ).start()
                            return _on_detached

                        stage.on_detached = _make_on_detached(sub, call, callback_path, sess["session_id"])
            except ImportError as e:
                _LOGGER.warning("_wire_leg_callbacks import error: %s", e)
            except Exception as e:
                _LOGGER.warning("_wire_leg_callbacks error: %s", e, exc_info=True)

        try:
            if "pipe" in config:
                run = builder.build(config["pipe"])
                _wire_leg_callbacks(run)
                lp.run = run
                lp.state = "running"
                _lp_register(lp)
                run.run()
            elif "pipes" in config:
                runs = builder.build_multi(config["pipes"])
                for r in runs:
                    _wire_leg_callbacks(r)
                lp.run = runs[0] if runs else None
                lp.state = "running"
                _lp_register(lp)
                if len(runs) == 1:
                    runs[0].run()
                else:
                    threads = [threading.Thread(target=r.run, daemon=True) for r in runs]
                    for t in threads:
                        t.start()
                    for t in threads:
                        t.join()
            else:
                ws.send(_json.dumps({"error": "Config must contain 'pipe' or 'pipes'"}))
        except Exception as e:
            _LOGGER.warning("ws_pipe error: %s", e, exc_info=True)
            try:
                ws.send(_json.dumps({"error": str(e)}))
            except Exception:
                pass
        finally:
            lp.state = "stopped"
            _lp_unregister(lp.id)

    # ---- Codec WebSocket endpoint ----
    @sock.route("/ws/socket/<session_id>")
    def ws_codec_socket(ws, session_id):
        """Fourier-codec bidirectional audio socket.

        Pipeline stages create a CodecSocketSession for *session_id*.
        The browser connects here; the session handles handshake +
        encode/decode and bridges to the pipeline via rx/tx queues.
        """
        from lib.CodecSocketSession import get_session
        from speech_pipeline.CodecSocketSession import CodecSocketSession
        from speech_pipeline.telephony.webclient import get_webclient_session
        import json as _json

        # Browser may open /ws/socket/<sid> in parallel with /ws/pipe
        # — there's no guarantee the pipeline has registered the
        # CodecSocketSession yet when we get here.  Pre-create the
        # session if it doesn't exist but the webclient slot is known.
        # PipelineBuilder's _get_or_create_codec_session will reuse it.
        session = get_session(session_id)
        if not session:
            # Webclient session IDs are now account-scoped ("account:wc-...").
            # Pre-create a codec session for every known webclient slot,
            # regardless of whether the ID is scoped or legacy-local.
            if get_webclient_session(session_id):
                session = CodecSocketSession(session_id)
                _LOGGER.info(
                    "ws_codec_socket: pre-created CodecSocketSession for %s",
                    session_id,
                )
            else:
                ws.send(_json.dumps({
                    "error": "Unknown session ID",
                    "session_id": session_id,
                }))
                return
        session.handle_ws(ws)
        # NB: don't close_webclient_session here — browsers do a normal
        # disconnect-reconnect cycle on page load, and tearing down the
        # session on the first WS end makes every reconnect fail.
        # Cleanup happens on:
        #   - explicit DELETE /api/calls/<id>  → close_call_sessions
        #   - reaper thread for never-connected sessions (5 min TTL)
        #   - call idle-timeout (ConferenceMixer.on_idle_cancel)

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--model", help="Optional: preload this voice (.onnx path)")
    parser.add_argument("--voices-path", default="voices-piper", help="Directory that contains *.onnx voices (default: voices-piper)")
    parser.add_argument("--scan-dir", help="(legacy) Single directory to scan for *.onnx voices; same as --voices-path")
    parser.add_argument("--cuda", action="store_true", help="Use GPU")
    parser.add_argument("--sentence-silence", type=float, default=0.0, help="Seconds of silence between sentences")
    parser.add_argument("--media-folder", default=None, help="Folder for relative play/vc media names. URLs remain allowed; '../' is rejected.")
    parser.add_argument("--bearer", default="", help="Bearer token for authorizing remote (http/https) downloads/streams")
    parser.add_argument("--whisper-model", default="base", help="Whisper model size for STT (default: base)")
    parser.add_argument("--admin-token", default="", help="Bearer token to enable the /api/ pipeline control endpoints")
    parser.add_argument("--accounts-file", help="Private JSON mapping of account IDs to account settings, provisioned at startup")
    parser.add_argument("--startup-callback", default="", help="URL to GET on startup. The called service provisions PBX/accounts via the API.")
    parser.add_argument("--startup-callback-token", default="", help="Bearer token for the startup callback (if different from --admin-token).")
    parser.add_argument("--sip-port", type=int, default=0, help="Enable built-in SIP stack on this UDP port (e.g. 5061). Disables pyVoIP.")
    parser.add_argument(
        "--telephony-log-level",
        type=int,
        default=2,
        choices=range(0, 5),
        help="Telephony verbosity: 0=off, 1=register/provision/init, 2=calls, 3=webhooks, 4=all SIP/debug.",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    try:
        from speech_pipeline.telephony import logcontrol as _tel_logcontrol
        effective_tel_level = _tel_logcontrol.configure(args.telephony_log_level, debug=args.debug)
        _LOGGER.info("Telephony log level set to %d", effective_tel_level)
    except Exception:
        _LOGGER.warning("Failed to configure telephony log level", exc_info=True)

    # Fix pyVoIP trans() timing (must be set before any calls)
    try:
        import pyVoIP
        pyVoIP.TRANSMIT_DELAY_REDUCTION = 1.0
    except ImportError:
        pass

    # Start built-in SIP stack if requested
    if args.sip_port:
        from speech_pipeline.telephony import sip_stack
        sip_stack.init(args.sip_port)
        _LOGGER.info("Built-in SIP stack enabled on port %d", args.sip_port)

    app = create_app(args)

    # Graceful shutdown: on SIGTERM/SIGINT, hang up active legs + close
    # webclient sessions so SIP/WebSocket peers see a clean BYE/close
    # instead of a dead socket.
    import signal as _signal

    def _graceful_shutdown(signum, _frame):
        _LOGGER.info("Signal %d received — draining active calls", signum)
        try:
            from speech_pipeline.telephony import call_state as _cs
            for call_id in list(_cs._calls.keys()):
                try:
                    _cs.delete_call(call_id)
                except Exception as e:
                    _LOGGER.warning("shutdown: delete_call(%s) failed: %s",
                                    call_id, e)
        except Exception as e:
            _LOGGER.warning("shutdown drain error: %s", e)
        # Re-raise default behaviour so the process exits.
        _signal.signal(signum, _signal.SIG_DFL)
        import os
        os.kill(os.getpid(), signum)

    _signal.signal(_signal.SIGTERM, _graceful_shutdown)
    _signal.signal(_signal.SIGINT, _graceful_shutdown)

    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
