"""Flask blueprint for pipeline control API.

Endpoints require account-level auth (account token or admin token).
Pipelines that reference calls via ``call:ID`` are subject to
ownership checks — accounts can only access their own calls.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from flask import Blueprint, g, jsonify, request

from . import live_pipeline as registry
from .telephony import auth

_LOGGER = logging.getLogger("pipeline-api")

api = Blueprint("pipeline_api", __name__, url_prefix="/api")


def init(admin_token: str) -> None:
    """Initialize auth (delegates to telephony auth)."""
    auth.init(admin_token)


# Use telephony auth: admin token OR account token
_require_auth = auth.require_account


# ---- Pipeline CRUD ----

@api.route("/pipelines", methods=["GET"])
@_require_auth
def list_pipelines():
    """List running pipelines, or resolve a DSL item to its live object.

    Query param ``?dsl=<single_item>`` looks up a specific live object:
    - ``call:call-xxx``   → call details + participants
    - ``conference:xxx``  → same as call
    - ``sip:leg-abc``     → leg details
    - ``play:hold_music`` → stage info (exists / cancelled)
    - ``bridge:leg-abc``  → stage info
    - ``tee:tap``         → tee info + sidechain count

    Without ``?dsl``: list all pipelines.
    """
    dsl = request.args.get("dsl", "").strip()
    if not dsl:
        return jsonify([p.to_dict() for p in registry.list_all()])

    # DSL must be a single item (no pipes)
    if "->" in dsl or "|" in dsl:
        return ("GET only accepts a single DSL item, not a pipeline\n", 400)

    # Parse the item
    from .dsl_parser import parse_dsl
    try:
        elements = parse_dsl(dsl)
    except ValueError as e:
        return (f"DSL parse error: {e}\n", 400)
    if len(elements) != 1:
        return ("GET accepts only a single DSL item\n", 400)
    typ, elem_id, _ = elements[0]

    from .telephony import call_state, leg as leg_mod

    if typ in ("call", "conference"):
        if not elem_id:
            return ("Missing ID\n", 400)
        call = call_state.get_call(elem_id)
        if not call:
            return (f"Call '{elem_id}' not found\n", 404)
        # Ownership check
        aid = _account_id()
        if aid and call.account_id != aid:
            return ("Forbidden\n", 403)
        return jsonify(call.to_dict())

    if typ == "sip":
        if not elem_id:
            return ("Missing ID\n", 400)
        leg = leg_mod.get_leg(elem_id)
        if not leg:
            return (f"Leg '{elem_id}' not found\n", 404)
        return jsonify(leg.to_dict())

    # Stage lookup (play:, bridge:, tee:, etc.) — search across executors
    for call in call_state.list_calls():
        ex = getattr(call, "pipe_executor", None)
        if not ex:
            continue
        if typ == "tee" and elem_id in ex._tees:
            tee = ex._tees[elem_id]
            aid = _account_id()
            if aid and call.account_id != aid:
                return ("Forbidden\n", 403)
            return jsonify({
                "type": "tee",
                "id": elem_id,
                "call_id": call.call_id,
                "cancelled": getattr(tee, "cancelled", False),
                "sidechain_count": sum(1 for s in ex._sidechain_specs if s[0] == elem_id),
            })
        # Stage IDs in _stages are full strings like "play:hold" or "bridge:leg-abc"
        full_id = dsl  # user passed e.g. "play:hold"
        if full_id in ex._stages:
            aid = _account_id()
            if aid and call.account_id != aid:
                return ("Forbidden\n", 403)
            handle = ex._stages[full_id]
            stage = getattr(handle, "_stage", None)
            return jsonify({
                "id": full_id,
                "call_id": call.call_id,
                "cancelled": getattr(stage, "cancelled", False) if stage else False,
            })

    return (f"'{dsl}' not found\n", 404)


@api.route("/pipelines/<pid>", methods=["GET"])
@_require_auth
def get_pipeline(pid: str):
    p = registry.get(pid)
    if not p:
        return ("Pipeline not found\n", 404)
    return jsonify(p.to_dict(detail=True))


def _account_id() -> Optional[str]:
    acct = getattr(g, "account", None)
    return acct["id"] if acct else None


def _check_call_ownership(dsl: str) -> Optional[str]:
    """Verify the caller owns every referenced live resource.

    Covers call/conference, sip/bridge, codec, and tee sidechain
    attachments.  Admin bypasses all checks.
    """
    aid = _account_id()
    if not aid:
        return None  # admin — no restriction

    from .telephony import call_state, leg as leg_mod, subscriber as sub_mod
    from .telephony import webclient as wc_mod

    def _sub_account(subscriber_id: str) -> Optional[str]:
        sub = sub_mod.get(subscriber_id)
        return sub.get("account_id") if sub else None

    # call/conference
    for call_id in re.findall(r'(?:call|conference):([^\s{|>]+)', dsl):
        call = call_state.get_call(call_id)
        if not call:
            return f"Call {call_id} not found"
        if call.account_id != aid:
            return f"Forbidden: call {call_id} belongs to another account"

    # sip / bridge — both reference a leg by id
    for leg_id in re.findall(r'(?:sip|bridge):([^\s{|>]+)', dsl):
        leg = leg_mod.get_leg(leg_id)
        if not leg:
            continue  # bridge: against unknown id → let the endpoint 404
        owner = _sub_account(leg.subscriber_id)
        if owner and owner != aid:
            return (f"Forbidden: leg {leg_id} belongs to another account")

    # codec — webclient session; its call determines the account
    for session_id in re.findall(r'codec:([^\s{|>]+)', dsl):
        sess = wc_mod.get_webclient_session(session_id)
        if not sess:
            continue
        call = call_state.get_call(sess.get("call_id", ""))
        if call and call.account_id != aid:
            return (f"Forbidden: codec session {session_id} belongs "
                    f"to another account")

    # tee sidechain attach — first element tee:NAME that already exists
    # in someone else's executor ⇒ reject.
    for m in re.finditer(r'(?:^|\|\s*|->\s*)tee:([^\s{|>]+)', dsl):
        tee_id = m.group(1)
        for call in call_state.list_calls():
            ex = getattr(call, "pipe_executor", None)
            if ex and tee_id in getattr(ex, "_tees", {}):
                if call.account_id != aid:
                    return (f"Forbidden: tee {tee_id} belongs to "
                            f"another account")
                break  # owning tee found in caller's account — fine
    return None


@api.route("/pipelines", methods=["POST"])
@_require_auth
def create_pipeline():
    """Create a pipeline from DSL.

    Body: ``{"dsl": "sip:leg1 -> call:call-xxx -> sip:leg1"}``

    The DSL supports all element types: sip, call, play, tts, stt,
    tee, webhook, gain, delay, pitch, vc, record, conference, text_input.
    """
    body = request.get_json(force=True, silent=True) or {}
    dsl = body.get("dsl", "").strip()
    if not dsl:
        return ("Missing 'dsl' in request body\n", 400)

    # Render mode: synchronous execution, returns WAV
    if body.get("render"):
        return _render_pipeline(dsl)

    # Ownership check
    err = _check_call_ownership(dsl)
    if err:
        return (err + "\n", 403)

    from .dsl_parser import parse_dsl
    from .telephony.pipe_executor import CallPipeExecutor
    from .telephony import _shared

    # Parse to detect call references and resolve context
    try:
        elements = parse_dsl(dsl)
    except ValueError as e:
        return (f"DSL parse error: {e}\n", 400)

    # Find call context from DSL (if any call/conference element present).
    # Single-element actions (webclient, answer) may carry a ``call_id``
    # param to target a specific call.
    call = None
    for typ, elem_id, params in elements:
        if typ in ("call", "conference") and elem_id:
            from .telephony import call_state
            call = call_state.get_call(elem_id)
            break
        if typ == "webclient":
            cid = (params or {}).get("call_id")
            if cid:
                from .telephony import call_state
                call = call_state.get_call(cid)
                break

    # If no call element but a tee sidechain (tee:X as first element),
    # find the executor that owns this tee.
    if not call and elements and elements[0][0] == "tee":
        tee_id = elements[0][1]
        from .telephony import call_state
        for c in call_state.list_calls():
            ex = getattr(c, "pipe_executor", None)
            if ex and tee_id in ex._tees:
                call = c
                break


    # Reuse existing executor for the call (preserves tees/stages),
    # or create a new one for standalone pipelines.
    tts_registry = _shared.tts_registry
    if call:
        executor = _shared.ensure_pipe_executor(call)
    else:
        executor = CallPipeExecutor(call=None, tts_registry=tts_registry)

    pipeline = registry.LivePipeline(dsl=dsl)

    # Build synchronously.  Stages must be registered in the
    # executor's ``_stages`` dict before this response returns so that
    # a quick follow-up ``DELETE /api/pipelines {dsl: "play:..."}``
    # finds them.  The heavy operations inside add_pipes (AudioReader,
    # SIPSink, etc.) already hand off to threads in ``_start_all``, so
    # this stays in the tens of milliseconds — fast enough for PHP.
    try:
        results = executor.add_pipes([dsl])
    except Exception as e:
        return (f"Build error: {e}\n", 400)

    for r in results:
        if not r.get("ok"):
            return (f"Pipe error: {r.get('error', 'unknown')}\n", 400)

    if executor._text_input_queue:
        pipeline._text_input_queue = executor._text_input_queue
    pipeline.state = "running"
    pipeline._executor = executor
    registry.register(pipeline)

    # Surface side-effect IDs (e.g. originate leg_id, webclient
    # session_id/nonce/iframe_url) in the response so the CRM can
    # store them on the Participant row deterministically.
    body = pipeline.to_dict()
    for r in results:
        for k in ("leg_id", "session_id", "nonce", "iframe_url"):
            if r.get(k):
                body[k] = r[k]
    return jsonify(body), 201


def _render_pipeline(dsl: str):
    """Render a pipeline synchronously and return audio as WAV."""
    from .dsl_parser import parse_dsl

    try:
        elements = parse_dsl(dsl)
    except ValueError as e:
        return (f"DSL parse error: {e}\n", 400)

    # Reject elements that require a live call
    live_types = {"sip", "call", "conference", "webhook", "text_input"}
    for typ, _, _ in elements:
        if typ in live_types:
            return (f"render does not support '{typ}' — use POST /api/pipelines instead\n", 400)

    from .telephony.pipe_executor import CallPipeExecutor
    from .telephony import _shared

    executor = CallPipeExecutor(call=None, tts_registry=_shared.tts_registry)

    try:
        results = executor.add_pipes([dsl])
    except Exception as e:
        return (f"Build error: {e}\n", 400)

    for r in results:
        if not r.get("ok"):
            return (f"Pipe error: {r.get('error', 'unknown')}\n", 400)

    # The executor ran the pipeline — for offline TTS, the stages have
    # already produced audio. But add_pipes runs asynchronously.
    # For synchronous rendering, we need to build and drain manually.
    import io
    import struct
    import wave

    try:
        resolved_elements = parse_dsl(dsl)
        stages = []
        for typ, elem_id, params in resolved_elements:
            stages.append(executor._create_stage(typ, elem_id, params))

        # Resolve streaming TTS
        for i, stage in enumerate(stages):
            if getattr(stage, '_streaming', False):
                if i == 0:
                    return ("Streaming tts requires upstream text source\n", 400)
                upstream = stages[i - 1]
                def _text_iter(src):
                    for chunk in src.stream_pcm24k():
                        yield chunk.decode("utf-8", errors="replace") if isinstance(chunk, bytes) else str(chunk)
                from .StreamingTTSProducer import StreamingTTSProducer
                stages[i] = StreamingTTSProducer(_text_iter(upstream), stage.voice, stage.syn)

        # Chain stages via pipe()
        from .StreamingTTSProducer import StreamingTTSProducer as _STTS
        for i in range(len(stages) - 1):
            if isinstance(stages[i + 1], _STTS):
                continue  # StreamingTTS reads from its own iterator
            stages[i].pipe(stages[i + 1])

        # Drain the last stage
        last = stages[-1]
        pcm = b""
        sample_rate = 22050  # default
        if last.output_format and last.output_format.sample_rate > 0:
            sample_rate = last.output_format.sample_rate
        for chunk in last.stream_pcm24k():
            pcm += chunk

    except Exception as e:
        return (f"Render error: {e}\n", 400)

    if not pcm:
        return ("Pipeline produced no audio\n", 422)

    # Build WAV
    buf = io.BytesIO()
    w = wave.open(buf, "wb")
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sample_rate)
    w.writeframes(pcm)
    w.close()

    from flask import Response
    return Response(
        buf.getvalue(),
        mimetype="audio/wav",
        headers={"Content-Disposition": "attachment; filename=render.wav"},
    )


@api.route("/pipelines/<pid>/input", methods=["POST"])
@_require_auth
def pipeline_input(pid: str):
    """Feed text into a running pipeline's text_input stage.

    Body: ``{"text": "Hello"}`` to feed text, ``{"eof": true}`` to end.
    """
    p = registry.get(pid)
    if not p:
        return ("Pipeline not found\n", 404)

    q = getattr(p, '_text_input_queue', None)
    if q is None:
        return ("Pipeline has no text_input stage\n", 422)

    body = request.get_json(force=True, silent=True) or {}
    text = body.get("text", "")
    if text:
        q.put(text)
    if body.get("eof"):
        q.put(None)

    return jsonify({"ok": True})


@api.route("/pipelines/render", methods=["POST"])
@_require_auth
def render_pipeline_legacy():
    """Deprecated: use POST /api/pipelines with {"render": true} instead."""
    body = request.get_json(force=True, silent=True) or {}
    dsl = body.get("dsl", "").strip()
    if not dsl:
        return ("Missing 'dsl' in request body\n", 400)
    return _render_pipeline(dsl)


@api.route("/saves/<filename>", methods=["GET"])
def download_save(filename: str):
    """Download a saved recording.

    Files are stored in the managed save directory. No auth required
    because filenames are unguessable tokens.
    """
    import os
    import tempfile
    from flask import send_from_directory

    # Sanitize filename
    filename = os.path.basename(filename)
    save_dir = os.path.join(tempfile.gettempdir(), "speech-pipeline-saves")

    filepath = os.path.join(save_dir, filename)
    if not os.path.isfile(filepath):
        return ("File not found\n", 404)

    return send_from_directory(save_dir, filename, as_attachment=True)


@api.route("/pipelines", methods=["DELETE"])
@_require_auth
def kill_stage_by_dsl():
    """Kill a stage by ID.

    Body: ``{"dsl": "play:hold_music"}`` or ``{"dsl": "bridge:leg-abc"}``

    Searches all active call executors for a stage matching the ID
    and kills it. Returns 204 on success, 404 if not found.
    """
    # Accept the stage ID from EITHER the JSON body or a ``?dsl=`` query
    # parameter.  Some HTTP clients (PHP curl with CURLOPT_CUSTOMREQUEST
    # ``DELETE``) silently drop the body — the query-string fallback
    # saves the request from a 400.
    body = request.get_json(force=True, silent=True) or {}
    stage_id = (body.get("dsl") or request.args.get("dsl") or "").strip()
    if not stage_id:
        return ("Missing 'dsl' (stage ID) in request body or ?dsl= query\n", 400)

    # DELETE accepts only a single stage ID, not a pipeline expression
    if "->" in stage_id or "|" in stage_id:
        return ("DELETE only accepts a single stage ID, not a pipeline\n", 400)

    # Search executors for this stage — scoped to the caller's
    # account (admin sees all).  Without this scope, account B could
    # kill account A's play/bridge stages by guessing the id.
    from .telephony import call_state
    aid = _account_id()
    for call in call_state.list_calls():
        if aid and call.account_id != aid:
            continue
        ex = getattr(call, "pipe_executor", None)
        if ex and ex.kill_stage(stage_id):
            return ("", 204)

    return (f"Stage '{stage_id}' not found\n", 404)


@api.route("/pipelines/<pid>", methods=["DELETE"])
@_require_auth
def delete_pipeline(pid: str):
    p = registry.get(pid)
    if not p:
        return ("Pipeline not found\n", 404)
    p.cancel()
    registry.unregister(pid)
    return ("", 204)


# ---- Stage inspection and manipulation ----

@api.route("/pipelines/<pid>/stages", methods=["GET"])
@_require_auth
def list_stages(pid: str):
    p = registry.get(pid)
    if not p:
        return ("Pipeline not found\n", 404)
    stages = []
    for sid, stage in p.stages.items():
        entry = {
            "id": sid,
            "type": p.stage_types.get(sid, "unknown"),
            "config": p.stage_configs.get(sid, {}),
            "cancelled": stage.cancelled,
        }
        stages.append(entry)
    return jsonify(stages)


@api.route("/pipelines/<pid>/stages/<sid>", methods=["GET"])
@_require_auth
def get_stage(pid: str, sid: str):
    p = registry.get(pid)
    if not p:
        return ("Pipeline not found\n", 404)
    stage = p.get_stage(sid)
    if not stage:
        return ("Stage not found\n", 404)
    entry = {
        "id": sid,
        "type": p.stage_types.get(sid, "unknown"),
        "config": p.stage_configs.get(sid, {}),
        "cancelled": stage.cancelled,
    }
    if stage.output_format:
        entry["output_format"] = {
            "sample_rate": stage.output_format.sample_rate,
            "encoding": stage.output_format.encoding,
        }
    if stage.input_format:
        entry["input_format"] = {
            "sample_rate": stage.input_format.sample_rate,
            "encoding": stage.input_format.encoding,
        }
    return jsonify(entry)


@api.route("/pipelines/<pid>/stages/<sid>", methods=["PATCH"])
@_require_auth
def patch_stage(pid: str, sid: str):
    """Hot-update stage config.  Supported: GainStage (gain), DelayLine (delay_ms)."""
    p = registry.get(pid)
    if not p:
        return ("Pipeline not found\n", 404)
    stage = p.get_stage(sid)
    if not stage:
        return ("Stage not found\n", 404)

    body = request.get_json(force=True, silent=True) or {}
    config = body.get("config", body)
    updated = {}

    # GainStage
    if hasattr(stage, "set_gain") and "gain" in config:
        stage.set_gain(float(config["gain"]))
        updated["gain"] = float(config["gain"])

    # DelayLine
    if hasattr(stage, "set_delay_ms") and "delay_ms" in config:
        stage.set_delay_ms(float(config["delay_ms"]))
        updated["delay_ms"] = float(config["delay_ms"])

    if not updated:
        return ("No hot-updatable config found for this stage type\n", 422)

    # Persist in stage config
    p.stage_configs.setdefault(sid, {}).update(updated)
    return jsonify({"updated": updated})


@api.route("/pipelines/<pid>/stages/<sid>", methods=["DELETE"])
@_require_auth
def delete_stage(pid: str, sid: str):
    """Remove a stage and reconnect its neighbors.

    Only works for processor stages (have both upstream and downstream).
    The upstream is reconnected to the downstream directly.
    """
    p = registry.get(pid)
    if not p:
        return ("Pipeline not found\n", 404)
    stage = p.get_stage(sid)
    if not stage:
        return ("Stage not found\n", 404)
    if not stage.upstream or not stage.downstream:
        return ("Cannot remove source or sink stage\n", 422)

    # Reconnect: upstream.downstream = stage.downstream
    upstream = stage.upstream
    downstream = stage.downstream
    upstream.downstream = downstream
    downstream.upstream = upstream

    # Update edges
    p.edges = [(f, t) for f, t in p.edges if f != sid and t != sid]
    p.edges.append((upstream.id, downstream.id))

    # Remove from registry
    del p.stages[sid]
    p.stage_types.pop(sid, None)
    p.stage_configs.pop(sid, None)

    stage.cancelled = True
    return ("", 204)


@api.route("/pipelines/<pid>/stages/<sid>/replace", methods=["POST"])
@_require_auth
def replace_stage(pid: str, sid: str):
    """Replace a stage with a new one built from a DSL element.

    Body: {"element": "gain:2.0"} or {"element": "stt:de:3.0:large-v3"}

    The stage must be a processor (has upstream and downstream).
    Uses CellRunner for queue-boundary swapping if the stage is already
    wrapped, otherwise does a direct swap (brief interruption).
    """
    p = registry.get(pid)
    if not p:
        return ("Pipeline not found\n", 404)
    stage = p.get_stage(sid)
    if not stage:
        return ("Stage not found\n", 404)

    body = request.get_json(force=True, silent=True) or {}
    element = body.get("element", "").strip()
    if not element:
        return ("Missing 'element' in request body\n", 400)

    # Build the replacement stage using PipelineBuilder.parse + factory
    from .PipelineBuilder import PipelineBuilder
    import argparse

    args = argparse.Namespace(
        whisper_model=body.get("whisper_model", "small"),
        cuda=body.get("cuda", False),
        voices_path=body.get("voices_path", "voices-piper"),
        soundpath=body.get("soundpath", "../voices/%s.wav"),
        bearer=body.get("bearer", ""),
    )
    from .registry import TTSRegistry
    tts_registry = TTSRegistry(args.voices_path, use_cuda=args.cuda)
    builder = PipelineBuilder(ws=None, registry=tts_registry, args=args)

    parsed = builder.parse(element)
    if len(parsed) != 1:
        return ("Element must be a single stage (e.g. 'gain:2.0')\n", 400)

    typ, params = parsed[0]
    new_stage = _build_single_stage(builder, typ, params)
    if new_stage is None:
        return (f"Cannot build stage from '{element}'\n", 400)

    # Wire the new stage in place of the old one
    if stage.upstream:
        new_stage.upstream = stage.upstream
        stage.upstream.downstream = new_stage
    if stage.downstream:
        new_stage.downstream = stage.downstream
        stage.downstream.upstream = new_stage

    # Update pipeline registry
    del p.stages[sid]
    p.stages[new_stage.id] = new_stage
    p.stage_types[new_stage.id] = typ
    p.stage_configs[new_stage.id] = {"params": params}
    p.stage_types.pop(sid, None)
    p.stage_configs.pop(sid, None)

    # Update edges
    new_edges = []
    for f, t in p.edges:
        new_f = new_stage.id if f == sid else f
        new_t = new_stage.id if t == sid else t
        new_edges.append((new_f, new_t))
    p.edges = new_edges

    stage.cancelled = True

    return jsonify({
        "old_stage": sid,
        "new_stage": new_stage.id,
        "type": typ,
    })


def _build_single_stage(builder, typ: str, params: list):
    """Build a single stage from parsed DSL element. Returns Stage or None."""
    try:
        if typ == "resample":
            from .SampleRateConverter import SampleRateConverter
            src = int(params[0]) if len(params) > 0 else 48000
            dst = int(params[1]) if len(params) > 1 else 16000
            return SampleRateConverter(src, dst)
        elif typ == "gain":
            from .GainStage import GainStage
            factor = float(params[0]) if params else 1.0
            return GainStage(16000, factor)
        elif typ == "delay":
            from .DelayLine import DelayLine
            ms = float(params[0]) if params else 0.0
            return DelayLine(16000, ms)
        elif typ == "stt":
            from .WhisperSTT import WhisperTranscriber
            lang = params[0] if params else None
            chunk_seconds = float(params[1]) if len(params) > 1 else 3.0
            model_size = params[2] if len(params) > 2 else "small"
            return WhisperTranscriber(model_size, chunk_seconds=chunk_seconds, language=lang)
        elif typ == "tts":
            from .StreamingTTSProducer import StreamingTTSProducer
            voice_id = params[0] if params else None
            if not voice_id or not builder.registry:
                return None
            voice = builder.registry.ensure_loaded(voice_id)
            syn = builder.registry.create_synthesis_config(voice, {})
            # TTS needs a text source — can't build standalone
            return None
        elif typ == "vc":
            from .VCConverter import VCConverter
            voice2 = params[0] if params else None
            if not voice2:
                return None
            from .FileFetcher import FileFetcher
            from pathlib import Path
            here = Path(__file__).resolve().parent.parent
            tmpl = getattr(builder.args, "soundpath", "../voices/%s.wav")
            ref = FileFetcher.build_ref(voice2, tmpl, here)
            bearer = getattr(builder.args, "bearer", "")
            return VCConverter(ref, bearer=bearer)
        elif typ == "pitch":
            from .PitchAdjuster import PitchAdjuster
            st = float(params[0]) if params else 0.0
            return PitchAdjuster("", pitch_disable=(abs(st) < 0.05), pitch_override_st=st, correction=1.0)
    except Exception as e:
        _LOGGER.warning("Failed to build stage %s: %s", typ, e)
    return None
