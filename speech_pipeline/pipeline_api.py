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
    return jsonify([p.to_dict() for p in registry.list_all()])


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
    """If DSL references call:ID or conference:ID, verify ownership.

    Returns error string on failure, None on success.
    """
    aid = _account_id()
    if not aid:
        return None  # admin — no restriction

    from .telephony import call_state
    for call_id in re.findall(r'(?:call|conference):([^\s{|>]+)', dsl):
        call = call_state.get_call(call_id)
        if not call:
            return f"Call {call_id} not found"
        if call.account_id != aid:
            return f"Forbidden: call {call_id} belongs to another account"
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

    # Find call context from DSL (if any call/conference element present)
    call = None
    for typ, elem_id, _ in elements:
        if typ in ("call", "conference") and elem_id:
            from .telephony import call_state
            call = call_state.get_call(elem_id)
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

    try:
        results = executor.add_pipes([dsl])
    except Exception as e:
        return (f"Build error: {e}\n", 400)

    # Check for per-pipe errors
    for r in results:
        if not r.get("ok"):
            return (f"Pipe error: {r.get('error', 'unknown')}\n", 400)

    # Propagate text_input queue
    if executor._text_input_queue:
        pipeline._text_input_queue = executor._text_input_queue

    pipeline.state = "running"
    pipeline._executor = executor
    registry.register(pipeline)

    return jsonify(pipeline.to_dict()), 201


@api.route("/pipelines/render", methods=["POST"])
@_require_auth
def render_pipeline():
    """Render a pipeline synchronously and return the audio as WAV.

    Body: ``{"dsl": "tts:de{\"text\":\"Hallo\"} | pitch:2.0"}``

    The pipeline must be a pure audio chain (no sip, call, conference,
    webhook). The output is collected in memory and returned as a WAV
    file with Content-Type audio/wav.
    """
    body = request.get_json(force=True, silent=True) or {}
    dsl = body.get("dsl", "").strip()
    if not dsl:
        return ("Missing 'dsl' in request body\n", 400)

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
