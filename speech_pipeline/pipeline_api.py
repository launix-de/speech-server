"""Flask blueprint for pipeline control API.

All endpoints require ``Authorization: Bearer <admin-token>``.
"""
from __future__ import annotations

import functools
import json
import logging
from typing import Optional

from flask import Blueprint, Response, jsonify, request

from . import live_pipeline as registry

_LOGGER = logging.getLogger("pipeline-api")

api = Blueprint("pipeline_api", __name__, url_prefix="/api")

_admin_token: Optional[str] = None


def init(admin_token: str) -> None:
    global _admin_token
    _admin_token = admin_token


def _require_auth(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        if not _admin_token:
            return ("Pipeline API disabled (no --admin-token set)\n", 403)
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return ("Missing Authorization: Bearer <token>\n", 401)
        token = auth[7:]
        if token != _admin_token:
            return ("Invalid token\n", 403)
        return f(*args, **kwargs)
    return wrapper


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


@api.route("/pipelines", methods=["POST"])
@_require_auth
def create_pipeline():
    """Create a pipeline from DSL.  Body: {"dsl": "cli:text | tts:voice | cli:raw"}"""
    body = request.get_json(force=True, silent=True) or {}
    dsl = body.get("dsl", "").strip()
    if not dsl:
        return ("Missing 'dsl' in request body\n", 400)

    from .PipelineBuilder import PipelineBuilder

    # Use a minimal args namespace for CLI-created pipelines
    import argparse
    args = argparse.Namespace(
        whisper_model=body.get("whisper_model", "small"),
        cuda=body.get("cuda", False),
        voices_path=body.get("voices_path", "voices-piper"),
        soundpath=body.get("soundpath", "../voices/%s.wav"),
        bearer=body.get("bearer", ""),
    )

    # Registry is needed for TTS voices
    from .registry import TTSRegistry
    tts_registry = TTSRegistry(args.voices_path, use_cuda=args.cuda)

    pipeline = registry.LivePipeline(dsl=dsl)
    builder = PipelineBuilder(ws=None, registry=tts_registry, args=args, live_pipeline=pipeline)

    # Inject conference mixers so DSL can reference live conferences
    from .PipelineBuilder import inject_conference_mixers
    inject_conference_mixers(builder, dsl)

    try:
        run = builder.build(dsl)
    except Exception as e:
        return (f"Build error: {e}\n", 400)

    # Propagate text_input queue to LivePipeline for API access
    if hasattr(builder, '_text_input_queue'):
        pipeline._text_input_queue = builder._text_input_queue

    pipeline.run = run
    pipeline.state = "running"
    registry.register(pipeline)

    # Run in background thread
    import threading
    def _run():
        try:
            run.run()
        except Exception as e:
            _LOGGER.warning("Pipeline %s error: %s", pipeline.id, e)
        finally:
            pipeline.state = "stopped"

    t = threading.Thread(target=_run, daemon=True, name=f"pipeline-{pipeline.id}")
    t.start()

    return jsonify(pipeline.to_dict(detail=True)), 201


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
