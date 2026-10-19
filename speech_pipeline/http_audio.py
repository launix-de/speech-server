"""Attach an expiring HTTP capability to a caller-defined audio pipeline."""
from __future__ import annotations

import math
import secrets
import threading

from flask import Response, jsonify, url_for

from . import live_pipeline as registry
from .base import Stage
from .ResponseWriter import ResponseWriter

_lock = threading.Lock()
_pending = {}


def prepare_audio(dsl, options, account_id):
    from .dsl_parser import parse_dsl
    from .telephony.pipe_executor import CallPipeExecutor
    from .telephony import _shared

    stages = []
    sink = None
    pipeline = registry.LivePipeline(dsl)
    pipeline.account_id = account_id
    try:
        if not isinstance(options, dict):
            raise ValueError("http must be an object")
        buffered = options.get("buffered", False)
        if not isinstance(buffered, bool):
            raise ValueError("http.buffered must be boolean")
        timeout = float(options.get("timeout", 180))
        if not math.isfinite(timeout) or not 0 < timeout <= 600:
            raise ValueError("http.timeout must be greater than 0 and at most 600 seconds")
        elements = parse_dsl(dsl)
        if not elements:
            raise ValueError("Empty pipeline")
        # These factory elements produce composable stages without starting
        # workers or call-control actions. Wiring always comes from the caller.
        supported = {"tts", "play", "vc", "pitch", "gain", "delay", "call", "conference"}
        if any(typ not in supported for typ, _, _ in elements):
            raise ValueError("HTTP output requires audio source/processor stages")
        executor = CallPipeExecutor(call=None, tts_registry=_shared.tts_registry,
                                    account_id=account_id)
        for typ, elem_id, params in elements:
            stage = executor._create_stage(typ, elem_id, params)
            if not isinstance(stage, Stage) or getattr(stage, "_streaming", False):
                raise ValueError("HTTP output requires a concrete audio stage")
            stages.append(stage)
            pipeline.add_stage(stage, typ, params)
            if len(stages) > 1:
                stages[-2].pipe(stage)
                pipeline.add_edge(stages[-2].id, stage.id)
        fmt = stages[-1].output_format
        if fmt is None or fmt.encoding != "s16le":
            raise ValueError("HTTP WAV output requires s16le audio")
        sink = ResponseWriter(timeout=timeout, buffered=buffered, estimated_length=False)
        pipeline._http_sink = sink
        pipeline.state = "waiting"
        token = secrets.token_urlsafe(32)

        def cleanup():
            with _lock:
                _pending.pop(token, None)
            pipeline.state = "stopped"
            registry.unregister(pipeline.id)

        sink.add_close_callback(cleanup)
        registry.register(pipeline)
        with _lock:
            _pending[token] = (pipeline, sink)
        stages[-1].pipe(sink)
        return jsonify(id=pipeline.id, audio_url=url_for("pipeline_api.consume_audio", token=token),
                       timeout=timeout), 201
    except Exception as exc:
        if sink is not None:
            sink.cancel()
        for stage in reversed(stages):
            stage.close()
        registry.unregister(pipeline.id)
        return (str(exc) + "\n", 400)


def consume_audio(token):
    with _lock:
        item = _pending.pop(token, None)
    if item is None:
        return ("Audio endpoint expired or already consumed\n", 404)
    pipeline, sink = item
    if sink.closed:
        return ("Audio endpoint expired\n", 410)
    pipeline.state = "running"
    response = Response(sink.stream_pcm24k(), mimetype="audio/wav")
    sink.apply_headers(response)
    response.headers["Cache-Control"] = "no-store"
    response.call_on_close(sink.cancel)
    return response
