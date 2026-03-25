"""DSL-first pipe executor for telephony calls.

Parses DSL pipes like::

    sip:leg-abc{"completed":"/hook"} -> tee:tap-abc -> call:xyz -> sip:leg-abc
    tee:tap-abc -> stt:de{"on_receive":"https://crm/stt?p=123"}

Elements: ``type:id{json_params}``
Separator: ``->`` (or ``|`` for backward compat)

Wiring rules:
- Output: call:/tee: are multi-output (add new output), else redirect
- Input: call: is multi-input (add new input), else redirect
- call: always creates a ConferenceLeg
"""
from __future__ import annotations

import json
import logging
import secrets
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

_LOGGER = logging.getLogger("telephony.pipe-executor")


# ---------------------------------------------------------------------------
# DSL Parser
# ---------------------------------------------------------------------------

def parse_dsl(dsl: str) -> List[Tuple[str, str, dict]]:
    """Parse a DSL pipe string into elements.

    Returns list of (type, id, params_dict) tuples.

    Example::

        >>> parse_dsl('sip:leg-abc{"completed":"/hook"} -> call:xyz')
        [('sip', 'leg-abc', {'completed': '/hook'}), ('call', 'xyz', {})]
    """
    elements = []
    pos = 0
    s = dsl.strip()

    while pos < len(s):
        # Skip whitespace and separators
        while pos < len(s) and s[pos] in ' \t':
            pos += 1
        if pos >= len(s):
            break
        if s[pos:pos+2] == '->':
            pos += 2
            continue
        if s[pos] == '|':
            pos += 1
            continue

        # Read type
        start = pos
        while pos < len(s) and s[pos] not in ':{| \t':
            if s[pos] == '-' and pos + 1 < len(s) and s[pos+1] == '>':
                break
            pos += 1
        typ = s[start:pos].strip()

        # Read id (after :)
        elem_id = ""
        if pos < len(s) and s[pos] == ':':
            pos += 1
            start = pos
            while pos < len(s) and s[pos] not in '{| \t':
                if s[pos] == '-' and pos + 1 < len(s) and s[pos+1] == '>':
                    break
                pos += 1
            elem_id = s[start:pos].strip()

        # Read optional JSON params
        params = {}
        if pos < len(s) and s[pos] == '{':
            json_start = pos
            depth = 0
            in_string = False
            escape = False
            while pos < len(s):
                ch = s[pos]
                if escape:
                    escape = False
                elif ch == '\\' and in_string:
                    escape = True
                elif ch == '"' and not escape:
                    in_string = not in_string
                elif not in_string:
                    if ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            pos += 1
                            break
                pos += 1
            json_str = s[json_start:pos]
            try:
                params = json.loads(json_str)
            except json.JSONDecodeError as e:
                _LOGGER.warning("Failed to parse JSON in DSL: %s", e)

        elements.append((typ, elem_id, params))

    return elements


# ---------------------------------------------------------------------------
# Pipe Executor
# ---------------------------------------------------------------------------

class CallPipeExecutor:
    """Manages DSL pipe execution for a single Call.

    Maintains a registry of named stage instances. Each ``add_pipes()``
    call parses DSL strings, resolves or creates stages, wires them
    according to the DSL wiring rules, and starts pipelines.
    """

    def __init__(self, call, tts_registry=None, subscriber=None):
        self.call = call
        self._tts_registry = tts_registry
        self._subscriber = subscriber
        self._stages: Dict[str, Any] = {}  # "type:id" -> stage object
        self._lock = threading.Lock()
        self._threads: List[threading.Thread] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_pipes(self, pipes: List[str]) -> List[dict]:
        """Parse and execute DSL pipe strings. Returns stage info."""
        results = []
        for pipe_str in pipes:
            try:
                elements = parse_dsl(pipe_str)
                self._execute_pipe(elements)
                results.append({"pipe": pipe_str, "ok": True})
            except Exception as e:
                _LOGGER.warning("Pipe failed: %s — %s", pipe_str, e, exc_info=True)
                results.append({"pipe": pipe_str, "ok": False, "error": str(e)})
        return results

    def kill_stage(self, stage_id: str) -> bool:
        """Cancel a named stage and clean up."""
        with self._lock:
            entry = self._stages.pop(stage_id, None)
        if not entry:
            return False

        if hasattr(entry, 'cancel'):
            try:
                entry.cancel()
            except Exception:
                pass
        # For mixer sources (play, tts): kill_source
        src_id = getattr(entry, '_src_id', None)
        if src_id:
            try:
                self.call.mixer.kill_source(src_id)
            except Exception:
                pass
        _LOGGER.info("Killed stage %s", stage_id)
        return True

    def list_stages(self) -> List[dict]:
        """Return info about all live stages."""
        with self._lock:
            return [{"id": k, "type": type(v).__name__}
                    for k, v in self._stages.items()]

    # ------------------------------------------------------------------
    # Stage resolution
    # ------------------------------------------------------------------

    def _key(self, typ: str, elem_id: str) -> str:
        return f"{typ}:{elem_id}" if elem_id else typ

    def _get_stage(self, key: str):
        with self._lock:
            return self._stages.get(key)

    def _put_stage(self, key: str, stage):
        with self._lock:
            self._stages[key] = stage

    def _resolve(self, typ: str, elem_id: str, params: dict,
                 is_first: bool, is_last: bool):
        """Get or create a stage. Returns (stage, is_new)."""
        key = self._key(typ, elem_id)
        existing = self._get_stage(key)
        if existing is not None:
            return existing, False

        stage = self._create_stage(typ, elem_id, params, is_first, is_last)
        if stage is not None:
            self._put_stage(key, stage)
        return stage, True

    def _create_stage(self, typ: str, elem_id: str, params: dict,
                      is_first: bool, is_last: bool):
        """Create a new stage instance by type."""

        if typ == "sip":
            # Look up existing leg session
            from . import leg as leg_mod
            leg = leg_mod.get_leg(elem_id)
            if not leg:
                raise ValueError(f"Leg {elem_id} not found")
            if not leg.voip_call:
                raise ValueError(f"Leg {elem_id} has no SIP call")
            # Mark leg as bridged (sip_listener._wait_for_bridge checks this)
            leg.call_id = self.call.call_id
            leg.status = "in-progress"
            leg.answered_at = leg.answered_at or __import__('time').time()
            session = leg_mod._CallSession(leg.voip_call)
            # Attach webhooks from params
            if params:
                leg.callbacks.update(params)
            return session  # SIPSource/SIPSink will wrap this

        elif typ == "call":
            # Conference — returns mixer reference (ConferenceLeg wraps it)
            return self.call.mixer

        elif typ == "tee":
            from speech_pipeline.AudioTee import AudioTee
            tee = AudioTee(48000, "s16le")
            return tee

        elif typ == "stt":
            lang = elem_id or "de"
            from speech_pipeline.WhisperSTT import WhisperTranscriber
            stt = WhisperTranscriber(
                model_name=getattr(self, '_stt_model', 'base'),
                language=lang)
            # Attach on_receive webhook from params
            if params and params.get("on_receive"):
                stt._webhook_url = params["on_receive"]
            return stt

        elif typ == "tts":
            if not self._tts_registry:
                raise ValueError("TTS registry not available")
            voice_name = elem_id or "de_DE-thorsten-medium"
            text = params.get("text", "")
            if not text:
                raise ValueError("tts requires text param")
            from speech_pipeline.TTSProducer import TTSProducer
            voice = self._tts_registry.ensure_loaded(voice_name)
            syn = self._tts_registry.create_synthesis_config(voice, {})
            return TTSProducer(voice, syn, text, sentence_silence=0.0)

        elif typ == "play":
            url = elem_id
            loop = params.get("loop", False)
            volume = params.get("volume", 100)
            if not url:
                raise ValueError("play requires URL")
            from speech_pipeline.AudioReader import AudioReader
            return AudioReader(url, loop=loop, volume=volume)

        elif typ == "webhook":
            url = elem_id
            if not url:
                raise ValueError("webhook requires URL")
            from speech_pipeline.WebhookSink import WebhookSink
            bearer = ""
            if self._subscriber:
                bearer = self._subscriber.get("bearer_token", "")
            return WebhookSink(url, bearer_token=bearer)

        else:
            raise ValueError(f"Unknown stage type: {typ}")

    # ------------------------------------------------------------------
    # Pipe execution (wiring)
    # ------------------------------------------------------------------

    def _execute_pipe(self, elements: List[Tuple[str, str, dict]]):
        """Wire a parsed DSL pipe."""
        if not elements:
            return

        n = len(elements)
        stages = []  # (typ, elem_id, params, resolved_obj, is_new)

        # Phase 1: resolve all elements
        for i, (typ, elem_id, params) in enumerate(elements):
            is_first = (i == 0)
            is_last = (i == n - 1)
            obj, is_new = self._resolve(typ, elem_id, params, is_first, is_last)
            stages.append((typ, elem_id, params, obj, is_new))

        # Phase 2: build pipeline chain
        # Walk left to right, creating Stage instances and piping them
        from speech_pipeline.SIPSource import SIPSource
        from speech_pipeline.SIPSink import SIPSink
        from speech_pipeline.ConferenceLeg import ConferenceLeg
        from speech_pipeline.ConferenceMixer import ConferenceMixer

        pipeline_stages = []  # actual Stage objects in pipe order
        terminal_sink = None
        sip_leg_ids = []  # leg IDs for BYE on pipeline end

        prev_stage = None  # last Stage in the chain

        for i, (typ, elem_id, params, obj, is_new) in enumerate(stages):
            is_first = (i == 0)
            is_last = (i == n - 1)

            if typ == "sip":
                # SIP session — source if first, sink if last, both if both
                sip_leg_ids.append(elem_id)
                if is_first:
                    src = SIPSource(obj)
                    pipeline_stages.append(src)
                    prev_stage = src
                if is_last and prev_stage:
                    sink = SIPSink(obj)
                    prev_stage.pipe(sink)
                    pipeline_stages.append(sink)
                    terminal_sink = sink
                    prev_stage = None

            elif typ == "call":
                # Conference mixer — wiring depends on position:
                # source -> call (last) = add_source (unidirectional input)
                # call -> sink (first) = add_output (unidirectional output)
                # source -> call -> sink = ConferenceLeg (bidirectional)
                mixer = obj
                has_input = prev_stage is not None
                has_output = not is_last

                if has_input and has_output:
                    # Bidirectional: add_source (input) + add_sink (output)
                    # Same pattern as e2e97ca bridge_to_call — proven to work.
                    # pipe() auto-inserts format converters (8k↔48k etc.)
                    src_id = mixer.add_source(prev_stage)
                    if pipeline_stages:
                        pipeline_stages[0]._src_id = src_id
                    # Output: next element will be piped from a QueueSource
                    from speech_pipeline.QueueSource import QueueSource
                    out_q = mixer.add_output(mute_source=src_id)
                    qs = QueueSource(out_q, mixer.sample_rate, "s16le")
                    pipeline_stages.append(qs)
                    prev_stage = qs

                elif has_input and not has_output:
                    # Source-only: pipe into mixer via add_source
                    src_id = mixer.add_source(prev_stage)
                    # Store src_id on the source for kill_stage
                    if pipeline_stages:
                        pipeline_stages[0]._src_id = src_id
                    # Block until source is consumed, then clean up
                    def _make_wait(m, s):
                        def _wait():
                            m.wait_source(s)
                            m.remove_source(s)
                        return _wait
                    terminal_sink = type('_Waiter', (), {
                        'run': _make_wait(mixer, src_id)})()
                    prev_stage = None

                elif not has_input and has_output:
                    # Output-only: read from mixer output queue
                    from speech_pipeline.QueueSource import QueueSource
                    out_q = mixer.add_output()
                    src = QueueSource(out_q, mixer.sample_rate, "s16le")
                    pipeline_stages.append(src)
                    prev_stage = src

            elif typ == "tee":
                # AudioTee — pass-through with sidechain outputs
                tee = obj
                if prev_stage:
                    prev_stage.pipe(tee)
                pipeline_stages.append(tee)
                prev_stage = tee

                # If tee is first element (referencing existing tee),
                # add a new sidechain output
                if is_first and not is_new:
                    # Remaining elements form a sidechain
                    self._build_sidechain(tee, stages[i+1:])
                    return  # sidechain handled, don't continue main chain

            elif typ == "stt":
                # STT processor
                stt = obj
                if prev_stage:
                    prev_stage.pipe(stt)
                pipeline_stages.append(stt)
                prev_stage = stt

            elif typ == "tts":
                # TTS source (always first)
                pipeline_stages.append(obj)
                prev_stage = obj

            elif typ == "play":
                # Play source (always first)
                pipeline_stages.append(obj)
                prev_stage = obj

            elif typ == "webhook":
                # Webhook sink (always last)
                sink = obj
                if prev_stage:
                    prev_stage.pipe(sink)
                pipeline_stages.append(sink)
                terminal_sink = sink
                prev_stage = None

            else:
                _LOGGER.warning("Unhandled element type: %s", typ)

        # Phase 3: start the pipeline
        if terminal_sink and hasattr(terminal_sink, 'run'):
            self._start_pipeline(terminal_sink, pipeline_stages,
                                 sip_legs=sip_leg_ids)
        elif pipeline_stages:
            # No terminal sink — source-only pipe (e.g., tts -> conference)
            # ConferenceLeg or add_source handles the run
            last = pipeline_stages[-1]
            if isinstance(last, ConferenceLeg):
                # ConferenceLeg as last = source-only into conference
                # Need SIPSink or something to drive it... actually
                # ConferenceLeg.stream_pcm24k drives itself via pump thread
                # But it needs a downstream sink to iterate its output.
                # Source-only: use add_source instead
                pass
            # For play/tts -> conference: the conference element handles it
            # via add_source in the ConferenceLeg path

    def _build_sidechain(self, tee, remaining_elements):
        """Build a sidechain pipe from an existing tee.

        tee:uuid -> stt:de -> webhook:url
        The tee already exists. We add a new sidechain sink chain.
        """
        from speech_pipeline.base import Stage

        # Build the downstream chain
        prev = None
        chain = []
        terminal = None

        for typ, elem_id, params, obj, is_new in remaining_elements:
            stage = obj
            if isinstance(stage, Stage):
                if prev:
                    prev.pipe(stage)
                chain.append(stage)
                prev = stage
                if hasattr(stage, 'run'):
                    terminal = stage

        if chain:
            # Add the first stage as a sidechain of the tee
            tee.add_sidechain(chain[0])
            _LOGGER.info("Added sidechain to tee: %s",
                         " -> ".join(type(s).__name__ for s in chain))

            # If there's a terminal sink, the sidechain thread handles it
            # via AudioTee._run_sink. No extra thread needed.

    def _start_pipeline(self, terminal_sink, stages, sip_legs=None):
        """Run the terminal sink in a background thread.

        When the pipeline ends (SIP hangup, conference deleted, etc.),
        sends BYE to all SIP legs and fires their callbacks.
        """
        def _run():
            try:
                terminal_sink.run()
            except Exception as e:
                _LOGGER.warning("Pipeline error: %s", e)
            finally:
                _LOGGER.info("Pipeline ended: %s",
                             " -> ".join(type(s).__name__ for s in stages))
                # Hangup all SIP legs used in this pipeline
                if sip_legs:
                    from . import leg as leg_mod
                    for leg_id in sip_legs:
                        leg = leg_mod.get_leg(leg_id)
                        if leg and leg.status != "completed":
                            leg.hangup()
                            leg.status = "completed"
                            duration = (time.time() - leg.answered_at
                                        if leg.answered_at else 0)
                            leg_mod._fire_callback(
                                leg, "completed", duration=duration)
                            _LOGGER.info("Leg %s ended (duration=%.1fs)",
                                         leg_id, duration)
                            leg_mod.delete_leg(leg_id)

        t = threading.Thread(target=_run, daemon=True,
                             name=f"pipe-{secrets.token_urlsafe(4)}")
        t.start()
        self._threads.append(t)
